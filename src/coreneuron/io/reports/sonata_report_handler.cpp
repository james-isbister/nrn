/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================
*/

#include <algorithm>
#include <cstdlib>

#include "sonata_report_handler.hpp"
#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/mpi/nrnmpi.h"
#include "coreneuron/network/netcvode.hpp"
#include "coreneuron/network/netcon.hpp"
#include "coreneuron/io/nrnsection_mapping.hpp"
#include "coreneuron/mechanism/mech_mapping.hpp"
#ifdef ENABLE_SONATA_REPORTS
#include "bbp/sonata/reports.h"
#endif  // ENABLE_SONATA_REPORTS

namespace coreneuron {

void SonataReportHandler::create_report(ReportConfiguration& report_conf,
                                        double dt,
                                        double tstop,
                                        double delay) {
#ifdef ENABLE_SONATA_REPORTS
    sonata_set_atomic_step(dt);
#endif  // ENABLE_SONATA_REPORTS
    ReportHandler::create_report(report_conf, dt, tstop, delay);
}

#ifdef ENABLE_SONATA_REPORTS
void SonataReportHandler::register_section_report(const NrnThread& nt,
                                                  const ReportConfiguration& report_conf,
                                                  const VarsToReport& vars_to_report,
                                                  bool is_soma_target) {
    register_report(nt, report_conf, vars_to_report);
}

void SonataReportHandler::register_custom_report(const NrnThread& nt,
                                                 const ReportConfiguration& report_conf,
                                                 const VarsToReport& vars_to_report) {
    register_report(nt, report_conf, vars_to_report);
}

std::pair<std::string, int> SonataReportHandler::get_population_info(int gid) {
    if (m_spikes_info.population_info.empty()) {
        return std::make_pair("All", 0);
    }
    std::pair<std::string, int> prev = m_spikes_info.population_info.front();
    for (const auto& name_offset: m_spikes_info.population_info) {
        std::string pop_name = name_offset.first;
        int pop_offset = name_offset.second;
        if (pop_offset > gid) {
            break;
        }
        prev = name_offset;
    }
    return prev;
}

void SonataReportHandler::register_report(const NrnThread& nt,
                                          const ReportConfiguration& report_conf,
                                          const VarsToReport& vars_to_report) {
    const auto type_str = to_string(report_conf.type);
    sonata_create_report(report_conf.output_path.data(),
                         report_conf.start,
                         report_conf.stop,
                         report_conf.report_dt,
                         report_conf.unit.data(),
                         type_str.data());
    sonata_set_report_max_buffer_size_hint(report_conf.output_path.data(), report_conf.buffer_size);

    for (const auto& kv: vars_to_report) {
        uint64_t gid = kv.first;
        const std::vector<VarWithMapping>& vars = kv.second;
        if (!vars.size())
            continue;

        const auto& pop_info = get_population_info(gid);
        std::string population_name = pop_info.first;
        int population_offset = pop_info.second;
        sonata_add_node(report_conf.output_path.data(),
                        population_name.data(),
                        population_offset,
                        gid);
        sonata_set_report_max_buffer_size_hint(report_conf.output_path.data(),
                                               report_conf.buffer_size);
        for (const auto& variable: vars) {
            sonata_add_element(report_conf.output_path.data(),
                               population_name.data(),
                               gid,
                               variable.id,
                               variable.var_value);
        }
    }
}

/** @brief Sets up an LFP report with method "sum": for each population, one trace per electrode.
 *
 * Called on every rank, since every rank takes part in the sum (see LFPSumReport). Rank 0 writes
 * each population's sum as its node 0, which stands for the population rather than for that cell.
 */
void SonataReportHandler::create_lfp_sum_report(ReportConfiguration& config, double dt) {
    // the report's populations, from its target, which every rank has in full
    std::vector<std::pair<std::string, int>> populations;
    for (const int gid: config.target) {
        const auto population = get_population_info(gid);
        if (std::find(populations.begin(), populations.end(), population) == populations.end()) {
            populations.push_back(population);
        }
    }
    const auto population_index = [&](int gid) -> size_t {
        const auto population = get_population_info(gid);
        return std::find(populations.begin(), populations.end(), population) - populations.begin();
    };

    // this rank's cells with electrodes in this report, and the number of electrodes of each
    // population, which every cell of the population with electrodes has
    struct ReportCell {
        const CellMapping* mapping;
        size_t population;
        size_t start;
        size_t end;
    };
    std::vector<std::vector<ReportCell>> thread_cells(nrn_nthread);
    std::vector<long> num_electrodes(populations.size(), 0);
    for (int ith = 0; ith < nrn_nthread; ++ith) {
        const NrnThread& nt = nrn_threads[ith];
        if (!nt.ncell) {
            continue;
        }
        const auto* mapinfo = static_cast<NrnThreadMappingInfo*>(nt.mapping);
        if (!mapinfo) {
            std::cerr << "[ERROR] : LFP report mapping information is missing for a Cell group "
                      << nt.ncell << '\n';
            nrn_abort(1);
        }
        for (const auto intersection_id: get_intersection_ids(nt, config.target)) {
            const auto gid = config.target[intersection_id];
            const auto& cell_mapping = mapinfo->get_cell_mapping(gid);
            if (cell_mapping == nullptr) {
                std::cerr << "[ERROR] : LFP report mapping information is missing for gid " << gid
                          << '\n';
                nrn_abort(1);
            }
            const auto [start, end] = cell_mapping->report_electrodes(config.lfp_report_index);
            if (start == end) {
                continue;
            }
            const size_t population = population_index(gid);
            num_electrodes[population] = std::max(num_electrodes[population],
                                                  static_cast<long>(end - start));
            thread_cells[ith].push_back({cell_mapping.get(), population, start, end});
        }
    }
#if NRNMPI
    if (corenrn_param.mpi_enable) {
        std::vector<long> local_num_electrodes(num_electrodes);
        int mpi_max{2};
        nrnmpi_long_allreduce_vec(local_num_electrodes.data(),
                                  num_electrodes.data(),
                                  static_cast<int>(num_electrodes.size()),
                                  mpi_max);
    }
#endif

    // a row holds the populations' electrodes one population after the other
    std::vector<size_t> columns(populations.size());
    size_t width = 0;
    for (size_t i = 0; i < populations.size(); i++) {
        columns[i] = width;
        width += num_electrodes[i];
    }
    if (width == 0) {
        if (nrnmpi_myid == 0) {
            std::cerr << "[WARNING] : No cell of LFP report '" << config.output_path
                      << "' has electrodes, so it is not written.\n";
        }
        return;
    }

    auto report = std::make_shared<LFPSumReport>(config.output_path, width);
    for (int ith = 0; ith < nrn_nthread; ++ith) {
        std::vector<LFPSumEvent::Cell> cells;
        for (const auto& cell: thread_cells[ith]) {
            const size_t count = cell.end - cell.start;
            if (count != static_cast<size_t>(num_electrodes[cell.population])) {
                std::cerr << "[ERROR] : In LFP report '" << config.output_path << "', gid "
                          << cell.mapping->gid << " has " << count << " electrodes but others of "
                          << "population " << populations[cell.population].first << " have "
                          << num_electrodes[cell.population] << '\n';
                nrn_abort(1);
            }
            cells.push_back({cell.mapping, columns[cell.population], cell.start, count});
        }
        // thread 0 counts the reporting steps, with or without cells
        if (ith == 0 || !cells.empty()) {
            NrnThread& nt = nrn_threads[ith];
            auto event =
                std::make_unique<LFPSumEvent>(report, std::move(cells), ith == 0, config, dt, t);
            event->send(t, net_cvode_instance, &nt);
            m_lfp_sum_events.push_back(std::move(event));
        }
    }

    if (nrnmpi_myid == 0) {
        const auto type_str = to_string(config.type);
        sonata_create_report(config.output_path.data(),
                             config.start,
                             config.stop,
                             config.report_dt,
                             config.unit.data(),
                             type_str.data());
        sonata_set_report_max_buffer_size_hint(config.output_path.data(), config.buffer_size);
        // libsonatareport takes one more off the gids it is given when they are one-based
        const int node_zero = std::getenv("LIBSONATAREPORT_ONE_BASED_GIDS") ? 1 : 0;
        std::vector<int> node_gids;
        for (size_t i = 0; i < populations.size(); i++) {
            if (num_electrodes[i] == 0) {
                continue;
            }
            const auto& [population_name, population_offset] = populations[i];
            const int gid = population_offset + node_zero;
            sonata_add_node(config.output_path.data(),
                            population_name.data(),
                            population_offset,
                            gid);
            for (long electrode = 0; electrode < num_electrodes[i]; electrode++) {
                sonata_add_element(config.output_path.data(),
                                   population_name.data(),
                                   gid,
                                   static_cast<uint32_t>(electrode),
                                   report->values() + columns[i] + electrode);
            }
            node_gids.push_back(gid);
        }
        report->set_nodes(std::move(node_gids));
    }
    register_lfp_sum_report(std::move(report));
}
#endif  // ENABLE_SONATA_REPORTS
}  // Namespace coreneuron
