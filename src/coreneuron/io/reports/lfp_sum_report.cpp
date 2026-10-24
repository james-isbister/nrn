/*
# =============================================================================
# See top-level LICENSE file for details.
# =============================================================================
*/

#include "lfp_sum_report.hpp"

#include <algorithm>

#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/io/nrnsection_mapping.hpp"
#include "coreneuron/mpi/nrnmpi.h"
#include "coreneuron/sim/multicore.hpp"
#ifdef ENABLE_SONATA_REPORTS
#include "bbp/sonata/reports.h"
#endif

namespace coreneuron {

#ifdef ENABLE_SONATA_REPORTS
static std::vector<std::shared_ptr<LFPSumReport>> lfp_sum_reports;

LFPSumReport::LFPSumReport(std::string output_path, size_t width)
    : output_path_(std::move(output_path))
    , width_(width)
    , values_(width) {}

double* LFPSumReport::row(size_t row_index) {
    const size_t end = (row_index + 1) * width_;
    if (rows_.size() < end) {
        rows_.resize(end, 0.0);
    }
    return rows_.data() + row_index * width_;
}

void LFPSumReport::add_step(double step) {
    steps_.push_back(step);
}

void LFPSumReport::set_nodes(std::vector<int> node_gids) {
    node_gids_ = std::move(node_gids);
}

void LFPSumReport::flush() {
    const size_t num_rows = steps_.size();
    if (num_rows == 0) {
        return;
    }
    const size_t count = num_rows * width_;
    // a rank without any of the report's cells has never touched a row
    row(num_rows - 1);
    totals_.resize(count);
#if NRNMPI
    if (corenrn_param.mpi_enable) {
        int mpi_sum{1};
        nrnmpi_dbl_allreduce_vec(rows_.data(), totals_.data(), static_cast<int>(count), mpi_sum);
    } else
#endif
    {
        std::copy_n(rows_.begin(), count, totals_.begin());
    }
    for (size_t i = 0; i < num_rows && !node_gids_.empty(); i++) {
        std::copy_n(totals_.begin() + i * width_, width_, values_.begin());
        sonata_record_node_data(steps_[i],
                                static_cast<int>(node_gids_.size()),
                                node_gids_.data(),
                                output_path_.data());
    }
    std::fill(rows_.begin(), rows_.end(), 0.0);
    rows_flushed_ += num_rows;
    steps_.clear();
}

LFPSumEvent::LFPSumEvent(std::shared_ptr<LFPSumReport> report,
                         std::vector<Cell> cells,
                         bool counts_steps,
                         const ReportConfiguration& config,
                         double dt,
                         double tstart)
    : report_(std::move(report))
    , cells_(std::move(cells))
    , counts_steps_(counts_steps)
    , report_path_(config.output_path)
    , dt_(dt)
    , step_(tstart / dt)
    , first_step_(config.start / dt)
    , last_step_(config.stop / dt)
    , reporting_period_(static_cast<int>(config.report_dt / dt)) {}

/// The same test as libsonatareport's SonataData::is_due_to_report, with the times it is given
bool LFPSumEvent::is_due(double step) const noexcept {
    return step >= first_step_ && step <= last_step_ &&
           static_cast<int>(step - first_step_) % reporting_period_ == 0;
}

/** @brief Adds the contributions of the thread's cells to each electrode.
 *
 * The same computation as ReportEvent::lfp_calc, restricted to the report's electrodes: the total
 * membrane current of every segment (imem + iclamp) weighted by its electrode factors.
 */
void LFPSumEvent::add_lfp(NrnThread* nt, double* row) const {
    if (cells_.empty()) {
        return;
    }
    const double* fast_imem_rhs = nt->nrn_fast_imem->nrn_sav_rhs;
    const auto& currents = nt->summation_report_handler_->summation_reports_[report_path_].currents_;
    for (const auto& cell: cells_) {
        const auto& segment_ids = cell.mapping->lfp_segment_ids;
        const size_t stride = cell.mapping->num_electrodes();
        double* lfp = row + cell.column;
        for (size_t i = 0; i < segment_ids.size(); i++) {
            const auto segment_id = segment_ids[i];
            double imem = fast_imem_rhs[segment_id];
            const auto it = currents.find(segment_id);
            if (it != currents.end()) {
                for (const auto& [current, scale]: it->second) {
                    imem += *current * scale;
                }
            }
            const double* factors = &cell.mapping->lfp_factors_flat[i * stride +
                                                                    cell.first_electrode];
            for (size_t e = 0; e < cell.num_electrodes; e++) {
                lfp[e] += imem * factors[e];
            }
        }
    }
}

void LFPSumEvent::deliver(double t, NetCvode* nc, NrnThread* nt) {
    // the threads of a rank add to the same rows
#pragma omp critical
    {
        if (is_due(step_)) {
            add_lfp(nt, report_->row(steps_seen_ - report_->rows_flushed()));
            if (counts_steps_) {
                report_->add_step(step_);
            }
            steps_seen_++;
        }
        send(t + dt_, nc, nt);
        step_++;
    }
}

void register_lfp_sum_report(std::shared_ptr<LFPSumReport> report) {
    lfp_sum_reports.push_back(std::move(report));
}

void flush_lfp_sum_reports() {
    for (auto& report: lfp_sum_reports) {
        report->flush();
    }
}

void clear_lfp_sum_reports() {
    lfp_sum_reports.clear();
}
#endif

}  // namespace coreneuron
