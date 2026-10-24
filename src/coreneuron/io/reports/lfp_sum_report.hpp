/*
# =============================================================================
# See top-level LICENSE file for details.
# =============================================================================
*/

/**
 * @file lfp_sum_report.hpp
 * @brief LFP reports summed over cells (method "sum")
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "coreneuron/network/netcon.hpp"
#include "coreneuron/io/reports/nrnreport.hpp"

namespace coreneuron {

#ifdef ENABLE_SONATA_REPORTS
struct CellMapping;

/**
 * @brief The state of one summed LFP report on this rank.
 *
 * A summed report writes, for every population of its cells, one trace per electrode: the sum of
 * the contributions of the population's cells. Its rows hold this rank's partial sums for each
 * reporting step since the last flush, laid out population after population.
 *
 * Every rank holds one for every summed report, whether or not it has any of the report's cells,
 * because the partial sums are added up across ranks by a collective. That happens once per spike
 * exchange interval, from nrn_flush_reports(), which every rank calls at the same simulation time,
 * rather than at every reporting step: one collective per interval, at a point where the ranks are
 * already synchronised. Rank 0 then hands the totals to libsonatareport, as one node per
 * population whose elements are the electrodes.
 */
class LFPSumReport {
  public:
    LFPSumReport(std::string output_path, size_t width);

    /// Number of reporting steps flushed so far, which is where the next interval's rows begin
    size_t rows_flushed() const noexcept {
        return rows_flushed_;
    }
    /// The row of the given reporting step since the last flush
    double* row(size_t row_index);
    /// Records a reporting step. Only thread 0's event calls it, and every rank has one, so every
    /// rank knows how many rows to add up, and rank 0 which steps to record.
    void add_step(double step);
    /// Rank 0: the gid of each population's node, which libsonatareport records
    void set_nodes(std::vector<int> node_gids);
    /// Rank 0: the values libsonatareport reads, laid out like a row
    double* values() noexcept {
        return values_.data();
    }
    /// Adds up the pending rows across ranks and records them on rank 0. Collective.
    void flush();

  private:
    std::string output_path_;
    size_t width_;
    size_t rows_flushed_ = 0;
    std::vector<double> rows_;
    std::vector<double> steps_;
    std::vector<double> totals_;
    std::vector<double> values_;
    std::vector<int> node_gids_;
};

/**
 * @brief Adds one thread's cells to a summed LFP report at every reporting step.
 *
 * Thread 0 always has one, with or without cells, to count the reporting steps. The steps are the
 * ones libsonatareport records (SonataData::is_due_to_report), so the values are computed at the
 * steps that are written.
 */
class LFPSumEvent: public DiscreteEvent {
  public:
    /// A cell of the report: its population's first value in a row, and the report's electrodes
    /// within the cell's electrode factors
    struct Cell {
        const CellMapping* mapping;
        size_t column;
        size_t first_electrode;
        size_t num_electrodes;
    };

    LFPSumEvent(std::shared_ptr<LFPSumReport> report,
                std::vector<Cell> cells,
                bool counts_steps,
                const ReportConfiguration& config,
                double dt,
                double tstart);

    /// at a reporting step, add the cells' contributions to the step's row; then reschedule
    void deliver(double t, NetCvode* nc, NrnThread* nt) override;
    bool require_checkpoint() override {
        return false;
    }
    int type() const override {
        return ReportEventType;
    }
    /// adds the cells' contributions to each of the report's electrodes to `row`
    void add_lfp(NrnThread* nt, double* row) const;

  private:
    bool is_due(double step) const noexcept;

    std::shared_ptr<LFPSumReport> report_;
    std::vector<Cell> cells_;
    bool counts_steps_;
    std::string report_path_;
    double dt_;
    double step_;
    double first_step_;
    double last_step_;
    int reporting_period_;
    size_t steps_seen_ = 0;
};

/// Registers a summed LFP report, for nrn_flush_reports() and finalize_report() to flush
void register_lfp_sum_report(std::shared_ptr<LFPSumReport> report);
/// Flushes every summed LFP report. Collective; nothing happens without any.
void flush_lfp_sum_reports();
/// Forgets the summed LFP reports, once their files are written
void clear_lfp_sum_reports();
#endif

}  // namespace coreneuron
