/**
 * @file load_balance.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cost model and trigger of dynamic load balancing (docs/design/mpi.md, section 10).
 * @version 0.4
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef LOAD_BALANCE_H
#define LOAD_BALANCE_H

#include <cstdint>
#include <vector>

#include <toml.hpp>

/** @brief One rank's load over a window of steps. */
struct RankLoad {
    double busy = 0.0;      // Seconds of own work: stepping time minus waits for other ranks
    double cells = 0.0;     // Owned cell-steps
    double troubled = 0.0;  // Owned cell-steps in which the cell was troubled
};

/** @brief The [parallel] rebalancing inputs. */
struct RebalancePolicy {
    bool enabled = false;
    uint64_t interval = 100;     // Steps between checks
    double threshold = 1.1;      // Rebalance above this max / mean busy time
    double max_cost = 0.01;      // Fraction of the run's wall time rebalancing may take
    double troubled_cost = 8.0;  // Cost of a troubled cell in smooth cells, until measured

    static RebalancePolicy from_input(const toml::value & input);
};

/** @brief Part weights are integers: this many units per smooth cell. */
inline constexpr double WEIGHT_UNIT = 16.0;

/** @brief Largest troubled-cell cost a fit may give. */
inline constexpr double MAX_TROUBLED_COST = 100.0;

/**
 * @brief Cost of a troubled cell in smooth cells from the ranks' loads: the
 *        least-squares fit busy = a cells + b troubled gives 1 + b / a,
 *        clamped to [1, MAX_TROUBLED_COST]. Returns prior when troubled
 *        counts don't vary independently of cell counts, or a <= 0.
 */
double fit_troubled_cost(const std::vector<RankLoad> & loads, double prior);

/**
 * @brief Whether rebalancing now pays: the imbalance max / mean of busy
 *        exceeds the threshold, bringing the slowest rank to 5% above the mean
 *        saves more than twice the predicted cost over the horizon, and the
 *        rebalancing time spent plus that cost stays within max_cost of the
 *        projected wall time of the run.
 * @param window_steps Steps the loads were measured over.
 * @param horizon_steps Steps the saving is expected to last.
 * @param cost Predicted wall time of a rebalance.
 * @param spent Wall time rebalancing took so far.
 * @param projected_wall Projected wall time of the whole run.
 */
bool rebalance_pays(const std::vector<RankLoad> & loads, uint64_t window_steps, double horizon_steps, double cost,
                    double spent, double projected_wall, const RebalancePolicy & policy);

/** @brief Max over mean of the ranks' busy times (1 when idle). */
double imbalance(const std::vector<RankLoad> & loads);

#endif // LOAD_BALANCE_H
