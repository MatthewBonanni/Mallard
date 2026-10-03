/**
 * @file load_balance.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cost model and trigger of dynamic load balancing.
 * @version 0.4
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "load_balance.h"

#include <algorithm>
#include <cmath>

#include "input.h"

RebalancePolicy RebalancePolicy::from_input(const toml::value & input) {
    RebalancePolicy policy;
    auto real = [&](const char * key, double fallback) {
        return double(find_real_or(input, "parallel", key, rtype(fallback)));
    };
    policy.enabled = toml::find_or<bool>(input, "parallel", "rebalance", false);
    policy.interval = toml::find_or<uint64_t>(input, "parallel", "rebalance_interval", policy.interval);
    policy.threshold = real("rebalance_threshold", policy.threshold);
    policy.max_cost = real("rebalance_max_cost", policy.max_cost);
    policy.troubled_cost = real("troubled_cost", policy.troubled_cost);
    if (policy.interval == 0) throw InputError("parallel.rebalance_interval must be positive.");
    if (!(policy.threshold >= 1.0)) throw InputError("parallel.rebalance_threshold must be at least 1.");
    if (!(policy.max_cost > 0.0)) throw InputError("parallel.rebalance_max_cost must be positive.");
    if (!(policy.troubled_cost >= 1.0 && policy.troubled_cost <= MAX_TROUBLED_COST)) {
        throw InputError("parallel.troubled_cost must be between 1 and 100.");
    }
    return policy;
}

double fit_troubled_cost(const std::vector<RankLoad> & loads, double prior) {
    double nn = 0.0, nt = 0.0, tt = 0.0, bn = 0.0, bt = 0.0;
    for (const RankLoad & l : loads) {
        nn += l.cells * l.cells;
        nt += l.cells * l.troubled;
        tt += l.troubled * l.troubled;
        bn += l.busy * l.cells;
        bt += l.busy * l.troubled;
    }
    // Cosine of the angle between the cell and troubled vectors: near 1, a and b can't be told apart
    const double det = nn * tt - nt * nt;
    if (!(nn > 0.0 && tt > 0.0) || det < 1e-4 * nn * tt) return prior;
    const double a = (bn * tt - bt * nt) / det;
    const double b = (bt * nn - bn * nt) / det;
    if (!(a > 0.0)) return prior;
    return std::clamp(1.0 + b / a, 1.0, MAX_TROUBLED_COST);
}

double imbalance(const std::vector<RankLoad> & loads) {
    double max_busy = 0.0, sum = 0.0;
    for (const RankLoad & l : loads) {
        max_busy = std::max(max_busy, l.busy);
        sum += l.busy;
    }
    return sum > 0.0 ? max_busy * loads.size() / sum : 1.0;
}

bool rebalance_pays(const std::vector<RankLoad> & loads, uint64_t window_steps, double horizon_steps, double cost,
                    double spent, double projected_wall, const RebalancePolicy & policy) {
    if (loads.empty() || window_steps == 0 || imbalance(loads) <= policy.threshold) return false;
    double max_busy = 0.0, sum = 0.0;
    for (const RankLoad & l : loads) {
        max_busy = std::max(max_busy, l.busy);
        sum += l.busy;
    }
    constexpr double TARGET = 1.05;  // Max over mean busy time a rebalance is expected to reach
    const double saving_per_step = (max_busy - TARGET * sum / loads.size()) / window_steps;
    if (!(saving_per_step * horizon_steps > 2.0 * cost)) return false;
    return spent + cost <= policy.max_cost * projected_wall;
}
