/**
 * @file load_balance_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cost model and trigger of dynamic load balancing.
 * @version 0.4
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <vector>

#include "load_balance.h"

/**
 * @brief Busy times that are exactly a smooth-cell cost times cells plus an
 *        extra per troubled cell give back the troubled-cell cost; when the
 *        troubled counts are proportional to the cell counts the two costs
 *        can't be separated and the prior stays.
 */
TEST(LoadBalanceTest, FitRecoversTheTroubledCellCost) {
    const double smooth = 2e-6, troubled = 7 * smooth;  // a troubled cell costs 7 smooth ones
    std::vector<RankLoad> loads;
    for (const auto & [cells, n_troubled] : {std::pair{1000.0, 0.0}, {1100.0, 300.0}, {950.0, 40.0}, {1020.0, 120.0}}) {
        loads.push_back({smooth * cells + (troubled - smooth) * n_troubled, cells, n_troubled});
    }
    EXPECT_NEAR(fit_troubled_cost(loads, 3.0), 7.0, 1e-9);

    for (RankLoad & l : loads) {
        l.troubled = 0.1 * l.cells;
        l.busy = 1.6 * smooth * l.cells;
    }
    EXPECT_EQ(fit_troubled_cost(loads, 3.0), 3.0);
}

/**
 * @brief The trigger rebalances an imbalance worth fixing, and each of its
 *        guards alone stops it: imbalance under the threshold, saving under
 *        twice the cost, and the cost budget.
 */
TEST(LoadBalanceTest, RebalanceOnlyWhenItPays) {
    RebalancePolicy policy;  // threshold 1.1, budget 1%
    // The slowest rank works 20 ms per step, the others 10 ms
    const std::vector<RankLoad> loads = {{0.02, 1, 0}, {0.01, 1, 0}, {0.01, 1, 0}, {0.01, 1, 0}};
    // Imbalance 1.6; bringing the slowest rank to 1.05 x the mean saves 6.9 s over 1000 steps
    EXPECT_TRUE(rebalance_pays(loads, 1000.0, 1.0, 0.0, 1000.0, policy));
    policy.threshold = 1.7;  // imbalance 1.6
    EXPECT_FALSE(rebalance_pays(loads, 1000.0, 1.0, 0.0, 1000.0, policy));
    policy.threshold = 1.1;
    EXPECT_FALSE(rebalance_pays(loads, 1000.0, 7.0, 0.0, 10000.0, policy));
    EXPECT_FALSE(rebalance_pays(loads, 1000.0, 1.0, 9.5, 1000.0, policy));
}
