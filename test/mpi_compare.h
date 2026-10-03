/**
 * @file mpi_compare.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Comparison of distributed runs with serial ones.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef MPI_COMPARE_H
#define MPI_COMPARE_H

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <span>
#include <string>
#include <vector>

#include "comm.h"
#include "solver.h"
#include "test_fixtures.h"

/**
 * @brief Every rank compares the gathered solution of a finished distributed
 *        run with that of the same input run serially.
 */
inline void expect_matches_serial(Solver & distributed, const std::string & input) {
    distributed.copy_device_to_host();

    Solver serial;
    serial.set_distributed(false);
    serial.init(parse_toml(input));
    serial.run();
    serial.copy_device_to_host();

    const uint32_t n_global = serial.get_mesh()->n_cells;
    const auto & dist = distributed.get_distribution();
    std::vector<double> gathered(n_global * N_CONSERVATIVE, 0.0);
    std::vector<double> count(n_global, 0.0);
    for (uint32_t c = 0; c < distributed.get_mesh()->n_owned(); c++) {
        const uint64_t g = distributed.is_distributed() ? dist.global_cell[c] : c;
        FOR_I_CONSERVATIVE gathered[g * N_CONSERVATIVE + i] = double(distributed.h_conservatives(c, i));
        count[g] += 1.0;
    }
    comm::allreduce(std::span<double>(gathered), comm::Op::SUM);
    comm::allreduce(std::span<double>(count), comm::Op::SUM);

    EXPECT_EQ(distributed.get_step(), serial.get_step());
    EXPECT_EQ(distributed.get_time(), serial.get_time());
    double max_rel = 0.0;
    for (uint32_t g = 0; g < n_global; g++) {
        ASSERT_EQ(count[g], 1.0) << "cell " << g << " owned " << count[g] << " times";
        FOR_I_CONSERVATIVE {
            const double ref = double(serial.h_conservatives(g, i));
            max_rel = std::max(max_rel, std::abs(gathered[g * N_CONSERVATIVE + i] - ref) / (std::abs(ref) + 1e-3));
        }
    }
    // Faces and stencils are ordered by global cell ids, so every rank count
    // computes the same sums in the same order
    EXPECT_EQ(max_rel, 0.0) << "on " << comm::size() << " ranks";
}

/** @brief Run the input distributed and serially, and compare. */
inline void expect_matches_serial(const std::string & input) {
    Solver distributed;
    distributed.init(parse_toml(input));
    distributed.run();
    expect_matches_serial(distributed, input);
}

/**
 * @brief Steps the input distributed, moving cells to new owners after steps
 *        3 and 6 (weights heavy on the first, then the last quarter of the
 *        cells by global id), and compares with the serial run.
 */
inline void expect_rebalanced_run_matches_serial(const std::string & input, uint32_t n_steps) {
    Solver solver;
    solver.init(parse_toml(input));
    if (!solver.is_distributed()) GTEST_SKIP() << "needs more than one rank";
    const uint64_t n = solver.get_mesh()->n_global_cells;
    uint64_t moved = 0;
    while (solver.get_step() < n_steps) {
        solver.calc_dt();
        solver.take_step();
        const uint32_t step = solver.get_step();
        if (step != 3 && step != 6) continue;
        const auto & dist = solver.get_distribution();
        const std::set<uint64_t> before(dist.global_cell.begin(), dist.global_cell.begin() + dist.n_owned);
        std::vector<uint64_t> weights(dist.n_owned);
        for (uint32_t c = 0; c < dist.n_owned; c++) {
            const uint64_t g = dist.global_cell[c];
            weights[c] = (step == 3 ? g < n / 4 : g >= 3 * n / 4) ? 8 : 1;
        }
        solver.rebalance(weights);
        const auto & now = solver.get_distribution();
        for (uint32_t c = 0; c < now.n_owned; c++) moved += !before.count(now.global_cell[c]);
    }
    EXPECT_GT(comm::allreduce(moved, comm::Op::SUM), 0u);
    expect_matches_serial(solver, input);
}

#endif // MPI_COMPARE_H
