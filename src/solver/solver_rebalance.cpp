/**
 * @file solver_rebalance.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Dynamic load balancing: repartition, migrate the solution, rebuild.
 * @version 0.4
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "comm.h"
#include "log.h"
#include "partition.h"

uint64_t Solver::rebalance(const std::vector<uint64_t> & weights) {
    if (!setup) {
        throw std::logic_error("Solver::rebalance: needs a distributed run with parallel.rebalance = true.");
    }
    const int p = comm::size();
    const uint32_t n_owned = mesh->n_owned();
    if (comm::allreduce(uint32_t(weights.size() != n_owned), comm::Op::MAX) != 0) {
        throw std::invalid_argument("Solver::rebalance: one weight per owned cell.");
    }

    // New owners, renumbered to keep as much weight in place as possible.
    // dKaMinPar 3.7 never finishes balancing parts of a few dozen cells when
    // single cells weigh a large share of them; the curve has no such limit.
    const std::vector<uint64_t> block_weights = setup->owned_to_block(weights);
    uint64_t max_weight = 0, total = 0;
    for (uint64_t w : block_weights) {
        max_weight = std::max(max_weight, w);
        total += w;
    }
    max_weight = comm::allreduce(max_weight, comm::Op::MAX);
    total = comm::allreduce(total, comm::Op::SUM);
    const bool graph = partitioner == "graph" && 32 * max_weight * uint64_t(p) <= total;
    std::vector<int> owner = graph ? partition_graph(*setup, p, block_weights) : partition_hilbert(*setup, p, block_weights);
    owner = relabel_parts(setup->block_owner(), owner, block_weights, p);

    // The owned solution to the new owners by global id, as doubles (exact for floats too)
    copy_device_to_host();
    const uint32_t n_species = species_names.size();
    const uint32_t n_values = N_CONSERVATIVE + n_species;
    const std::vector<int> dest = setup->owners_of_owned(owner);
    std::vector<std::vector<uint64_t>> send_ids(p);
    std::vector<std::vector<double>> send_values(p);
    uint64_t moved = 0;
    for (uint32_t c = 0; c < n_owned; c++) {
        moved += dest[c] != comm::rank();
        send_ids[dest[c]].push_back(distribution.global_cell[c]);
        auto & out = send_values[dest[c]];
        FOR_I_CONSERVATIVE out.push_back(double(h_conservatives(c, i)));
        for (uint32_t k = 0; k < n_species; k++) out.push_back(double(h_species(c, k)));
    }
    const std::vector<uint64_t> ids = comm::exchange(std::move(send_ids)).data;
    const std::vector<double> values = comm::exchange(std::move(send_values)).data;

    // The old partition's reconstruction data goes before the new one is built
    face_reconstruction.reset();
    solution_vec.clear();
    rhs_vec.clear();
    setup->distribute(owner);
    init_mesh();
    init_boundaries();
    init_numerics();
    while (halo_too_shallow()) {
        init_mesh();
        init_boundaries();
        init_numerics();
    }
    allocate_memory();
    init_rhs_split();
    init_sources();
    register_data();
    bind_outputs();

    // Owned cells are numbered in global order
    const auto first = distribution.global_cell.begin(), last = first + distribution.n_owned;
    if (ids.size() != distribution.n_owned) throw std::logic_error("Solver::rebalance: owned cells lost in migration.");
    for (size_t k = 0; k < ids.size(); k++) {
        const uint32_t c = std::lower_bound(first, last, ids[k]) - first;
        FOR_I_CONSERVATIVE h_conservatives(c, i) = rtype(values[k * n_values + i]);
        for (uint32_t j = 0; j < n_species; j++) h_species(c, j) = rtype(values[k * n_values + N_CONSERVATIVE + j]);
    }
    copy_host_to_device();
    halo.exchange(state());
    halo_current = false;
    update_primitives();
    return comm::allreduce(moved, comm::Op::SUM);
}

namespace {

/** @brief Counts, per owned cell, the steps in which TENO marked it troubled. */
struct TroubledCountFunctor {
    Kokkos::View<rtype *> sigma;
    rtype threshold;
    Kokkos::View<uint32_t *> steps;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell) const {
        if (sigma(i_cell) >= threshold) steps(i_cell)++;
    }
};

struct TroubledSumFunctor {
    Kokkos::View<uint32_t *> steps;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i_cell, uint64_t & sum) const { sum += steps(i_cell); }
};

} // namespace

void Solver::count_troubled() {
    const auto * teno = dynamic_cast<const TENO *>(face_reconstruction.get());
    if (teno == nullptr) return;
    Kokkos::parallel_for("count_troubled", troubled_steps.extent(0),
                         TroubledCountFunctor{teno->troubled, teno->sigma_threshold, troubled_steps});
}

void Solver::consider_rebalance() {
    const uint64_t steps = step - window_step;
    uint64_t troubled = 0;
    Kokkos::View<uint32_t *> counts = troubled_steps;
    Kokkos::parallel_reduce("sum_troubled", counts.extent(0), TroubledSumFunctor{counts}, troubled);
    const std::vector<double> flat = comm::allgatherv(
        std::vector<double>{window_busy, double(mesh->n_owned()) * double(steps), double(troubled)});
    std::vector<RankLoad> loads(flat.size() / 3);
    for (size_t r = 0; r < loads.size(); r++) loads[r] = {flat[3 * r], flat[3 * r + 1], flat[3 * r + 2]};
    troubled_cost = fit_troubled_cost(loads, troubled_cost);
    last_imbalance = imbalance(loads);

    // Every rank must decide alike: clocks and progress from the slowest rank
    const auto [elapsed, f] = comm::allreduce(std::array<double, 2>{timer.seconds(), progress()}, comm::Op::MAX);
    const double df = f - progress_run_start;
    const double horizon = std::min(df > 0.0 ? double(step - step_run_start) * (1.0 - f) / df
                                             : double(rebalance_policy.interval),
                                    10.0 * double(rebalance_policy.interval));
    const double projected = f > 0.0 ? elapsed / f : elapsed;
    if (rebalance_pays(loads, steps, horizon, rebalance_cost, t_wall_rebalance, projected, rebalance_policy)) {
        auto h_counts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), counts);
        std::vector<uint64_t> weights(h_counts.extent(0));
        for (size_t c = 0; c < weights.size(); c++) {
            const double cost = 1.0 + (troubled_cost - 1.0) * h_counts(c) / double(steps);
            weights[c] = static_cast<uint64_t>(std::llround(WEIGHT_UNIT * cost));
        }
        Kokkos::Timer rebalance_timer;
        const uint64_t moved = rebalance(weights);
        const double seconds = comm::allreduce(rebalance_timer.seconds(), comm::Op::MAX);
        rebalance_cost = seconds;
        t_wall_rebalance += seconds;
        n_rebalances++;
        logging::event(step, double(t), "rebalance",
                       logging::format("imbalance %.2f, troubled cells cost %.1f, %s cells moved in %s",
                                       last_imbalance, troubled_cost, logging::count(moved).c_str(),
                                       logging::duration(seconds).c_str()));
    } else {
        Kokkos::deep_copy(troubled_steps, 0u);
    }
    window_step = step;
    window_busy = 0.0;
}
