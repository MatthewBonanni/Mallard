/**
 * @file cell_chemistry.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Finite-rate chemistry of the cells of a mesh.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "cell_chemistry.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <Kokkos_Sort.hpp>

#ifndef MALLARD_CHEMISTRY_MIN_BLOCKS
#define MALLARD_CHEMISTRY_MIN_BLOCKS 16
#endif

namespace {

constexpr double WORK_MEMORY_BYTES = 1024.0 * 1024.0 * 1024.0;

template <bool Sparse>
struct TeamTag {};
// Fewer registers per thread, more cells per SM: the team kernels are latency bound
constexpr unsigned TEAM_MIN_BLOCKS = MALLARD_CHEMISTRY_MIN_BLOCKS;
template <bool Sparse>
using TeamPolicy = Kokkos::TeamPolicy<TeamTag<Sparse>, Kokkos::LaunchBounds<32, TEAM_MIN_BLOCKS>>;
using Member = TeamPolicy<false>::member_type;

/**
 * @brief Mass fractions (double) and temperature of a cell; returns its
 *        specific internal energy.
 */
KOKKOS_INLINE_FUNCTION
double cell_composition(const Mixture & gas, const StateView & U, const SpeciesView & rhoY,
                        const Kokkos::View<rtype *> & T_seed, const uint32_t c, double * Y, double & T) {
    const double rho = static_cast<double>(U(c, 0));
    double u2 = 0.0;
    FOR_I_DIM u2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
    const double e = static_cast<double>(U(c, N_DIM + 1)) / rho - 0.5 * u2 / (rho * rho);
    for (uint32_t k = 0; k < gas.n_species; k++) Y[k] = static_cast<double>(rhoY(c, k)) / rho;
    T = gas.thermo.T_from_e(e, chemistry::MassFractions{Y}, static_cast<double>(T_seed(c)));
    return e;
}

/** @brief Flags the cells of a chunk that need chemistry (see CellChemistry). */
struct ActivityFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t *> active;
    uint32_t first;
    double dt, T_frozen, threshold;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        const uint32_t c = first + i, ns = gas.n_species;
        double * Y = &work(i, 0);
        double * C = Y + ns;
        double * g_RT = C + ns;
        double * omega = g_RT + ns;
        double * q = omega + ns;
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        active(i) = 0;
        if (T < T_frozen) return;
        const double rho = static_cast<double>(U(c, 0));
        const auto p = chemistry::ThermoTable<>::powers(T);
        for (uint32_t k = 0; k < ns; k++) {
            C[k] = rho * Y[k] * gas.thermo.inv_W(k);
            g_RT[k] = gas.thermo.h_RT(k, p) - gas.thermo.s_R(k, p);
        }
        kinetics.rates_of_progress(T, C, g_RT, q);
        kinetics.production_rates(q, omega);
        double rate = 0.0;
        for (uint32_t k = 0; k < ns; k++) rate = Kokkos::fmax(rate, Kokkos::fabs(omega[k]) / (rho * gas.thermo.inv_W(k)));
        active(i) = dt * rate > threshold ? 1 : 0;
    }
};

/** @brief Compacts the flagged cells of a chunk into a queue; the total is the queue length. */
struct QueueFunctor {
    Kokkos::View<uint32_t *> active;
    Kokkos::View<uint32_t *> queue;
    Kokkos::View<float *> cost;
    Kokkos::View<float *> previous_cost;
    uint32_t first;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & offset, const bool final) const {
        if (final && active(i)) {
            queue(offset) = first + i;
            cost(offset) = -previous_cost(first + i);  // most expensive first
        }
        offset += active(i);
    }
};

/**
 * @brief Advances the cells of a queue over dt, writes their partial
 *        densities, last sub-step and cost, and counts failures. The
 *        temperature seed stays as the RHS left it: halo copies of a cell
 *        get no chemistry, and their seeds must follow the owner's.
 */
struct AdvanceFunctor {
    using value_type = uint32_t;  // failures
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<rtype *> chem_h;
    Kokkos::View<rtype *> chem_cost;
    Kokkos::View<float *> previous_cost;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<uint32_t **, Kokkos::LayoutRight> pivot;
    Kokkos::View<uint32_t *> queue;
    chemistry::ReactorOptions options;
    double dt;
    uint32_t lanes;
    chemistry::SparseLUPattern<> pattern;
    bool sparse;
    uint32_t fast_size;  // doubles of team scratch for the work memory, 0: global memory

    /** @brief One thread per cell. */
    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i, uint32_t & failures) const {
        const uint32_t c = queue(i), ns = gas.n_species;
        double * Y = &work(i, 0);
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        double h = static_cast<double>(chem_h(c));
        const chemistry::RosenbrockResult r =
            chemistry::advance_reactor(gas.thermo, kinetics, rho, dt, Y, T, h, options, Y + ns, &pivot(i, 0),
                                       chemistry::NoObserver(), sparse ? &pattern : nullptr);
        for (uint32_t k = 0; k < ns; k++) rhoY(c, k) = static_cast<rtype>(rho * Y[k]);
        chem_h(c) = static_cast<rtype>(h);
        chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
        previous_cost(c) = static_cast<float>(r.steps + r.rejected);
        if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
    }

    /** @brief All threads and lanes of a team per cell. */
    template <bool Sparse>
    KOKKOS_INLINE_FUNCTION void operator()(TeamTag<Sparse>, const Member & member, uint32_t & failures) const {
        const uint32_t slot = static_cast<uint32_t>(member.league_rank()), c = queue(slot), ns = gas.n_species;
        const chemistry::TeamLanes<Member> team(member, lanes * static_cast<uint32_t>(member.team_size()));
        double * fast =
            fast_size > 0 ? static_cast<double *>(member.team_scratch(0).get_shmem(fast_size * sizeof(double))) : nullptr;
        double * Y = &work(slot, 0);
        const double rho = static_cast<double>(U(c, 0));
        team.for_each(ns, [&](const uint32_t k) { Y[k] = static_cast<double>(rhoY(c, k)) / rho; });
        team.sync();
        double u2 = 0.0;
        FOR_I_DIM u2 += static_cast<double>(U(c, 1 + i)) * static_cast<double>(U(c, 1 + i));
        const double e = static_cast<double>(U(c, N_DIM + 1)) / rho - 0.5 * u2 / (rho * rho);
        double T = gas.thermo.T_from_e(e, chemistry::MassFractions{Y}, static_cast<double>(T_seed(c)));
        double h = static_cast<double>(chem_h(c));
        const chemistry::RosenbrockResult r =
            chemistry::advance_reactor<Sparse>(team, gas.thermo, kinetics, rho, dt, Y, T, h, options, Y + ns,
                                               &pivot(slot, 0), chemistry::NoObserver(), &pattern, fast);
        team.for_each(ns, [&](const uint32_t k) { rhoY(c, k) = static_cast<rtype>(rho * Y[k]); });
        team.single([&]() {
            chem_h(c) = static_cast<rtype>(h);
            chem_cost(c) += static_cast<rtype>(r.steps + r.rejected);
            previous_cost(c) = static_cast<float>(r.steps + r.rejected);
        });
        Kokkos::single(Kokkos::PerThread(member), [&]() {
            if (r.status != chemistry::RosenbrockStatus::SUCCESS) failures++;
        });
    }
};

/**
 * @brief Heat release rate -sum_k h_k omega_k [W/m^3] and mass production
 *        rates W_k omega_k [kg/(m^3 s)] of the cells of a chunk.
 */
struct HeatReleaseFunctor {
    Mixture gas;
    chemistry::KineticsTable<> kinetics;
    StateView U;
    SpeciesView rhoY;
    Kokkos::View<rtype *> T_seed;
    Kokkos::View<double **, Kokkos::LayoutRight> work;
    Kokkos::View<rtype *> hrr;
    Kokkos::View<rtype **, Kokkos::LayoutRight> production;
    uint32_t first;

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint32_t i) const {
        const uint32_t c = first + i, ns = gas.n_species;
        double * Y = &work(i, 0);
        double * C = Y + ns;
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        double * omega = h_RT + ns;
        double * q = omega + ns;
        double T;
        cell_composition(gas, U, rhoY, T_seed, c, Y, T);
        const double rho = static_cast<double>(U(c, 0));
        const auto p = chemistry::ThermoTable<>::powers(T);
        for (uint32_t k = 0; k < ns; k++) {
            C[k] = rho * Y[k] * gas.thermo.inv_W(k);
            h_RT[k] = gas.thermo.h_RT(k, p);
            g_RT[k] = h_RT[k] - gas.thermo.s_R(k, p);
        }
        kinetics.rates_of_progress(T, C, g_RT, q);
        kinetics.production_rates(q, omega);
        double sum = 0.0;
        for (uint32_t k = 0; k < ns; k++) {
            sum += h_RT[k] * omega[k];
            production(c, k) = static_cast<rtype>(omega[k] / gas.thermo.inv_W(k));
        }
        hrr(c) = static_cast<rtype>(-chemistry::GAS_CONSTANT * T * sum);
    }
};

} // namespace

void CellChemistry::init(const Mixture & gas_in, const chemistry::Mechanism & mechanism,
                         const chemistry::KineticsTable<> & kinetics_in, const CellChemistryOptions & options_in,
                         const uint32_t n_cells) {
    gas = gas_in;
    kinetics = kinetics_in;
    options = options_in;
    const uint32_t ns = gas.n_species;
    const uint32_t lanes_max = static_cast<uint32_t>(TeamPolicy<false>::vector_length_max());
    if (options.lanes == 0) {
        // A warp per cell where the device has lanes and the mechanism fills them
        n_lanes = (lanes_max >= 32 && ns >= 16) ? 32 : 1;
    } else {
        if ((options.lanes & (options.lanes - 1)) != 0) {
            throw std::invalid_argument("chemistry.lanes must be a power of 2.");
        }
        n_lanes = std::min(options.lanes, lanes_max);
    }
    n_threads = n_lanes == 1 ? 1 : (options.threads > 0 ? options.threads : 1);
    if (n_lanes * n_threads > 32) throw std::invalid_argument("chemistry: at most 32 threads and lanes per cell.");
    sparse = options.reactor.use_sparse(ns);
    if (sparse) pattern = chemistry::make_sparse_lu_pattern(mechanism);
    // Teams keep all but the Jacobian in scratch memory where it fits
    fast_bytes = 0;
    if (n_lanes > 1 && options.shared != 0) {
        const size_t bytes = sizeof(double) * chemistry::reactor_fast_size(kinetics, sparse ? &pattern : nullptr);
        const size_t room = static_cast<size_t>(TeamPolicy<false>::scratch_size_max(0));
        if (bytes + chemistry::TeamLanes<Member>::scratch_bytes(n_lanes * n_threads) + 64 <= room) fast_bytes = bytes;
    }
    // Mass fractions, then the reactor's work memory (also enough for the activity and heat release kernels)
    const uint32_t work_size = ns + chemistry::reactor_work_size(kinetics, sparse ? &pattern : nullptr);
    const double per_cell = 8.0 * work_size + 4.0 * (ns + 3) + 8.0;
    const size_t concurrency = static_cast<size_t>(Kokkos::DefaultExecutionSpace().concurrency());
    size_t chunk = std::max<size_t>(4096, 4 * concurrency);
    chunk = std::min<size_t>(chunk, static_cast<size_t>(WORK_MEMORY_BYTES / per_cell));
    chunk = std::max<size_t>(1, std::min<size_t>(chunk, n_cells));
    work = Kokkos::View<double **, Kokkos::LayoutRight>("chem_work", chunk, work_size);
    pivot = Kokkos::View<uint32_t **, Kokkos::LayoutRight>("chem_pivot", chunk, ns + 1);
    active = Kokkos::View<uint32_t *>("chem_active", chunk);
    queue = Kokkos::View<uint32_t *>("chem_queue", chunk);
    cost = Kokkos::View<float *>("chem_queue_cost", chunk);
    previous_cost = Kokkos::View<float *>("chem_previous_cost", n_cells);
}

CellChemistry::Statistics CellChemistry::advance(const StateView & U, const SpeciesView & rhoY,
                                                 const Kokkos::View<rtype *> & T_seed,
                                                 const Kokkos::View<rtype *> & chem_h,
                                                 const Kokkos::View<rtype *> & chem_cost, const uint32_t n,
                                                 const double dt) {
    Statistics stats;
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        Kokkos::parallel_for("chemistry_activity", m,
                             ActivityFunctor{gas, kinetics, U, rhoY, T_seed, work, active, first, dt, options.T_frozen,
                                             1e-2 * options.reactor.atol_Y});
        uint32_t n_active = 0;
        Kokkos::parallel_scan("chemistry_queue", m, QueueFunctor{active, queue, cost, previous_cost, first}, n_active);
        stats.active += n_active;
        if (n_active == 0) continue;
        const auto queued = Kokkos::make_pair(0u, n_active);
        if (n_lanes == 1 && options.bin_by_cost) {
            // Neighboring threads get cells of similar cost
            Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), Kokkos::subview(cost, queued),
                                              Kokkos::subview(queue, queued));
        }
        const AdvanceFunctor functor{gas,  kinetics, U,     rhoY,  T_seed,          chem_h, chem_cost, previous_cost,
                                     work, pivot,    queue, options.reactor, dt, n_lanes, pattern, sparse,
                                     static_cast<uint32_t>(fast_bytes / sizeof(double))};
        uint32_t failures = 0;
        if (n_lanes == 1) {
            Kokkos::parallel_reduce("chemistry_advance", n_active, functor, Kokkos::Sum<uint32_t>(failures));
        } else {
            const size_t scratch = chemistry::TeamLanes<Member>::scratch_bytes(n_lanes * n_threads) + fast_bytes + 64;
            if (sparse) {
                const auto policy = TeamPolicy<true>(static_cast<int>(n_active), static_cast<int>(n_threads),
                                                     static_cast<int>(n_lanes))
                                        .set_scratch_size(0, Kokkos::PerTeam(scratch));
                Kokkos::parallel_reduce("chemistry_advance_teams", policy, functor, Kokkos::Sum<uint32_t>(failures));
            } else {
                const auto policy = TeamPolicy<false>(static_cast<int>(n_active), static_cast<int>(n_threads),
                                                      static_cast<int>(n_lanes))
                                        .set_scratch_size(0, Kokkos::PerTeam(scratch));
                Kokkos::parallel_reduce("chemistry_advance_teams", policy, functor, Kokkos::Sum<uint32_t>(failures));
            }
        }
        stats.failures += failures;
    }
    return stats;
}

void CellChemistry::heat_release(const StateView & U, const SpeciesView & rhoY, const Kokkos::View<rtype *> & T_seed,
                                 const Kokkos::View<rtype *> & hrr,
                                 const Kokkos::View<rtype **, Kokkos::LayoutRight> & production, const uint32_t n) {
    const uint32_t chunk = static_cast<uint32_t>(work.extent(0));
    for (uint32_t first = 0; first < n; first += chunk) {
        const uint32_t m = std::min(chunk, n - first);
        Kokkos::parallel_for("heat_release_rate", m,
                             HeatReleaseFunctor{gas, kinetics, U, rhoY, T_seed, work, hrr, production, first});
    }
}
