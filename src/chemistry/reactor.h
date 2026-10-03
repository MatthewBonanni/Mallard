/**
 * @file reactor.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Adiabatic constant-volume reactor: the chemistry of one cell over a
 *        splitting step, integrated with RODAS.
 * @version 0.3
 * @date 2026-10-03
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_REACTOR_H
#define CHEMISTRY_REACTOR_H

#include <Kokkos_Core.hpp>

#include <cstdint>

#include "kinetics.h"
#include "rosenbrock.h"
#include "sparse_lu.h"
#include "thermo.h"

namespace chemistry {

struct ReactorOptions {
    RosenbrockOptions integrator;
    double atol_Y = 1e-10;
    int sparse = -1;  // linear solver: 0 dense LU, 1 sparse LU, -1 sparse above SPARSE_SPECIES species

    static constexpr uint32_t SPARSE_SPECIES = 100;

    bool use_sparse(const uint32_t n_species) const {
        return sparse == 1 || (sparse < 0 && n_species > SPARSE_SPECIES);
    }
};

/**
 * @brief Adiabatic, constant-volume reactor at density rho with state
 *        y = (Y_1 .. Y_Ns, T):
 *        dY_k/dt = W_k omega_k / rho,
 *        dT/dt = -sum_k u_k omega_k / (rho cv),
 *        with u_k the molar internal energies. The ODE system of integrate().
 */
template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct ConstantVolumeReactor {
    const ThermoTable<MemorySpace> & thermo;
    const KineticsTable<MemorySpace> & kinetics;
    double rho;
    double atol_Y;
    double * scratch;   // scratch_size(kinetics) doubles
    double * rank_one;  // null: J in full; else (n_species + 1) doubles: J = J_s + u v^T with u here, v_j = 1 / W_j

    KOKKOS_INLINE_FUNCTION
    static uint32_t scratch_size(const KineticsTable<MemorySpace> & kinetics) {
        return 6 * kinetics.n_species + kinetics.n_reactions + kinetics.derivatives_size();
    }

    KOKKOS_INLINE_FUNCTION uint32_t size() const { return thermo.n_species + 1; }

    KOKKOS_INLINE_FUNCTION double atol(const uint32_t i) const { return i < thermo.n_species ? atol_Y : 0.0; }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION bool admissible(const Lanes & lanes, const double * y) const {
        const double negative =
            lanes.sum(thermo.n_species, [&](const uint32_t k) { return y[k] < -atol_Y ? 1.0 : 0.0; });
        return negative == 0.0 && y[thermo.n_species] > 0.0;
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void rhs(const Lanes & lanes, const double * y, double * f) const {
        evaluate(lanes, y, f, nullptr);
    }

    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void rhs_jacobian(const Lanes & lanes, const double * y, double * f, double * J) const {
        evaluate(lanes, y, f, J);
    }

    /**
     * @brief f(y) and, if J is not null, the row-major Jacobian df/dy, from
     *        d omega / dC and d q / dT by the chain rule (C_k = rho Y_k / W_k).
     */
    template <typename Lanes>
    KOKKOS_INLINE_FUNCTION void evaluate(const Lanes & lanes, const double * y, double * f, double * J) const {
        const uint32_t ns = thermo.n_species, nr = kinetics.n_reactions, n = ns + 1;
        double * C = scratch;
        double * g_RT = C + ns;
        double * h_RT = g_RT + ns;
        double * cp_R = h_RT + ns;
        double * omega = cp_R + ns;
        double * domega_dT = omega + ns;
        double * q = domega_dT + ns;
        const ReactionDerivatives d =
            ReactionDerivatives::at(q + nr, nr, static_cast<uint32_t>(kinetics.forward_species.extent(0)),
                                    static_cast<uint32_t>(kinetics.reverse_species.extent(0)));

        const double T = y[ns];
        const auto p = ThermoTable<MemorySpace>::powers(T);
        lanes.for_each(ns, [&](const uint32_t k) {
            C[k] = rho * y[k] * thermo.inv_W(k);
            h_RT[k] = thermo.h_RT(k, p);
            g_RT[k] = h_RT[k] - thermo.s_R(k, p);
            cp_R[k] = thermo.cp_R(k, p);
        });
        lanes.sync();
        const double cv =
            GAS_CONSTANT * lanes.sum(ns, [&](const uint32_t k) { return y[k] * thermo.inv_W(k) * (cp_R[k] - 1.0); });
        const double C_total = lanes.sum(ns, [&](const uint32_t k) { return C[k]; });
        kinetics.rates_of_progress(lanes, T, C, C_total, g_RT, h_RT, q, J ? &d : nullptr);
        kinetics.production_rates(lanes, q, omega);
        // u_k = R T (h_k / RT - 1)
        lanes.for_each(ns, [&](const uint32_t k) { f[k] = omega[k] / (rho * thermo.inv_W(k)); });
        const double sum_u_omega =
            lanes.sum(ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * omega[k]; });
        const double inv_rho_cv = 1.0 / (rho * cv);
        const double dT_dt = -sum_u_omega * inv_rho_cv;
        lanes.single([&]() { f[ns] = dT_dt; });
        lanes.sync();
        if (!J) return;

        kinetics.production_rates(lanes, d.dq_dT, domega_dT);
        kinetics.production_jacobian(lanes, d, J, n, rank_one);
        // With the rank-one part apart, its share of the T row: sum_k u_k a_k
        const double sum_a =
            rank_one ? lanes.sum(ns, [&](const uint32_t k) { return GAS_CONSTANT * T * (h_RT[k] - 1.0) * rank_one[k]; })
                     : 0.0;

        // T row: d(sum u_k omega_k)/dY_j = rho / W_j sum_k u_k dw_kj; d cv / dY_j = cv_j
        lanes.for_each(ns, [&](const uint32_t j) {
            double sum = sum_a;
            for (uint32_t k = 0; k < ns; k++) sum += GAS_CONSTANT * T * (h_RT[k] - 1.0) * J[k * n + j];
            const double cv_j = GAS_CONSTANT * (cp_R[j] - 1.0) * thermo.inv_W(j);
            J[ns * n + j] = -sum * rho * thermo.inv_W(j) * inv_rho_cv - dT_dt * cv_j / cv;
        });
        lanes.sync();
        const double d_sum_dT = lanes.sum(ns, [&](const uint32_t k) {
            return GAS_CONSTANT * ((cp_R[k] - 1.0) * omega[k] + T * (h_RT[k] - 1.0) * domega_dT[k]);
        });
        const double dcv_dT =
            GAS_CONSTANT * lanes.sum(ns, [&](const uint32_t k) { return y[k] * thermo.inv_W(k) * thermo.dcp_R_dT(k, p); });
        lanes.single([&]() { J[ns * n + ns] = -d_sum_dT * inv_rho_cv - dT_dt * dcv_dT / cv; });

        // Species rows: df_k/dY_j = W_k / W_j dw_kj, df_k/dT = W_k / rho domega_k/dT
        lanes.for_each(ns, [&](const uint32_t k) {
            const double W_k = 1.0 / thermo.inv_W(k);
            for (uint32_t j = 0; j < ns; j++) J[k * n + j] *= W_k * thermo.inv_W(j);
            J[k * n + ns] = W_k * domega_dT[k] / rho;
            if (rank_one) rank_one[k] *= W_k;  // u_k = W_k a_k, with v_j = 1 / W_j
        });
        if (rank_one) lanes.single([&]() { rank_one[ns] = 0.0; });
        lanes.sync();
    }
};

/**
 * @brief Doubles of work memory advance_reactor() needs, the linear solver's
 *        included: dense, or sparse with a SparseLUPattern.
 */
template <typename MemorySpace>
KOKKOS_INLINE_FUNCTION uint32_t reactor_work_size(const KineticsTable<MemorySpace> & kinetics,
                                                  const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    const uint32_t n = kinetics.n_species + 1;
    const uint32_t solver = sparse ? sparse->work_size() + 1 : n * n;
    return n + ConstantVolumeReactor<MemorySpace>::scratch_size(kinetics) + solver + rosenbrock_work_size(n);
}

/**
 * @brief Advance one adiabatic constant-volume reactor over dt: integrate
 *        (Y, T), then clip negative mass fractions, renormalize, and take T
 *        from the initial internal energy (so energy is exact). Negative
 *        initial mass fractions are clipped the same way before integrating.
 * @param Y Mass fractions (n_species), T temperature: updated in place.
 * @param h Sub-step size: first guess in, proposal for the next call out.
 * @param work reactor_work_size(kinetics, sparse) doubles.
 * @param pivot n_species + 1 integers (dense LU).
 * @param sparse Null for the dense LU, else the pattern of the sparse one.
 */
template <typename Lanes, typename MemorySpace, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult advance_reactor(const Lanes & lanes, const ThermoTable<MemorySpace> & thermo,
                                                        const KineticsTable<MemorySpace> & kinetics, const double rho,
                                                        const double dt, double * Y, double & T, double & h,
                                                        const ReactorOptions & options, double * work,
                                                        uint32_t * pivot, Observer && observer = Observer(),
                                                        const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    const uint32_t ns = thermo.n_species, n = ns + 1;
    double * y = work;
    double * scratch = y + n;
    double * solver_work = scratch + ConstantVolumeReactor<MemorySpace>::scratch_size(kinetics);
    double * integrator_work = solver_work + (sparse ? sparse->work_size() + 1 : n * n);
    // Scalar code runs on every lane with the same values; writes go through for_each or single
    double e, cv;
    thermo.e_cv(T, MassFractions{Y}, e, cv);
    lanes.for_each(ns, [&](const uint32_t k) { y[k] = Kokkos::fmax(Y[k], 0.0); });
    lanes.sync();
    const double negative = lanes.sum(ns, [&](const uint32_t k) { return Y[k] < 0.0 ? 1.0 : 0.0; });
    double T0 = T;
    if (negative > 0.0) {
        // Small negative mass fractions (e.g. from explicit diffusion) would
        // reject every sub-step: start from the clipped composition at the same energy
        const double sum = lanes.sum(ns, [&](const uint32_t k) { return y[k]; });
        lanes.for_each(ns, [&](const uint32_t k) { y[k] /= sum; });
        lanes.sync();
        T0 = thermo.T_from_e(e, MassFractions{y}, T);
    }
    lanes.single([&]() { y[ns] = T0; });
    lanes.sync();
    RosenbrockResult result;
    if (sparse) {
        // LU values, z = A_s^-1 u, u, x, beta
        double * values = solver_work;
        double * z = values + sparse->nnz;
        double * u = z + n;
        double * x = u + n;
        const ConstantVolumeReactor<MemorySpace> reactor{thermo, kinetics, rho, options.atol_Y, scratch, u};
        const SparseLU<MemorySpace> solver{*sparse, values, z, u, x, x + n};
        result = integrate(lanes, reactor, solver, 0.0, dt, y, h, options.integrator, integrator_work,
                           static_cast<Observer &&>(observer));
    } else {
        const ConstantVolumeReactor<MemorySpace> reactor{thermo, kinetics, rho, options.atol_Y, scratch, nullptr};
        const DenseLU solver{n, solver_work, pivot};
        result = integrate(lanes, reactor, solver, 0.0, dt, y, h, options.integrator, integrator_work,
                           static_cast<Observer &&>(observer));
    }
    lanes.for_each(ns, [&](const uint32_t k) { y[k] = Kokkos::fmax(y[k], 0.0); });
    lanes.sync();
    const double total = lanes.sum(ns, [&](const uint32_t k) { return y[k]; });
    lanes.for_each(ns, [&](const uint32_t k) { Y[k] = y[k] / total; });
    lanes.sync();
    T = thermo.T_from_e(e, MassFractions{Y}, y[ns]);
    return result;
}

/** @brief advance_reactor() by one thread. */
template <typename MemorySpace, typename Observer = NoObserver>
KOKKOS_INLINE_FUNCTION RosenbrockResult advance_reactor(const ThermoTable<MemorySpace> & thermo,
                                                        const KineticsTable<MemorySpace> & kinetics, const double rho,
                                                        const double dt, double * Y, double & T, double & h,
                                                        const ReactorOptions & options, double * work,
                                                        uint32_t * pivot, Observer && observer = Observer(),
                                                        const SparseLUPattern<MemorySpace> * sparse = nullptr) {
    return advance_reactor(SerialLanes(), thermo, kinetics, rho, dt, Y, T, h, options, work, pivot,
                           static_cast<Observer &&>(observer), sparse);
}

/**
 * @brief Ignition delay as the time of the maximum of dT/dt: the vertex of
 *        the parabola through the largest sample and its two neighbors.
 */
struct IgnitionObserver {
    uint32_t index = 0;     // of T in the state
    double t_offset = 0.0;  // added to the integrator's time
    double t_ignition = 0.0;
    double peak = 0.0;      // largest sampled dT/dt
    double t[3] = {0.0, 0.0, 0.0};
    double g[3] = {0.0, 0.0, 0.0};
    uint32_t count = 0;

    KOKKOS_INLINE_FUNCTION void operator()(const double time, const double *, const double * f) {
        for (int i = 0; i < 2; i++) {
            t[i] = t[i + 1];
            g[i] = g[i + 1];
        }
        t[2] = t_offset + time;
        g[2] = f[index];
        if (++count < 3 || !(g[1] > peak && g[1] >= g[0] && g[1] >= g[2])) return;
        peak = g[1];
        const double num = (t[1] - t[0]) * (t[1] - t[0]) * (g[1] - g[2]) - (t[1] - t[2]) * (t[1] - t[2]) * (g[1] - g[0]);
        const double den = (t[1] - t[0]) * (g[1] - g[2]) - (t[1] - t[2]) * (g[1] - g[0]);
        t_ignition = den != 0.0 ? t[1] - 0.5 * num / den : t[1];
    }
};

} // namespace chemistry

#endif // CHEMISTRY_REACTOR_H
