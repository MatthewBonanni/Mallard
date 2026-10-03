/**
 * @file thermo.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device tables of species thermodynamics and thermally perfect
 *        mixture properties.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef CHEMISTRY_THERMO_H
#define CHEMISTRY_THERMO_H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include "mechanism.h"

namespace chemistry {

/** @brief Mass fractions held in a contiguous array. */
struct MassFractions {
    const double * Y;
    KOKKOS_INLINE_FUNCTION double operator()(const uint32_t k) const { return Y[k]; }
};

/**
 * @brief Species thermodynamics of a mechanism on the device, in double
 *        precision in every build: per species its NASA-9 form ranges
 *        (SpeciesThermo), and the ideal-gas mixture properties per unit mass.
 *
 * A plain aggregate of Views, captured by value in kernels. Mixture
 * functions take the mass fractions as any callable y(k).
 */
struct ThermoTable {
    uint32_t n_species = 0;
    Kokkos::View<double *> inv_W;               // kmol/kg
    Kokkos::View<uint32_t *> range_offset;      // (n_species + 1): ranges of species k
    Kokkos::View<double *> range_upper;         // (range): upper bound; the last range of a species is unbounded
    Kokkos::View<uint8_t *> upper_closed;       // (species): a bound belongs to the range below it (NASA-7)
    Kokkos::View<double *[9], Kokkos::LayoutRight> coeffs;  // (range, coefficient)
    double T_low = 1.0;                          // bracket of T(e)
    double T_high = 1.0e5;

    /** @brief Powers of T shared by all species. */
    struct Powers {
        double T, inv_T, inv_T2, log_T;
    };

    KOKKOS_INLINE_FUNCTION
    static Powers powers(const double T) {
        const double inv_T = 1.0 / T;
        return {T, inv_T, inv_T * inv_T, Kokkos::log(T)};
    }

    KOKKOS_INLINE_FUNCTION
    uint32_t range(const uint32_t k, const double T) const {
        uint32_t r = range_offset(k);
        const uint32_t last = range_offset(k + 1) - 1;
        if (upper_closed(k)) {
            while (r < last && T > range_upper(r)) r++;
        } else {
            while (r < last && T >= range_upper(r)) r++;
        }
        return r;
    }

    /** @brief cp_k / R. */
    KOKKOS_INLINE_FUNCTION
    double cp_R(const uint32_t k, const Powers & p) const {
        const uint32_t r = range(k, p.T);
        const double T = p.T;
        return coeffs(r, 0) * p.inv_T2 + coeffs(r, 1) * p.inv_T +
               (coeffs(r, 2) + T * (coeffs(r, 3) + T * (coeffs(r, 4) + T * (coeffs(r, 5) + T * coeffs(r, 6)))));
    }

    /** @brief h_k / (R T), molar enthalpy including the formation enthalpy. */
    KOKKOS_INLINE_FUNCTION
    double h_RT(const uint32_t k, const Powers & p) const {
        const uint32_t r = range(k, p.T);
        const double T = p.T;
        return -coeffs(r, 0) * p.inv_T2 + coeffs(r, 1) * p.log_T * p.inv_T +
               (coeffs(r, 2) + T * (coeffs(r, 3) / 2.0 + T * (coeffs(r, 4) / 3.0 +
                                    T * (coeffs(r, 5) / 4.0 + T * coeffs(r, 6) / 5.0)))) +
               coeffs(r, 7) * p.inv_T;
    }

    /** @brief s_k / R at the standard pressure. */
    KOKKOS_INLINE_FUNCTION
    double s_R(const uint32_t k, const Powers & p) const {
        const uint32_t r = range(k, p.T);
        const double T = p.T;
        return -0.5 * coeffs(r, 0) * p.inv_T2 - coeffs(r, 1) * p.inv_T + coeffs(r, 2) * p.log_T +
               T * (coeffs(r, 3) + T * (coeffs(r, 4) / 2.0 + T * (coeffs(r, 5) / 3.0 + T * coeffs(r, 6) / 4.0))) +
               coeffs(r, 8);
    }

    /** @brief Mixture gas constant R_u sum_k Y_k / W_k [J/(kg K)]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double gas_constant(const F_Y & y) const {
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k);
        return GAS_CONSTANT * sum;
    }

    /** @brief Mixture cp [J/(kg K)]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double cp_mass(const double T, const F_Y & y) const {
        const Powers p = powers(T);
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k) * cp_R(k, p);
        return GAS_CONSTANT * sum;
    }

    /** @brief Mixture enthalpy [J/kg]. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double h_mass(const double T, const F_Y & y) const {
        const Powers p = powers(T);
        double sum = 0.0;
        for (uint32_t k = 0; k < n_species; k++) sum += y(k) * inv_W(k) * h_RT(k, p);
        return GAS_CONSTANT * T * sum;
    }

    /** @brief Mixture internal energy e [J/kg] and cv [J/(kg K)] at T. */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION void e_cv(const double T, const F_Y & y, double & e, double & cv) const {
        const Powers p = powers(T);
        double sum_e = 0.0, sum_cv = 0.0;
        for (uint32_t k = 0; k < n_species; k++) {
            const double n = y(k) * inv_W(k);
            sum_e += n * (h_RT(k, p) - 1.0);
            sum_cv += n * (cp_R(k, p) - 1.0);
        }
        e = GAS_CONSTANT * T * sum_e;
        cv = GAS_CONSTANT * sum_cv;
    }

    /**
     * @brief Temperature at which the mixture has internal energy e: Newton
     *        from T_guess (300 K if outside the bracket), safeguarded by
     *        bisection within [T_low, T_high], until the update is below 1e-10 T.
     */
    template <typename F_Y>
    KOKKOS_INLINE_FUNCTION double T_from_e(const double e, const F_Y & y, const double T_guess) const {
        double a = T_low, b = T_high;
        double T = (T_guess > a && T_guess < b) ? T_guess : Kokkos::fmin(Kokkos::fmax(300.0, a), b);
        for (int it = 0; it < 200; it++) {
            double e_T, cv;
            e_cv(T, y, e_T, cv);
            const double f = e_T - e;
            if (f > 0.0) {
                b = T;
            } else {
                a = T;
            }
            double T_new = T - f / cv;
            if (!(T_new >= a && T_new <= b)) T_new = 0.5 * (a + b);
            if (Kokkos::fabs(T_new - T) <= 1e-10 * T) return T_new;
            T = T_new;
        }
        return T;
    }
};

/** @brief Copy a mechanism's species thermodynamics to the device. */
ThermoTable make_thermo_table(const Mechanism & mechanism);

} // namespace chemistry

#endif // CHEMISTRY_THERMO_H
