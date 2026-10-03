/**
 * @file thermo.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Device tables of species thermodynamics.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "thermo.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace chemistry {

ThermoTable make_thermo_table(const Mechanism & mechanism) {
    ThermoTable table;
    const uint32_t n = static_cast<uint32_t>(mechanism.n_species());
    table.n_species = n;
    uint32_t n_ranges = 0;
    for (const auto & sp : mechanism.species) n_ranges += static_cast<uint32_t>(sp.thermo.coeffs.size());

    table.inv_W = Kokkos::View<double *>("thermo_inv_W", n);
    table.range_offset = Kokkos::View<uint32_t *>("thermo_range_offset", n + 1);
    table.range_upper = Kokkos::View<double *>("thermo_range_upper", n_ranges);
    table.upper_closed = Kokkos::View<uint8_t *>("thermo_upper_closed", n);
    table.coeffs = Kokkos::View<double *[9], Kokkos::LayoutRight>("thermo_coeffs", n_ranges);
    auto h_inv_W = Kokkos::create_mirror_view(table.inv_W);
    auto h_offset = Kokkos::create_mirror_view(table.range_offset);
    auto h_upper = Kokkos::create_mirror_view(table.range_upper);
    auto h_closed = Kokkos::create_mirror_view(table.upper_closed);
    auto h_coeffs = Kokkos::create_mirror_view(table.coeffs);

    // Common range of the fits, where every species is fitted
    double T_min = 0.0, T_max = std::numeric_limits<double>::infinity();
    uint32_t r = 0;
    for (uint32_t k = 0; k < n; k++) {
        const Species & sp = mechanism.species[k];
        h_inv_W(k) = 1.0 / sp.molecular_weight;
        h_offset(k) = r;
        h_closed(k) = sp.thermo.model == ThermoModel::NASA7;
        for (size_t i = 0; i < sp.thermo.coeffs.size(); i++, r++) {
            h_upper(r) = sp.thermo.T_bounds[i + 1];
            for (int j = 0; j < 9; j++) h_coeffs(r, j) = sp.thermo.coeffs[i][j];
        }
        T_min = std::max(T_min, sp.thermo.T_bounds.front());
        T_max = std::min(T_max, sp.thermo.T_bounds.back());
    }
    h_offset(n) = r;
    // T(e) may extrapolate somewhat past the common range; constant cp has no bounds
    table.T_low = std::max(0.5 * T_min, 1.0);
    table.T_high = std::isfinite(T_max) ? 2.0 * T_max : 1.0e5;

    Kokkos::deep_copy(table.inv_W, h_inv_W);
    Kokkos::deep_copy(table.range_offset, h_offset);
    Kokkos::deep_copy(table.range_upper, h_upper);
    Kokkos::deep_copy(table.upper_closed, h_closed);
    Kokkos::deep_copy(table.coeffs, h_coeffs);
    return table;
}

} // namespace chemistry
