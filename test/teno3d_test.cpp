/**
 * @file teno3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for the TENO-E reconstruction on 3D meshes.
 * @version 0.1
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <string>
#include <tuple>

#include "test_fixtures.h"
#include "face_reconstruction.h"
#include "physics.h"

namespace {

constexpr rtype GAMMA = 1.4;

// Smooth field satisfying symmetry conditions on the unit cube: density,
// pressure and tangential velocities are even across each wall, normal
// velocity odd
void smooth_conservatives(double x, double y, double z, double * U) {
    const double rho = 1.0 + 0.2 * std::cos(2.0 * M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
    const double u = 0.1 * std::sin(2.0 * M_PI * x) * std::cos(M_PI * y) * std::cos(M_PI * z);
    const double v = -0.1 * std::sin(M_PI * y) * std::cos(M_PI * x) * std::cos(M_PI * z);
    const double w = 0.1 * std::sin(M_PI * z) * std::cos(M_PI * x) * std::cos(2.0 * M_PI * y);
    const double p = 1.0 + 0.1 * std::cos(M_PI * x) * std::cos(2.0 * M_PI * y) * std::cos(M_PI * z);
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
}

// Quadratic density (velocity and energy constant in conservative form)
void quadratic_conservatives(double x, double y, double z, double * U) {
    U[0] = 1.0 + 0.3 * x * x + 0.2 * x * y - 0.1 * z * z + 0.2 * y * z + 0.1 * x - 0.2 * z;
    U[1] = 0.1;
    U[2] = -0.1;
    U[3] = 0.05;
    U[4] = 10.0;
}

using Field = void (*)(double, double, double, double *);

std::unique_ptr<TENO> make_teno(std::shared_ptr<Mesh> mesh, const BoundaryData & bd, int order,
                                const std::string & extra = "") {
    auto teno = std::make_unique<TENO>();
    teno->set_mesh(mesh);
    teno->set_boundaries(bd);
    teno->init(parse_toml("type = \"TENO\"\norder = " + std::to_string(order) + "\n" + extra));
    return teno;
}

/**
 * @brief Max density error of the reconstruction at the face quadrature points
 *        (both sides), optionally only on faces at least margin from the walls.
 */
double reconstruction_error(const std::string & mesh_type, uint32_t n, int order, double margin = 0.0,
                            const std::string & extra = "", Field field = smooth_conservatives) {
    auto mesh = make_mesh_3d(mesh_type, n, n, n);
    BoundaryData bd = make_uniform_boundaries(*mesh, BoundaryType::SYMMETRY, GAMMA);
    auto avg = cell_averages_3d(*mesh, field);
    Euler euler = Euler::from_reference(GAMMA, 1.0, 1.0, 1.0);
    Kokkos::View<rtype *[N_CONSERVATIVE]> W("W", mesh->n_cells);
    auto h_W = Kokkos::create_mirror_view(W);
    for (uint32_t c = 0; c < mesh->n_cells; c++) {
        rtype U[N_CONSERVATIVE], Wc[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE U[i] = avg(c, i);
        euler.compute_W_from_conservatives(Wc, U);
        FOR_I_CONSERVATIVE h_W(c, i) = Wc[i];
    }
    Kokkos::deep_copy(W, h_W);

    auto teno = make_teno(mesh, bd, order, extra);
    const uint8_t n_quad = teno->n_face_quadrature_points();
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_W("face_W", mesh->n_faces, n_quad);
    teno->calc_face_values(W, face_W);
    auto h_face_W = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), face_W);
    auto h_points = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_points);
    auto h_weights = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), teno->face_quad_weights);

    double err = 0.0;
    for (uint32_t f = 0; f < mesh->n_faces; f++) {
        for (uint8_t q = 0; q < n_quad; q++) {
            if (h_weights(f, q) == 0.0) continue;
            const double x = h_points(f, q, 0), y = h_points(f, q, 1), z = h_points(f, q, 2);
            if (std::min({x, y, z}) < margin || std::max({x, y, z}) > 1.0 - margin) continue;
            double U[N_CONSERVATIVE];
            field(x, y, z, U);
            for (uint8_t side = 0; side < 2; side++) {
                if (mesh->h_cells_of_face(f, side) < 0) continue;
                err = std::max(err, std::abs(h_face_W(f, q, side, 0) - U[0]));
            }
        }
    }
    return err;
}

// Smooth: every cell takes the central stencil. Troubled: every cell runs the
// stencil selection, which must keep the central stencil on smooth data.
const char * SMOOTH = "troubled_threshold = 1e9\n";
const char * TROUBLED = "troubled_threshold = 0\ncharacteristic = false\n";

class TENO3DExactness : public ::testing::TestWithParam<std::string> {};

} // namespace

TEST_P(TENO3DExactness, ReproducesQuadraticsAwayFromWalls) {
    // Exactness checks the cell averages of the basis, the stencil
    // least-squares system (including rank detection) and the face quadrature
    // points; the wall mirrors are only exact for symmetric fields
    EXPECT_LT(reconstruction_error(GetParam(), 8, 3, 0.35, SMOOTH, quadratic_conservatives), 1e-11);
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DExactness,
                         ::testing::Values("cartesian", "cartesian_tet", "cartesian_prism", "cartesian_pyramid",
                                           "cartesian_mixed"));

namespace {

using OrderParam = std::tuple<std::string, int, uint32_t, const char *>;
class TENO3DOrder : public ::testing::TestWithParam<OrderParam> {};

} // namespace

TEST_P(TENO3DOrder, SmoothReconstructionConvergesAtDesignOrder) {
    // Max error at face quadrature points over all faces, with mirrored
    // stencils at the symmetry walls
    const auto [mesh_type, order, n, mode] = GetParam();
    const double e1 = reconstruction_error(mesh_type, n, order, 0.0, mode);
    const double e2 = reconstruction_error(mesh_type, 2 * n, order, 0.0, mode);
    const double rate = std::log2(e1 / e2);
    std::cout << mesh_type << " order " << order << ": max errors " << e1 << ", " << e2 << ", rate " << rate
              << std::endl;
    EXPECT_GT(rate, order - 0.5);
}

INSTANTIATE_TEST_SUITE_P(TENO, TENO3DOrder,
    ::testing::Values(OrderParam{"cartesian", 3, 6, SMOOTH}, OrderParam{"cartesian", 4, 6, SMOOTH},
                      OrderParam{"cartesian", 5, 6, SMOOTH}, OrderParam{"cartesian_tet", 3, 4, SMOOTH},
                      OrderParam{"cartesian_tet", 4, 4, SMOOTH}, OrderParam{"cartesian_tet", 5, 4, SMOOTH},
                      OrderParam{"cartesian", 5, 6, TROUBLED},
                      OrderParam{"cartesian_tet", 4, 4, TROUBLED}));

