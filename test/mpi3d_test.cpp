/**
 * @file mpi3d_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Distributed 3D runs reproduce the serial solution, on any number of ranks.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "mpi_compare.h"

namespace {

const char * ZONES[6] = {"left", "right", "bottom", "top", "back", "front"};

std::string blast_input(const std::string & mesh, const std::string & recon) {
    const std::string blob = "exp(-20 * ((x - 0.4)^2 + (y - 0.55)^2 + (z - 0.45)^2))";
    std::ostringstream s;
    s << "[run]\nn_steps = 3\ncfl = 0.5\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 12\nNy = 12\nNz = 12\nLx = 1.0\nLy = 1.0\nLz = 1.0\n"
      << "[initialize]\ntype = \"analytical\"\nrho = \"1.0 + 0.5 * " << blob << "\"\n"
      << "u = [\"0.3\", \"-0.1\", \"0.2\"]\np = \"1.0 + 0.8 * " << blob << "\"\n";
    for (int k = 0; k < 6; k++) {
        s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\ntype = \""
          << (k % 2 ? "extrapolation" : "symmetry") << "\"\n";
    }
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\ntype = \"euler\"\ngamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

std::string box_input(const std::string & mesh, const std::string & recon, const std::string & physics) {
    const char * types[6] = {"type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"wall_adiabatic\"\n",
                             "type = \"extrapolation\"\n", "type = \"symmetry\"\n", "type = \"extrapolation\"\n"};
    std::ostringstream s;
    s << "[run]\nn_steps = 8\ncfl = 0.5\n"
      << "[mesh]\ntype = \"" << mesh << "\"\nNx = 6\nNy = 4\nNz = 3\nLx = 1.0\nLy = 0.8\nLz = 0.6\n"
      << "[initialize]\ntype = \"analytical\"\n"
      << "rho = \"1.0 + 0.8 * exp(-20 * ((x - 0.35)^2 + (y - 0.45)^2 + (z - 0.3)^2))\"\n"
      << "u = [\"0.3\", \"-0.1\", \"0.2\"]\n"
      << "p = \"1.0 + 2.0 * exp(-20 * ((x - 0.35)^2 + (y - 0.45)^2 + (z - 0.3)^2))\"\n";
    for (int k = 0; k < 6; k++) s << "[[boundaries]]\nname = \"" << ZONES[k] << "\"\n" << types[k];
    s << "[numerics]\nriemann_solver = \"HLLC\"\ntime_integrator = \"SSPRK3\"\n"
      << "[numerics.face_reconstruction]\n" << recon
      << "[physics]\n" << physics << "gamma = 1.4\np_ref = 1.0\nT_ref = 1.0\nrho_ref = 1.0\n"
      << "[output]\ncheck_interval = 1000000\n";
    return s.str();
}

} // namespace

TEST(MPI3DTest, TENOOnHexahedraMatchesSerial) {
    // Order 5 needs five distinct cell planes across a planar partition
    // interface; a halo that cuts the stencil search short must be deepened
    // rather than reported as a rank-deficient stencil
    expect_matches_serial(blast_input("cartesian", "type = \"TENO\"\norder = 5\n"));
}

TEST(MPI3DTest, MUSCLOnMixedCellsMatchesSerial) {
    expect_matches_serial(box_input("cartesian_mixed", "type = \"MUSCL\"\n", "type = \"euler\"\n"));
}

TEST(MPI3DTest, NavierStokesOnTetrahedraMatchesSerial) {
    expect_matches_serial(box_input("cartesian_tet", "type = \"MUSCL\"\n",
                                    "type = \"navier_stokes\"\nmu = 0.01\nPr = 0.72\n"));
}
