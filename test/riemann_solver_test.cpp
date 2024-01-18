/**
 * @file riemann_solver_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for Riemann solvers
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include "riemann_solver.h"

TEST(RiemannSolverTest, HLLCFlux) {
    rtype gamma = 1.4;

    State flux;
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 0.1;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit,
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-1;
    EXPECT_NEAR(flux[0], 0.415322226496596, tol);
    EXPECT_NEAR(flux[1], 0.508584114470313, tol);
    EXPECT_NEAR(flux[2], 0.000000000000000, tol);
    EXPECT_NEAR(flux[3], 1.139144729421316, tol);
}

TEST(RiemannSolverTest, HLLCFluxZero) {
    rtype gamma = 1.4;

    State flux;
    NVector n_unit = {1.0, 0.0};

    rtype rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    rtype p_l = 1.0;

    rtype rho_r = 1.0;
    NVector u_r = {0.0, 0.0};
    rtype p_r = 1.0;

    rtype e_r = p_r / ((gamma - 1.0) * rho_r);
    rtype e_l = p_l / ((gamma - 1.0) * rho_l);
    rtype h_l = e_l + p_l / rho_l;
    rtype h_r = e_r + p_r / rho_r;

    HLLC riemann_solver;
    riemann_solver.calc_flux(flux, n_unit,
                             rho_l, u_l.data(), p_l, gamma, h_l,
                             rho_r, u_r.data(), p_r, gamma, h_r);

    rtype tol = 1e-1;
    EXPECT_NEAR(flux[0], 0.000000000000000, tol);
    EXPECT_NEAR(flux[1], 1.000000000000000, tol);
    EXPECT_NEAR(flux[2], 0.000000000000000, tol);
    EXPECT_NEAR(flux[3], 0.000000000000000, tol);
}