/**
 * @file physics_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for physics
 * @version 0.1
 * @date 2024-01-05
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "physics/physics.h"

TEST(PhysicsTest, EulerFlux) {
    Euler physics;
    double gamma = 1.4;
    double p_ref = 101325.0;
    double T_ref = 298.15;
    double rho_ref = 1.225;
    physics.init(gamma, p_ref, T_ref, rho_ref);

    State flux;
    NVector n_unit = {1.0, 0.0};

    double rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    double p_l = 1.0;

    double rho_r = 0.125;
    NVector u_r = {0.0, 0.0};
    double p_r = 0.1;

    double e_r = p_r / ((gamma - 1.0) * rho_r);
    double e_l = p_l / ((gamma - 1.0) * rho_l);
    double h_l = e_l + p_l / rho_l;
    double h_r = e_r + p_r / rho_r;

    physics.calc_euler_flux(flux, n_unit, rho_l, u_l, p_l, gamma, h_l, rho_r, u_r, p_r, gamma, h_r);

    double tol = 1e-1;
    EXPECT_NEAR(flux[0], 0.415322226496596, tol);
    EXPECT_NEAR(flux[1], 0.508584114470313, tol);
    EXPECT_NEAR(flux[2], 0.000000000000000, tol);
    EXPECT_NEAR(flux[3], 1.139144729421316, tol);
}

TEST(PhysicsTest, EulerFluxZero) {
    Euler physics;
    double gamma = 1.4;
    double p_ref = 101325.0;
    double T_ref = 298.15;
    double rho_ref = 1.225;
    physics.init(gamma, p_ref, T_ref, rho_ref);

    State flux;
    NVector n_unit = {1.0, 0.0};

    double rho_l = 1.0;
    NVector u_l = {0.0, 0.0};
    double p_l = 1.0;

    double rho_r = 1.0;
    NVector u_r = {0.0, 0.0};
    double p_r = 1.0;

    double e_r = p_r / ((gamma - 1.0) * rho_r);
    double e_l = p_l / ((gamma - 1.0) * rho_l);
    double h_l = e_l + p_l / rho_l;
    double h_r = e_r + p_r / rho_r;

    physics.calc_euler_flux(flux, n_unit, rho_l, u_l, p_l, gamma, h_l, rho_r, u_r, p_r, gamma, h_r);

    EXPECT_DOUBLE_EQ(flux[0], 0.0);
    EXPECT_DOUBLE_EQ(flux[1], 1.0);
    EXPECT_DOUBLE_EQ(flux[2], 0.0);
    EXPECT_DOUBLE_EQ(flux[3], 0.0);
}