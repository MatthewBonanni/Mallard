/**
 * @file time_integrator_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for numerics/time_integrator
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "numerics/time_integrator.h"

void calc_rhs_test(std::vector<std::array<double, 4>> * solution,
                   std::vector<std::array<double, 4>> * rhs) {
    (*rhs)[0][0] = 0.0 * (*solution)[0][0] + 0.0;
    (*rhs)[0][1] = 0.0 * (*solution)[0][1] + 1.0;
    (*rhs)[0][2] = 0.0 * (*solution)[0][2] + 2.0;
    (*rhs)[0][3] = 0.0 * (*solution)[0][3] + 3.0;
    (*rhs)[1][0] = 0.0 * (*solution)[1][0] + 4.0;
    (*rhs)[1][1] = 0.0 * (*solution)[1][1] + 5.0;
    (*rhs)[1][2] = 0.0 * (*solution)[1][2] + 6.0;
    (*rhs)[1][3] = 0.0 * (*solution)[1][3] + 7.0;
}

TEST(TimeIntegratorTest, ForwardEuler) {
    ForwardEuler forward_euler;
    std::vector<std::vector<std::array<double, 4>> *> solution_pointers;
    std::vector<std::vector<std::array<double, 4>> *> rhs_pointers;
    std::vector<std::array<double, 4>> * solution;
    std::vector<double> coefficients = {1.0, 2.0};

    solution_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    rhs_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    solution = solution_pointers[0];

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    forward_euler.take_step(0.1, solution_pointers, rhs_pointers, calc_rhs_test);

    EXPECT_DOUBLE_EQ((*solution)[0][0], 0.0);
    EXPECT_DOUBLE_EQ((*solution)[0][1], 1.1);
    EXPECT_DOUBLE_EQ((*solution)[0][2], 2.2);
    EXPECT_DOUBLE_EQ((*solution)[0][3], 3.3);
    EXPECT_DOUBLE_EQ((*solution)[1][0], 4.4);
    EXPECT_DOUBLE_EQ((*solution)[1][1], 5.5);
    EXPECT_DOUBLE_EQ((*solution)[1][2], 6.6);
    EXPECT_DOUBLE_EQ((*solution)[1][3], 7.7);
}

TEST(TimeIntegratorTest, RK4) {
    RK4 rk4;
    std::vector<std::vector<std::array<double, 4>> *> solution_pointers;
    std::vector<std::vector<std::array<double, 4>> *> rhs_pointers;
    std::vector<std::array<double, 4>> * solution;
    std::vector<double> coefficients = {1.0, 2.0};

    for (int i = 0; i < 2; i++) {
        solution_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    }
    for (int i = 0; i < 4; i++) {
        rhs_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    }
    solution = solution_pointers[0];

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    rk4.take_step(0.1, solution_pointers, rhs_pointers, calc_rhs_test);

    EXPECT_DOUBLE_EQ((*solution)[0][0], 0.0);
    EXPECT_DOUBLE_EQ((*solution)[0][1], 1.1);
    EXPECT_DOUBLE_EQ((*solution)[0][2], 2.2);
    EXPECT_DOUBLE_EQ((*solution)[0][3], 3.3);
    EXPECT_DOUBLE_EQ((*solution)[1][0], 4.4);
    EXPECT_DOUBLE_EQ((*solution)[1][1], 5.5);
    EXPECT_DOUBLE_EQ((*solution)[1][2], 6.6);
    EXPECT_DOUBLE_EQ((*solution)[1][3], 7.7);
}

TEST(TimeIntegratorTest, SSPRK3) {
    SSPRK3 ssprk3;
    std::vector<std::vector<std::array<double, 4>> *> solution_pointers;
    std::vector<std::vector<std::array<double, 4>> *> rhs_pointers;
    std::vector<std::array<double, 4>> * solution;
    std::vector<double> coefficients = {1.0, 2.0};

    for (int i = 0; i < 2; i++) {
        solution_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    }
    for (int i = 0; i < 3; i++) {
        rhs_pointers.push_back(new std::vector<std::array<double, 4>>(2));
    }
    solution = solution_pointers[0];

    (*solution)[0][0] = 0.0;
    (*solution)[0][1] = 1.0;
    (*solution)[0][2] = 2.0;
    (*solution)[0][3] = 3.0;
    (*solution)[1][0] = 4.0;
    (*solution)[1][1] = 5.0;
    (*solution)[1][2] = 6.0;
    (*solution)[1][3] = 7.0;

    ssprk3.take_step(0.1, solution_pointers, rhs_pointers, calc_rhs_test);

    EXPECT_DOUBLE_EQ((*solution)[0][0], 0.0);
    EXPECT_DOUBLE_EQ((*solution)[0][1], 1.1);
    EXPECT_DOUBLE_EQ((*solution)[0][2], 2.2);
    EXPECT_DOUBLE_EQ((*solution)[0][3], 3.3);
    EXPECT_DOUBLE_EQ((*solution)[1][0], 4.4);
    EXPECT_DOUBLE_EQ((*solution)[1][1], 5.5);
    EXPECT_DOUBLE_EQ((*solution)[1][2], 6.6);
    EXPECT_DOUBLE_EQ((*solution)[1][3], 7.7);
}