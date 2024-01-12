/**
 * @file common_math_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Tests for common/common_math
 * @version 0.1
 * @date 2023-12-19
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include <gtest/gtest.h>
#include "test_utils.h"
#include "common_math.h"

TEST(CommonMathTest, Dot) {
    NVector v0 = {1.0, 2.0};
    NVector v1 = {3.0, 4.0};

    rtype expected_dot = 11.0;
    rtype actual_dot = dot<2>(v0.data(), v1.data());

    EXPECT_RTYPE_EQ(expected_dot, actual_dot);
}

TEST(CommonMathTest, Dot3) {
    std::vector<rtype> v0 = {1.0, 2.0, 3.0};
    std::vector<rtype> v1 = {4.0, 5.0, 6.0};

    rtype expected_dot = 32.0;
    rtype actual_dot = dot<3>(v0.data(), v1.data());

    EXPECT_RTYPE_EQ(expected_dot, actual_dot);
}

TEST(CommonMathTest, Norm2) {
    NVector v = {3.0, 4.0};

    rtype expected_norm = 5.0;
    rtype actual_norm = norm_2(v);

    EXPECT_RTYPE_EQ(expected_norm, actual_norm);
}

TEST(CommonMathTest, Unit) {
    NVector v = {3.0, 4.0};

    NVector expected_unit = {0.6, 0.8};
    NVector actual_unit = unit(v);

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, UnitNegative) {
    NVector v = {-3.0, -4.0};

    NVector expected_unit = {-0.6, -0.8};
    NVector actual_unit = unit(v);

    EXPECT_RTYPE_EQ(expected_unit[0], actual_unit[0]);
    EXPECT_RTYPE_EQ(expected_unit[1], actual_unit[1]);
}

TEST(CommonMathTest, TriangleArea2) {
    NVector v0 = {0.0, 0.0};
    NVector v1 = {1.0, 0.0};
    NVector v2 = {0.0, 1.0};

    rtype expected_area = 0.5;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2NegativeCoordinates) {
    NVector v0 = {-1.0, -1.0};
    NVector v1 = {1.0, -1.0};
    NVector v2 = {-1.0, 1.0};

    rtype expected_area = 2.0;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, TriangleArea2ZeroArea) {
    NVector v0 = {0.0, 0.0};
    NVector v1 = {0.0, 0.0};
    NVector v2 = {0.0, 0.0};

    rtype expected_area = 0.0;
    rtype actual_area = triangle_area_2(v0, v1, v2);

    EXPECT_RTYPE_EQ(expected_area, actual_area);
}

TEST(CommonMathTest, LinearCombination) {
    std::vector<StateVector *> vectors_in;
    StateVector * vector_out;
    std::vector<rtype> coefficients = {1.0, 2.0};

    vectors_in.push_back(new StateVector(2));
    vectors_in.push_back(new StateVector(2));
    vector_out = new StateVector(2);

    (*vectors_in[0])[0][0] = 0.0;
    (*vectors_in[0])[0][1] = 1.0;
    (*vectors_in[0])[0][2] = 2.0;
    (*vectors_in[0])[0][3] = 3.0;
    (*vectors_in[0])[1][0] = 4.0;
    (*vectors_in[0])[1][1] = 5.0;
    (*vectors_in[0])[1][2] = 6.0;
    (*vectors_in[0])[1][3] = 7.0;

    (*vectors_in[1])[0][0] = 8.0;
    (*vectors_in[1])[0][1] = 9.0;
    (*vectors_in[1])[0][2] = 10.0;
    (*vectors_in[1])[0][3] = 11.0;
    (*vectors_in[1])[1][0] = 12.0;
    (*vectors_in[1])[1][1] = 13.0;
    (*vectors_in[1])[1][2] = 14.0;
    (*vectors_in[1])[1][3] = 15.0;

    linear_combination(vectors_in, vector_out, coefficients);

    EXPECT_RTYPE_EQ((*vector_out)[0][0], 16.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][1], 19.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][2], 22.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][3], 25.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][0], 28.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][1], 31.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][2], 34.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][3], 37.0);
}

TEST(CommonMathTest, LinearCombinationWrongNumberOfCoefficients) {
    std::vector<StateVector *> vectors_in;
    StateVector * vector_out;
    std::vector<rtype> coefficients = {1.0, 2.0, 3.0};

    vectors_in.push_back(new StateVector(2));
    vectors_in.push_back(new StateVector(2));
    vector_out = new StateVector(2);

    (*vectors_in[0])[0][0] = 0.0;
    (*vectors_in[0])[0][1] = 1.0;
    (*vectors_in[0])[0][2] = 2.0;
    (*vectors_in[0])[0][3] = 3.0;
    (*vectors_in[0])[1][0] = 4.0;
    (*vectors_in[0])[1][1] = 5.0;
    (*vectors_in[0])[1][2] = 6.0;
    (*vectors_in[0])[1][3] = 7.0;

    (*vectors_in[1])[0][0] = 8.0;
    (*vectors_in[1])[0][1] = 9.0;
    (*vectors_in[1])[0][2] = 10.0;
    (*vectors_in[1])[0][3] = 11.0;
    (*vectors_in[1])[1][0] = 12.0;
    (*vectors_in[1])[1][1] = 13.0;
    (*vectors_in[1])[1][2] = 14.0;
    (*vectors_in[1])[1][3] = 15.0;

    EXPECT_THROW(linear_combination(vectors_in, vector_out, coefficients), std::runtime_error);
}

TEST(CommonMathTest, LinearCombinationOutputVectorIsInputVector) {
    std::vector<StateVector *> vectors_in;
    StateVector * vector_out;
    std::vector<rtype> coefficients = {1.0, 2.0};

    vectors_in.push_back(new StateVector(2));
    vectors_in.push_back(new StateVector(2));
    vector_out = vectors_in[0];

    (*vectors_in[0])[0][0] = 0.0;
    (*vectors_in[0])[0][1] = 1.0;
    (*vectors_in[0])[0][2] = 2.0;
    (*vectors_in[0])[0][3] = 3.0;
    (*vectors_in[0])[1][0] = 4.0;
    (*vectors_in[0])[1][1] = 5.0;
    (*vectors_in[0])[1][2] = 6.0;
    (*vectors_in[0])[1][3] = 7.0;

    (*vectors_in[1])[0][0] = 8.0;
    (*vectors_in[1])[0][1] = 9.0;
    (*vectors_in[1])[0][2] = 10.0;
    (*vectors_in[1])[0][3] = 11.0;
    (*vectors_in[1])[1][0] = 12.0;
    (*vectors_in[1])[1][1] = 13.0;
    (*vectors_in[1])[1][2] = 14.0;
    (*vectors_in[1])[1][3] = 15.0;

    linear_combination(vectors_in, vector_out, coefficients);

    EXPECT_RTYPE_EQ((*vector_out)[0][0], 16.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][1], 19.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][2], 22.0);
    EXPECT_RTYPE_EQ((*vector_out)[0][3], 25.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][0], 28.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][1], 31.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][2], 34.0);
    EXPECT_RTYPE_EQ((*vector_out)[1][3], 37.0);
}