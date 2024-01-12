/**
 * @file common_math.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Common math functions.
 * @version 0.1
 * @date 2023-12-18
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef COMMON_MATH_H
#define COMMON_MATH_H

#include <array>
#include <vector>

#include "common_typedef.h"

/**
 * @brief Compute the dot product of two arrays.
 * 
 * @param v0 First array.
 * @param v1 Second array.
 * @param n Length of the arrays.
 * @tparam T Type of the arrays.
 * @return Dot product.
 */
template <typename T>
T dot(const T * const v0, const T * const v1, const int n) {
    T dot = 0.0;
    for (int i = 0; i < n; i++) {
        dot += v0[i] * v1[i];
    }
    return dot;
}

/**
 * @brief Compute the NVector dotted with itself.
 * @param v Vector.
 */
rtype dot_self(const NVector& v);

/**
 * @brief Compute the 2-norm of an NVector.
 * @param v Vector.
 * @return Norm.
 */
rtype norm_2(const NVector& v);

/**
 * @brief Compute the unit vector of an NVector.
 * @param v Vector.
 * @return Unit vector.
 */
NVector unit(const NVector & v);

/**
 * @brief Compute the area of a triangle from vertices in R^2.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
rtype triangle_area_2(const NVector& v0,
                      const NVector& v1,
                      const NVector& v2);

/**
 * @brief Compute the area of a triangle from vertices in R^3.
 * 
 * @param v0 Coordinates of the first vertex.
 * @param v1 Coordinates of the second vertex.
 * @param v2 Coordinates of the third vertex.
 * @return Area of the triangle.
 */
rtype triangle_area_3(const std::array<rtype, 3>& v0,
                      const std::array<rtype, 3>& v1,
                      const std::array<rtype, 3>& v2);

/**
 * @brief Compute the linear combination of a vector of vectors.
 * 
 * @param vectors_in
 * @param vector_out
 * @param coefficients
 */
void linear_combination(const std::vector<StateVector *> & vectors_in,
                        StateVector * const vector_out,
                        const std::vector<rtype> & coefficients);

/**
 * @brief Compute the extremum of each element in a vector of arrays.
 * 
 * @param arrays
 * @param comp Comparator.
 * @return std::array<rtype, N> Extremum of each element.
 */
template <int N, typename Compare>
std::array<rtype, N> extrema_array(const std::vector<std::array<rtype, N>> &arrays,
                                   Compare comp) {
    std::array<rtype, N> extrema;
    for (int i = 0; i < N; ++i) {
        extrema[i] = arrays[0][i];
        for (int j = 1; j < arrays.size(); ++j) {
            if (comp(arrays[j][i], extrema[i])) {
                extrema[i] = arrays[j][i];
            }
        }
    }
    return extrema;
}

/**
 * @brief Compute the maximum of each element in a vector of arrays.
 * 
 * @param arrays
 * @return Maximum of each element.
 */
template <int N>
std::array<rtype, N> max_array(const std::vector<std::array<rtype, N>> & arrays) {
    return extrema_array<N>(arrays, std::greater<rtype>());
}

/**
 * @brief Compute the minimum of each element in a vector of arrays.
 * 
 * @param arrays
 * @return Minimum of each element.
 */
template <int N>
std::array<rtype, N> min_array(const std::vector<std::array<rtype, N>> &arrays) {
    return extrema_array<N>(arrays, std::less<rtype>());
}

#endif // COMMON_MATH_H