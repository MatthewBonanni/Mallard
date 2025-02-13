/**
 * @file riemann_solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Riemann solver class implementation
 * @version 0.1
 * @date 2024-01-17
 * 
 * @copyright Copyright (c) 2024 Matthew Bonanni
 * 
 */

#include "riemann_solver.h"

#include <iostream>

#include <Kokkos_Core.hpp>

#include "common_io.h"

RiemannSolver::RiemannSolver() {
    // Empty
}

RiemannSolver::~RiemannSolver() {
    // Empty
}

void RiemannSolver::init(const toml::value & input) {
    (void)(input);
    print();
}

void RiemannSolver::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Riemann solver: " << RIEMANN_SOLVER_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

Rusanov::Rusanov() {
    type = RiemannSolverType::RUSANOV;
}

Rusanov::~Rusanov() {
    // Empty
}

HLL::HLL() {
    type = RiemannSolverType::HLL;
}

HLL::~HLL() {
    // Empty
}

HLLC::HLLC() {
    type = RiemannSolverType::HLLC;
}

HLLC::~HLLC() {
    // Empty
}