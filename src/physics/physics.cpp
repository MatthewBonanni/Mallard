/**
 * @file physics.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Physics class implementation.
 * @version 0.1
 * @date 2023-12-27
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "physics.h"

#include <iostream>

#include "common.h"

Physics::Physics() {
    p_bounds = Kokkos::View<rtype[2]>("p_bounds");
    h_p_bounds = Kokkos::create_mirror_view(p_bounds);
}

Physics::~Physics() {
    std::cout << "Destroying physics: " << PHYSICS_NAMES.at(type) << std::endl;
}

void Physics::init(const toml::value & input) {
    h_p_bounds(0) = toml::find_or<rtype>(input, "physics", "p_min", -1e20);
    h_p_bounds(1) = toml::find_or<rtype>(input, "physics", "p_max", 1e20);
}

void Physics::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Physics: " << PHYSICS_NAMES.at(type) << std::endl;
    std::cout << "> p_min: " << h_p_bounds(0) << std::endl;
    std::cout << "> p_max: " << h_p_bounds(1) << std::endl;
}

void Physics::copy_host_to_device() {
    Kokkos::deep_copy(p_bounds, h_p_bounds);
}

void Physics::copy_device_to_host() {
    Kokkos::deep_copy(h_p_bounds, p_bounds);
}

Euler::Euler() {
    type = PhysicsType::EULER;
}

Euler::~Euler() {
    // Empty
}

void Euler::init(const toml::value & input) {
    Physics::init(input);

    toml::value input_physics = toml::find(input, "physics");

    if (!input_physics.contains("gamma")) {
        throw std::runtime_error("Missing gamma for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("p_ref")) {
        throw std::runtime_error("Missing p_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("T_ref")) {
        throw std::runtime_error("Missing T_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }
    if (!input_physics.contains("rho_ref")) {
        throw std::runtime_error("Missing rho_ref for physics: " + PHYSICS_NAMES.at(type) + ".");
    }

    gamma = toml::find<rtype>(input, "physics", "gamma");
    p_ref = toml::find<rtype>(input, "physics", "p_ref");
    T_ref = toml::find<rtype>(input, "physics", "T_ref");
    rho_ref = toml::find<rtype>(input, "physics", "rho_ref");

    set_R_cp_cv();

    print();
}


// FOR TESTING PURPOSES ONLY
void Euler::init(rtype p_min, rtype p_max, rtype gamma,
                 rtype p_ref, rtype T_ref, rtype rho_ref) {
    h_p_bounds(0) = p_min;
    h_p_bounds(1) = p_max;
    Kokkos::deep_copy(p_bounds, h_p_bounds);
    this->gamma = gamma;
    this->p_ref = p_ref;
    this->T_ref = T_ref;
    this->rho_ref = rho_ref;

    set_R_cp_cv();
}

void Euler::set_R_cp_cv() {
    R = p_ref / (T_ref * rho_ref);
    cp = R * gamma / (gamma - 1.0);
    cv = cp / gamma;
}

void Euler::print() const {
    Physics::print();
    std::cout << "> gamma: " << gamma << std::endl;
    std::cout << "> p_ref: " << p_ref << std::endl;
    std::cout << "> T_ref: " << T_ref << std::endl;
    std::cout << "> rho_ref: " << rho_ref << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}