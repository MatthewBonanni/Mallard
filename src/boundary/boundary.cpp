/**
 * @file boundary.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class implementation.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "boundary.h"

#include <iostream>
#include <utility>

#include "common.h"
#include "flux_functor.h"

Boundary::Boundary() {
    // Empty
}

Boundary::~Boundary() {
    std::cout << "Destroying boundary: " << zone->get_name() << std::endl;
}

void Boundary::set_zone(FaceZone * zone) {
    this->zone = zone;
}

void Boundary::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

void Boundary::set_face_quad_weights(Kokkos::View<rtype *> face_quad_weights) {
    this->face_quad_weights = face_quad_weights;
}

void Boundary::set_physics(std::shared_ptr<PhysicsWrapper> physics) {
    this->physics = physics;
}

void Boundary::set_riemann_solver(std::shared_ptr<RiemannSolver> riemann_solver) {
    this->riemann_solver = riemann_solver;
}

void Boundary::print() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Boundary: " << zone->get_name() << std::endl;
    std::cout << "Type: " << BOUNDARY_NAMES.at(type) << std::endl;
}

void Boundary::init(const toml::value & input) {
    (void)(input);
    throw std::runtime_error("Boundary::init() not implemented.");
}

void Boundary::copy_host_to_device() {
    // Empty
}

void Boundary::copy_device_to_host() {
    // Empty
}