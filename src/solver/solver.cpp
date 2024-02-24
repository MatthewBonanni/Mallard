/**
 * @file solver.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Solver class implementation.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "solver.h"

#include <iostream>
#include <functional>
#include <memory>

#include <Kokkos_Core.hpp>

#include "common.h"
#include "mesh.h"
#include "boundary.h"
#include "boundary_symmetry.h"
#include "boundary_extrapolation.h"
#include "boundary_wall_adiabatic.h"
#include "boundary_upt.h"
#include "boundary_p_out.h"

Solver::Solver() {
    // Empty
}

Solver::~Solver() {
    std::cout << "Destroying solver..." << std::endl;
    deallocate_memory();
}

int Solver::init(const std::string& input_file_name) {
    print_logo();
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Initializing solver..." << std::endl;
#ifdef Mallard_USE_DOUBLES
    std::cout << "Mallard has been compiled with DOUBLE precision." << std::endl;
#else   
    std::cout << "Mallard has been compiled with SINGLE precision." << std::endl;
#endif
    std::cout << "Parsing input file: " << input_file_name << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;

    input = toml::parse(input_file_name);
    std::cout << input << std::endl;

    std::cout << LOG_SEPARATOR << std::endl;

    /** \todo Implement restarts. */
    t = 0.0;
    step = 0;
    t_wall_0 = timer.seconds();

    init_mesh();
    init_physics();
    init_numerics();
    init_boundaries();
    init_run_parameters();

    allocate_memory();
    register_data();
    
    init_output();
    init_solution();

    return 0;
}

void Solver::init_mesh() {
    std::cout << "Initializing mesh..." << std::endl;

    std::string type_str = toml::find_or<std::string>(input, "mesh", "type", "file");
    MeshType type;
    typename std::unordered_map<std::string, MeshType>::const_iterator it = MESH_TYPES.find(type_str);
    if (it == MESH_TYPES.end()) {
        throw std::runtime_error("Unknown mesh type: " + type_str + ".");
    } else {
        type = it->second;
    }

    mesh = std::make_shared<Mesh>();
    mesh->set_type(type);

    if (type == MeshType::FILE) {
        std::string filename = toml::find_or<std::string>(input, "mesh", "filename", "mesh.msh");
        throw std::runtime_error("MeshType::FILE not implemented.");
    } else if (type == MeshType::CARTESIAN) {
        u_int32_t Nx = toml::find_or<u_int32_t>(input, "mesh", "Nx", 100);
        u_int32_t Ny = toml::find_or<u_int32_t>(input, "mesh", "Ny", 100);
        rtype Lx = toml::find_or<rtype>(input, "mesh", "Lx", 1.0);
        rtype Ly = toml::find_or<rtype>(input, "mesh", "Ly", 1.0);
        mesh->init_cart(Nx, Ny, Lx, Ly);
    } else if (type == MeshType::WEDGE) {
        u_int32_t Nx = toml::find_or<u_int32_t>(input, "mesh", "Nx", 100);
        u_int32_t Ny = toml::find_or<u_int32_t>(input, "mesh", "Ny", 100);
        rtype Lx = toml::find_or<rtype>(input, "mesh", "Lx", 1.0);
        rtype Ly = toml::find_or<rtype>(input, "mesh", "Ly", 1.0);
        mesh->init_wedge(Nx, Ny, Lx, Ly);
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown mesh type.");
    }
}

void Solver::init_physics() {
    std::cout << "Initializing physics..." << std::endl;

    std::string physics_str = toml::find_or<std::string>(input, "physics", "type", "euler");

    PhysicsType type;
    typename std::unordered_map<std::string, PhysicsType>::const_iterator it = PHYSICS_TYPES.find(physics_str);
    if (it == PHYSICS_TYPES.end()) {
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    } else {
        type = it->second;
    }

    if (type == PhysicsType::EULER) {
        physics = std::make_shared<Euler>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown physics type: " + physics_str + ".");
    }

    physics->init(input);
}

void Solver::init_numerics() {
    std::cout << "Initializing numerics..." << std::endl;

    std::string face_reconstruction_str = toml::find_or<std::string>(input, "numerics", "face_reconstruction", "FO");
    std::string riemann_solver_str = toml::find_or<std::string>(input, "numerics", "riemann_solver", "HLLC");
    std::string time_integrator_str = toml::find_or<std::string>(input, "numerics", "time_integrator", "LSSSPRK3");

    FaceReconstructionType face_reconstruction_type;
    typename std::unordered_map<std::string, FaceReconstructionType>::const_iterator it_face = FACE_RECONSTRUCTION_TYPES.find(face_reconstruction_str);
    if (it_face == FACE_RECONSTRUCTION_TYPES.end()) {
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    } else {
        face_reconstruction_type = it_face->second;
    }

    RiemannSolverType riemann_solver_type;
    typename std::unordered_map<std::string, RiemannSolverType>::const_iterator it_riemann = RIEMANN_SOLVER_TYPES.find(riemann_solver_str);
    if (it_riemann == RIEMANN_SOLVER_TYPES.end()) {
        throw std::runtime_error("Unknown Riemann solver type: " + riemann_solver_str + ".");
    } else {
        riemann_solver_type = it_riemann->second;
    }

    TimeIntegratorType time_integrator_type;
    typename std::unordered_map<std::string, TimeIntegratorType>::const_iterator it_time = TIME_INTEGRATOR_TYPES.find(time_integrator_str);
    if (it_time == TIME_INTEGRATOR_TYPES.end()) {
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    } else {
        time_integrator_type = it_time->second;
    }

    if (face_reconstruction_type == FaceReconstructionType::FirstOrder) {
        face_reconstruction = std::make_unique<FirstOrder>();
    } else if (face_reconstruction_type == FaceReconstructionType::WENO3_JS) {
        face_reconstruction = std::make_unique<WENO3_JS>();
    } else if (face_reconstruction_type == FaceReconstructionType::WENO5_JS) {
        face_reconstruction = std::make_unique<WENO5_JS>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown face reconstruction type: " + face_reconstruction_str + ".");
    }

    if (riemann_solver_type == RiemannSolverType::Rusanov) {
        riemann_solver = std::make_unique<Rusanov>();
    } else if (riemann_solver_type == RiemannSolverType::Roe) {
        riemann_solver = std::make_unique<Roe>();
    } else if (riemann_solver_type == RiemannSolverType::HLL) {
        riemann_solver = std::make_unique<HLL>();
    } else if (riemann_solver_type == RiemannSolverType::HLLC) {
        riemann_solver = std::make_unique<HLLC>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown Riemann solver type: " + face_reconstruction_str + ".");
    }

    if (time_integrator_type == TimeIntegratorType::FE) {
        time_integrator = std::make_unique<FE>();
    } else if (time_integrator_type == TimeIntegratorType::RK4) {
        time_integrator = std::make_unique<RK4>();
    } else if (time_integrator_type == TimeIntegratorType::SSPRK3) {
        time_integrator = std::make_unique<SSPRK3>();
    } else if (time_integrator_type == TimeIntegratorType::LSRK4) {
        time_integrator = std::make_unique<LSRK4>();
    } else if (time_integrator_type == TimeIntegratorType::LSSSPRK3) {
        time_integrator = std::make_unique<LSSSPRK3>();
    } else {
        // Should never get here due to the enum class.
        throw std::runtime_error("Unknown time integrator type: " + time_integrator_str + ".");
    }

    face_reconstruction->set_mesh(mesh);
    face_reconstruction->set_cell_conservatives(&conservatives);
    face_reconstruction->set_face_conservatives(&face_conservatives);
    face_reconstruction->init();

    riemann_solver->init(input);

    rhs_func = std::bind(&Solver::calc_rhs,
                         this,
                         std::placeholders::_1,
                         std::placeholders::_2,
                         std::placeholders::_3);
    time_integrator->init();

    check_nan = toml::find_or<bool>(input, "numerics", "check_nan", false);
}

void Solver::init_boundaries() {
    std::cout << "Initializing boundaries..." << std::endl;

    std::vector<toml::value> input_boundaries = toml::find<std::vector<toml::value>>(input, "boundaries");
    for (const auto & bound : input_boundaries) {
        std::optional<std::string> name = toml::find<std::string>(bound, "name");
        std::optional<std::string> type = toml::find<std::string>(bound, "type");

        if (!name.has_value()) {
            throw std::runtime_error("Boundary name not specified.");
        }
        if (!type.has_value()) {
            throw std::runtime_error("Boundary type not specified.");
        }

        BoundaryType btype;
        typename std::unordered_map<std::string, BoundaryType>::const_iterator it = BOUNDARY_TYPES.find(*type);
        if (it == BOUNDARY_TYPES.end()) {
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        } else {
            btype = it->second;
        }

        if (mesh->get_face_zone(*name) == nullptr) {
            throw std::runtime_error("Boundary name " + *name + " not found in mesh.");
        }

        if (btype == BoundaryType::SYMMETRY) {
            boundaries.push_back(std::make_unique<BoundarySymmetry>());
        } else if (btype == BoundaryType::EXTRAPOLATION) {
            boundaries.push_back(std::make_unique<BoundaryExtrapolation>());
        } else if (btype == BoundaryType::WALL_ADIABATIC) {
            boundaries.push_back(std::make_unique<BoundaryWallAdiabatic>());
        } else if (btype == BoundaryType::UPT) {
            boundaries.push_back(std::make_unique<BoundaryUPT>());
        } else if (btype == BoundaryType::P_OUT) {
            boundaries.push_back(std::make_unique<BoundaryPOut>());
        } else {
            // Should never get here due to the enum class.
            throw std::runtime_error("Unknown boundary type: " + *type + ".");
        }

        boundaries.back()->set_zone(mesh->get_face_zone(*name));
        boundaries.back()->set_mesh(mesh);
        boundaries.back()->set_physics(physics);
        boundaries.back()->set_riemann_solver(riemann_solver);
        boundaries.back()->init(bound);
    }
}

void Solver::init_run_parameters() {
    std::cout << "Initializing run parameters..." << std::endl;

    std::optional<rtype> dt_in = toml::find<rtype>(input, "run", "dt");
    std::optional<rtype> cfl_in = toml::find<rtype>(input, "run", "cfl");
    std::optional<u_int32_t> n_steps_in = toml::find<u_int32_t>(input, "run", "n_steps");
    std::optional<rtype> t_stop_in = toml::find<rtype>(input, "run", "t_stop");
    std::optional<rtype> t_wall_stop_in = toml::find<rtype>(input, "run", "t_wall_stop");

    if (!dt_in.has_value() && !cfl_in.has_value()) {
        throw std::runtime_error("Either dt or cfl must be specified.");
    } else if (dt_in.has_value() && cfl_in.has_value()) {
        throw std::runtime_error("Only one of dt or cfl can be specified.");
    }

    if (!n_steps_in.has_value() &&
        !t_stop_in.has_value() &&
        !t_wall_stop_in.has_value()) {
        throw std::runtime_error("Either n_steps, t_stop, or t_wall_stop must be specified.");
    }

    if (dt_in.has_value()) {
        std::cout << "Using specified dt." << std::endl;
        use_cfl = false;
        dt = dt_in.value();
    } else {
        std::cout << "Using specified cfl." << std::endl;
        use_cfl = true;
        cfl = cfl_in.value();
    }

    n_steps = n_steps_in.value_or(-1);
    t_stop = t_stop_in.value_or(-1.0);
    t_wall_stop = t_wall_stop_in.value_or(-1.0);
}

void Solver::init_output() {
    std::cout << "Initializing output..." << std::endl;

    check_interval = toml::find_or<u_int32_t>(input, "output", "check_interval", 1);

    init_data_writers();
}

void Solver::init_data_writers() {
    std::cout << "Initializing data writers..." << std::endl;

    if (!input.contains("write_data")) {
        return;
    }

    std::vector<toml::value> outputs = toml::find<std::vector<toml::value>>(input, "write_data");
    for (const auto & output : outputs) {
        data_writers.push_back(std::make_unique<DataWriter>());
        data_writers.back()->init(output, data, mesh);
    }
}

void Solver::allocate_memory() {
    std::cout << "Allocating memory..." << std::endl;

    Kokkos::resize(conservatives, mesh->n_cells(), N_CONSERVATIVE);
    Kokkos::resize(primitives, mesh->n_cells(), N_PRIMITIVE);
    Kokkos::resize(face_conservatives, mesh->n_faces(), 2, N_CONSERVATIVE);
    Kokkos::resize(face_primitives, mesh->n_faces(), 2, N_PRIMITIVE);

    h_conservatives = Kokkos::create_mirror_view(conservatives);
    h_primitives = Kokkos::create_mirror_view(primitives);
    h_face_conservatives = Kokkos::create_mirror_view(face_conservatives);
    h_face_primitives = Kokkos::create_mirror_view(face_primitives);

    solution_pointers.push_back(&conservatives);
    for (u_int8_t i = 0; i < time_integrator->get_n_solution_vectors() - 1; i++) {
        solution_pointers.push_back(new view_2d("solution", mesh->n_cells(), N_CONSERVATIVE));
    }

    for (u_int8_t i = 0; i < time_integrator->get_n_rhs_vectors(); i++) {
        rhs_pointers.push_back(new view_2d("rhs", mesh->n_cells(), N_CONSERVATIVE));
    }

    Kokkos::resize(cfl_local, mesh->n_cells());
}

void Solver::copy_host_to_device() {
    Kokkos::deep_copy(conservatives, h_conservatives);
    Kokkos::deep_copy(primitives, h_primitives);
    Kokkos::deep_copy(face_conservatives, h_face_conservatives);
    Kokkos::deep_copy(face_primitives, h_face_primitives);
}

void Solver::copy_device_to_host() {
    Kokkos::deep_copy(h_conservatives, conservatives);
    Kokkos::deep_copy(h_primitives, primitives);
    Kokkos::deep_copy(h_face_conservatives, face_conservatives);
    Kokkos::deep_copy(h_face_primitives, face_primitives);
}

void Solver::register_data() {
    std::cout << "Registering data..." << std::endl;

    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        auto subview = Kokkos::subview(conservatives, Kokkos::ALL(), i);
        data.push_back(Data(CONSERVATIVE_NAMES[i], subview));
    }

    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        auto subview = Kokkos::subview(primitives, Kokkos::ALL(), i);
        data.push_back(Data(PRIMITIVE_NAMES[i], subview));
    }

    data.push_back(Data("CFL", cfl_local));
}

int Solver::run() {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Running solver..." << std::endl;

    write_data(true);

    while (!done()) {
        calc_dt();
        do_checks();
        take_step();
        check_fields();
        write_data();
    }

    write_data(true);

    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Solver finished." << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;

    return 0;
}

bool Solver::done() const {
    bool done_steps = 0;
    bool done_t = 0;
    bool done_t_wall = 0;

    if (n_steps > 0) {
        done_steps = (step >= n_steps);
        if (done_steps) {
            std::cout << "Stop condition reached: step = " << step << std::endl;
        }
    }

    if (t_stop > 0) {
        done_t = (t >= t_stop);
        if (done_t) {
            std::cout << "Stop condition reached: t = " << t << std::endl;
        }
    }

    rtype t_wall = timer.seconds() - t_wall_0;
    if (t_wall_stop > 0) {
        done_t_wall = (t_wall >= t_wall_stop);
        if (done_t_wall) {
            std::cout << "Stop condition reached: t_wall = " << t_wall << std::endl;
        }
    }

    return done_steps || done_t || done_t_wall;
}

void Solver::print_step_info() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Starting step: " << step
              << " time: " << t
              << " dt: " << dt
              << std::endl;
}

void print_range(const std::string & name,
                 const rtype & min,
                 const rtype & max) {
    std::cout << "> Scalar range: "
              << name << " = ["
              << min << ", "
              << max << "]" << std::endl;
}

void Solver::do_checks() {
    bool check_now = (step % check_interval == 0);
    if (!check_now) {
        return;
    }

    print_step_info();

    State max_cons = max_array<4>(conservatives);
    State min_cons = min_array<4>(conservatives);
    Primitives max_prim = max_array<5>(primitives);
    Primitives min_prim = min_array<5>(primitives);

    for (size_t i = 0; i < CONSERVATIVE_NAMES.size(); i++) {
        print_range(CONSERVATIVE_NAMES[i], min_cons[i], max_cons[i]);
    }

    for (size_t i = 0; i < PRIMITIVE_NAMES.size(); i++) {
        print_range(PRIMITIVE_NAMES[i], min_prim[i], max_prim[i]);
    }

    rtype t_wall = timer.seconds();
    rtype dt_check = t - t_last_check;
    rtype dt_wall_check = t_wall - t_wall_last_check;
    std::cout << "Performance:" << std::endl;
    std::cout << "> Time since last check: "
              << dt_wall_check
              << " s" << std::endl;
    std::cout << "> Time / step / cell: "
              << dt_wall_check / check_interval / mesh->n_cells()
              << " s" << std::endl;
    std::cout << "> Simulation time / wall time: "
              << dt_check / dt_wall_check
              << std::endl;
    t_last_check = t;
    t_wall_last_check = t_wall;

    std::cout << LOG_SEPARATOR << std::endl;
}

void Solver::check_fields() const {
    if (!check_nan) {
        return;
    }

    bool nan_found = false;
    for (u_int32_t i = 0; i < mesh->n_cells(); i++) {
        for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
            if (Kokkos::isnan(conservatives(i, j))) {
                nan_found = true;
            }
        }

        for (u_int16_t j = 0; j < N_PRIMITIVE; j++) {
            if (Kokkos::isnan(primitives(i, j))) {
                nan_found = true;
            }
        }

        if (nan_found) {
            std::stringstream msg;
            msg << "NaN found in solution." << std::endl;
            msg << "t: " << t << std::endl;
            msg << "step: " << step << std::endl;
            msg << "i_cell: " << i << std::endl;
            msg << "> x: " << mesh->cell_coords(i)[0] << std::endl;
            msg << "> y: " << mesh->cell_coords(i)[1] << std::endl;
            msg << "conservatives:" << std::endl;
            for (u_int16_t j = 0; j < N_CONSERVATIVE; j++) {
                msg << "> " << CONSERVATIVE_NAMES[j] << ": " << conservatives(i, j) << std::endl;
            }
            msg << "primitives:" << std::endl;
            for (u_int16_t j = 0; j < N_PRIMITIVE; j++) {
                msg << "> " << PRIMITIVE_NAMES[j] << ": " << primitives(i, j) << std::endl;
            }
            throw std::runtime_error(msg.str());
        }
    }
}

void Solver::write_data(bool force) const {
    for (auto & writer : data_writers) {
        writer->write(step, force);
    }
}

void Solver::deallocate_memory() {
    std::cout << "Deallocating memory..." << std::endl;

    // Make sure all Kokkos threads are done.
    Kokkos::fence();
}

void Solver::print_logo() const {
    std::cout << R"(    __  ___      ____               __)" << std::endl
              << R"(   /  |/  /___ _/ / /___ __________/ /)" << std::endl
              << R"(  / /|_/ / __ `/ / / __ `/ ___/ __  / )" << std::endl
              << R"( / /  / / /_/ / / / /_/ / /  / /_/ /  )" << std::endl
              << R"(/_/  /_/\__,_/_/_/\__,_/_/   \__,_/   )" << std::endl;
}

void Solver::take_step() {
    /** \todo GPU compatibility - need to manage host and device views */
    /** \todo MPI implementation */
    time_integrator->take_step(dt,
                               solution_pointers,
                               &face_conservatives,
                               rhs_pointers,
                               &rhs_func);
    update_primitives();
    Kokkos::fence();
    step++;
    t += dt;
}

void Solver::update_primitives() {
    Kokkos::parallel_for(mesh->n_cells(), KOKKOS_LAMBDA(const u_int32_t i_cell) {
        State cell_conservatives;
        Primitives cell_primitives;
        for (u_int16_t i = 0; i < N_CONSERVATIVE; i++) {
            cell_conservatives[i] = conservatives(i_cell, i);
        }
        physics->compute_primitives_from_conservatives(cell_primitives.data(),
                                                       cell_conservatives.data());
        for (u_int16_t i = 0; i < N_PRIMITIVE; i++) {
            primitives(i_cell, i) = cell_primitives[i];
        }
    });
}

void Solver::calc_dt() {
    if (use_cfl) {
        rtype max_spectral_radius = calc_spectral_radius();
        dt = cfl / max_spectral_radius;
        cA_to_A(mesh->n_cells(), dt, cfl_local.data());
    }

    if (dt < 0.0) {
        throw std::runtime_error("dt negative: " + std::to_string(dt) + ".");
    }
}

rtype Solver::calc_spectral_radius() {
    rtype max_spectral_radius = -1.0;
    Kokkos::parallel_reduce(mesh->n_cells(), 
                            KOKKOS_LAMBDA(const u_int32_t i_cell, rtype & max_spectral_radius_i) {
        rtype spectral_radius_convective;
        rtype spectral_radius_acoustic;
        // rtype spectral_radius_viscous;
        // rtype spectral_radius_heat;
        rtype spectral_radius_overall;
        rtype rho_l, rho_r, p_l, p_r, sos_l, sos_r, sos_f;
        NVector s, u_l, u_r, u_f;
        rtype dx_n, u_n;
        rtype geom_factor;
        NVector n_unit;

        spectral_radius_convective = 0.0;
        spectral_radius_acoustic = 0.0;
        // spectral_radius_viscous = 0.0;
        // spectral_radius_heat = 0.0;

        for (u_int32_t i_face = 0; i_face < mesh->faces_of_cell(i_cell).size(); i_face++) {
            int32_t i_cell_l = mesh->cells_of_face(i_face)[0];
            int32_t i_cell_r = mesh->cells_of_face(i_face)[1];

            n_unit = unit(mesh->face_normal(i_face));

            if (i_cell_r == -1) {
                // Boundary face, hack
                s[0] = 2.0 * (mesh->face_coords(i_face)[0] - mesh->cell_coords(i_cell_l)[0]);
                s[1] = 2.0 * (mesh->face_coords(i_face)[1] - mesh->cell_coords(i_cell_l)[1]);
                i_cell_r = i_cell_l;
            } else {
                s[0] = mesh->cell_coords(i_cell_r)[0] - mesh->cell_coords(i_cell_l)[0];
                s[1] = mesh->cell_coords(i_cell_r)[1] - mesh->cell_coords(i_cell_l)[1];
            }

            dx_n = fabs(dot<N_DIM>(s.data(), n_unit.data()));

            rho_l = conservatives(i_cell_l, 0);
            rho_r = conservatives(i_cell_r, 1);
            u_l[0] = primitives(i_cell_l, 0);
            u_l[1] = primitives(i_cell_l, 1);
            u_r[0] = primitives(i_cell_r, 0);
            u_r[1] = primitives(i_cell_r, 1);
            p_l = primitives(i_cell_l, 2);
            p_r = primitives(i_cell_r, 2);
            sos_l = physics->get_sound_speed_from_pressure_density(p_l, rho_l);
            sos_r = physics->get_sound_speed_from_pressure_density(p_r, rho_r);
            sos_f = 0.5 * (sos_l + sos_r);

            u_f[0] = 0.5 * (u_l[0] + u_r[0]);
            u_f[1] = 0.5 * (u_l[1] + u_r[1]);
            u_n = fabs(dot<N_DIM>(u_f.data(), n_unit.data()));

            spectral_radius_convective += u_n / dx_n;
            spectral_radius_acoustic += pow(sos_f / dx_n, 2.0);
        }

        geom_factor = 3.0 / mesh->faces_of_cell(i_cell).size();
        spectral_radius_convective *= 1.37 * geom_factor;
        spectral_radius_acoustic = 1.37 * sqrt(geom_factor * spectral_radius_acoustic);

        /** \todo Implement viscous and heat spectral radii */
        spectral_radius_overall = spectral_radius_convective + spectral_radius_acoustic;

        // Update max spectral radius
        max_spectral_radius_i = std::max(max_spectral_radius_i,
                                         spectral_radius_overall);

        // Store spectral radius in cfl_local, will be used to compute local cfl
        cfl_local(i_cell) = spectral_radius_overall;
    }, Kokkos::Max<rtype>(max_spectral_radius));

    return max_spectral_radius;
}