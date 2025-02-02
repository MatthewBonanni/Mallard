/**
 * @file face_reconstruction.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class implementation.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#include "face_reconstruction.h"

#include <iostream>

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include "common.h"
#include "basis.h"
#include "quadrature.h"

FaceReconstruction::FaceReconstruction() {
    // Empty
}

FaceReconstruction::~FaceReconstruction() {
    std::cout << "Destroying face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
}

void FaceReconstruction::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void FaceReconstruction::set_cell_conservatives(Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives) {
    this->cell_conservatives = cell_conservatives;
}

void FaceReconstruction::set_face_conservatives(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives) {
    this->face_conservatives = face_conservatives;
}

void FaceReconstruction::set_mesh(std::shared_ptr<Mesh> mesh) {
    this->mesh = mesh;
}

FirstOrder::FirstOrder() {
    type = FaceReconstructionType::FIRST_ORDER;
}

FirstOrder::~FirstOrder() {
    // Empty
}

void FirstOrder::init(const toml::value & input) {
    (void)(input);
    print();
}

struct FirstOrderFunctor {
    public:
        /**
         * @brief Construct a new FirstOrderFunctor object
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        FirstOrderFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                          Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                              cells_of_face(cells_of_face),
                              face_solution(face_solution),
                              solution(solution) {}

        /**
         * @brief Overloaded operator for first order face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            int32_t i_cell_l = cells_of_face(i_face, 0);
            int32_t i_cell_r = cells_of_face(i_face, 1);

            FOR_I_CONSERVATIVE {
                face_solution(i_face, 0, i) = solution(i_cell_l, i);
                face_solution(i_face, 1, i) = solution(i_cell_r, i);
            }
        }

    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void FirstOrder::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                  Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    FirstOrderFunctor recon_functor(mesh->cells_of_face, face_solution, solution);
    Kokkos::parallel_for(mesh->n_faces, recon_functor);
}

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::init(const toml::value & input) {
    std::string basis_type_str = toml::find_or<std::string>(input, "basis_type", "monomial");
    poly_order = toml::find<u_int8_t>(input, "basis_order");
    max_stencil_size_factor = toml::find_or<rtype>(input, "max_stencil_size_factor", 2.0);
    std::string quadrature_type_str = toml::find_or<std::string>(input, "quadrature_type", "triangle_dunavant");
    u_int8_t quadrature_order = toml::find_or<u_int8_t>(input, "quadrature_order", 2 * poly_order);

    if (BASIS_TYPES.find(basis_type_str) == BASIS_TYPES.end()) {
        throw std::runtime_error("Unknown basis type: " + basis_type_str + ".");
    } else {
        basis_type = BASIS_TYPES.at(basis_type_str);
    }

    QuadratureType quadrature_type;
    if (QUADRATURE_TYPES.find(quadrature_type_str) == QUADRATURE_TYPES.end()) {
        throw std::runtime_error("Unknown quadrature type: " + quadrature_type_str + ".");
    } else {
        quadrature_type = QUADRATURE_TYPES.at(quadrature_type_str);
    }

    switch (quadrature_type) {
        case QuadratureType::TRIANGLE_CENTROID:
            quadrature = TriangleCentroid();
            break;
        case QuadratureType::TRIANGLE_DUNAVANT:
            quadrature = TriangleDunavant(quadrature_order);
            break;
        default:
            throw std::runtime_error("Unknown quadrature type.");
    }

    calc_max_stencil_size();
    calc_polynomial_indices();
    compute_stencils();
    compute_reconstruction_matrices();
    compute_oscillation_indicator();
    print();
}

void TENO::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << "> Polynomial order: " << (u_int16_t)poly_order << std::endl;
    std::cout << "> Quadrature order: " << (u_int16_t)quadrature.order << std::endl;
    std::cout << "> Maximum stencil size factor: " << max_stencil_size_factor << std::endl;
    std::cout << "> Maximum cells per stencil: " << max_cells_per_stencil << std::endl;
    std::cout << "> Number of degrees of freedom: " << n_dof << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

void TENO::calc_max_stencil_size() {
    // Nd = (prod(r + m) from m=1 to N_DIM) / N_DIM!
    n_dof = 1;
    u_int16_t denom = 1;
    for (u_int8_t i = 1; i <= N_DIM; ++i) {
        denom *= i;
        n_dof *= poly_order + i;
    }
    n_dof /= denom;
    max_cells_per_stencil = max_stencil_size_factor * n_dof;
}

void TENO::calc_polynomial_indices() {
    poly_indices = Kokkos::View<u_int32_t *[N_DIM]>("poly_indices", n_dof);
    h_poly_indices = Kokkos::create_mirror_view(poly_indices);

    // Use a lexicographic ordering to generate the polynomial indices
    int idx = 0;
    std::vector<int> indices(N_DIM);
    for (int i_poly = 0; i_poly <= poly_order; i_poly++) {
        // First set of indices has i_dim set to i_poly
        std::fill(indices.begin(), indices.end(), 0);
        indices[0] = i_poly;

        // Write to the view
        for (int i_dim = 0; i_dim < N_DIM; ++i_dim) {
            h_poly_indices(idx, i_dim) = indices[i_dim];
        }
        ++idx;

        // Propagate 1 from each dimension's index to the right,
        // repeating until the last dimension's index is equal to i_poly
        while (indices[N_DIM-1] < i_poly) {
            for (int i_dim = 0; i_dim < N_DIM-1; ++i_dim) {
                if (indices[i_dim] > 0) {
                    indices[i_dim] -= 1;
                    indices[i_dim + 1] += 1;
                }
            }

            // Write to the view
            for (int i_dim = 0; i_dim < N_DIM; ++i_dim) {
                h_poly_indices(idx, i_dim) = indices[i_dim];
            }
            ++idx;
        }
    }

    Kokkos::deep_copy(poly_indices, h_poly_indices);
}

void TENO::get_next_ring(std::vector<std::vector<u_int32_t>> & neighbor_rings,
                         u_int32_t i_target_cell) {
    // Get the next ring of neighbors
    std::vector<u_int32_t> next_ring;
    for (auto i_neighbor_cell : neighbor_rings.back()) {
        std::vector<u_int32_t> neighbors;
        mesh->h_neighbors_of_cell(i_neighbor_cell, 1, neighbors);
        for (auto neighbor : neighbors) {
            // Check if this neighbor is already in a previous ring or
            // the ring we are currently building
            bool in_old_ring = false;
            for (auto ring : neighbor_rings) {
                if (std::find(ring.begin(), ring.end(), neighbor) != ring.end()) {
                    in_old_ring = true;
                    break;
                }
            }
            bool in_next_ring = (std::find(next_ring.begin(),
                                           next_ring.end(),
                                           neighbor) != next_ring.end());
            if (!in_old_ring && !in_next_ring) {
                next_ring.push_back(neighbor);
            }
        }
    }

    // Sort the ring by distance to the target cell
    std::vector<std::pair<u_int32_t, rtype>> distances;
    for (auto i_neighbor_cell : next_ring) {
        rtype distance = 0.0;
        FOR_I_DIM {
            distance += std::pow(mesh->cell_coords(i_neighbor_cell, i) -
                                 mesh->cell_coords(i_target_cell,   i), 2);
        }
        distances.push_back(std::make_pair(i_neighbor_cell, distance));
    }
    std::sort(distances.begin(),
              distances.end(),
              [](auto & left, auto & right) {
                  return left.second < right.second;
              });
    next_ring.clear();
    for (auto pair : distances) {
        next_ring.push_back(pair.first);
    }

    // Add the ring to the neighbor_rings vector
    neighbor_rings.push_back(next_ring);
}

std::vector<u_int32_t> TENO::compute_stencil_of_cell_centered(u_int32_t i_cell) {
    // Naive Cell Based (NCB) algorithm (Tsoutsanis 2023)
    std::vector<u_int32_t> stencil;
    std::vector<std::vector<u_int32_t>> neighbor_rings;
    stencil.push_back(i_cell);
    neighbor_rings.push_back(stencil);
    while (stencil.size() < max_cells_per_stencil) {
        // Get the next ring of neighbors
        get_next_ring(neighbor_rings, i_cell);

        // Add the neighbors to the stencil as needed
        for (auto i_neighbor_cell : neighbor_rings.back()) {
            if (stencil.size() == max_cells_per_stencil) {
                break;
            }
            stencil.push_back(i_neighbor_cell);
        }
    }
    return stencil;
}

std::vector<std::vector<u_int32_t>> TENO::compute_stencils_of_cell_directional(u_int32_t i_cell) {
    // Type 4 algorithm (Tsoutsanis 2023)
    u_int8_t n_stencils = mesh->n_faces_of_cell(i_cell);
    std::vector<std::vector<u_int32_t>> stencils(n_stencils);
    std::vector<bool> stencil_grew(n_stencils);
    bool all_done = false;

    // First, compute the transformation matrices for each subvolume of the target cell
    std::vector<std::vector<rtype>> J_sub_matrices;
    std::vector<std::vector<rtype>> J_sub_inverses;
    for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
        u_int32_t i_face = mesh->h_face_of_cell(i_cell, i_stencil); // Take i_face_local = i_stencil
        u_int32_t i_node_0 = mesh->h_node_of_face(i_face, 0);
        u_int32_t i_node_1 = mesh->h_node_of_face(i_face, 1);

        std::vector<rtype> v0(N_DIM);
        std::vector<rtype> v1(N_DIM);
        std::vector<rtype> v2(N_DIM);

        FOR_I_DIM {
            v0[i] = mesh->h_cell_coords(i_cell,   i);
            v1[i] = mesh->h_node_coords(i_node_0, i);
            v2[i] = mesh->h_node_coords(i_node_1, i);
        }

        std::vector<rtype> J(4);
        std::vector<rtype> J_inv(4);
        triangle_J(v0.data(), v1.data(), v2.data(), J.data());
        invert_matrix<2>(J.data(), J_inv.data());
        J_sub_matrices.push_back(J);
        J_sub_inverses.push_back(J_inv);
    }

    // Add the target cell to the stencils
    for (auto & stencil : stencils) {
        stencil.push_back(i_cell);
    }

    std::vector<std::vector<u_int32_t>> neighbor_rings;
    neighbor_rings.push_back({i_cell});
    while (!all_done) {
        // Get the next ring of neighbors
        get_next_ring(neighbor_rings, i_cell);

        // Add the neighbors to the stencils as needed
        std::fill(stencil_grew.begin(), stencil_grew.end(), false);
        for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
            for (auto i_neighbor_cell : neighbor_rings.back()) {
                // Check if the stencil is full
                if (stencils[i_stencil].size() == max_cells_per_stencil) {
                    break;
                }

                // Get the transformed coordinates of the neighbor cell in the
                // subvolume's local system
                NVector dx;
                FOR_I_DIM dx[i] = mesh->h_cell_coords(i_neighbor_cell, i) -
                                  mesh->h_cell_coords(i_cell,          i);
                NVector dx_transformed;
                gemv<N_DIM>(J_sub_inverses[i_stencil].data(), dx.data(), dx_transformed.data());

                // Check if the neighbor cell is in the stencil region, and if it is, add it
                bool in_region = true;
                FOR_I_DIM if (dx_transformed[i] < 0.0) in_region = false;
                if (in_region) {
                    stencils[i_stencil].push_back(i_neighbor_cell);
                    stencil_grew[i_stencil] = true;
                }
            }
        }

        // Check if all stencils are done
        all_done = true;
        for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
            // If this face is a boundary face, it has no directional stencil,
            // and being empty is allowable so we can be done even if it is empty
            // Otherwise, if the stencil is not full, as long as the stencil grew in the last
            // iteration, there may be more neighbors to add so we are not done
            if ((mesh->cells_of_face(mesh->h_face_of_cell(i_cell, i_stencil), 1) != -1) &&
                (stencils[i_stencil].size() < max_cells_per_stencil) &&
                stencil_grew[i_stencil]) {
                all_done = false;
                break;
            }
        }
    }

    // If a stencil is not full, it must be near a boundary. In this case, we loosen
    // the requirement of directionality and build the stencil outwards until it is full
    for (u_int8_t i_stencil = 0; i_stencil < n_stencils; ++i_stencil) {
        // If this is a boundary face, it simply has no stencil. Empty it
        // to be sure and continue to the next stencil
        if (mesh->cells_of_face(mesh->h_face_of_cell(i_cell, i_stencil), 1) == -1) {
            stencils[i_stencil].clear();
            continue;
        }

        // Otherwise, it is near a boundary, but not a boundary face itself.
        // We will build the stencil outwards until it is full
        neighbor_rings.clear();
        neighbor_rings.push_back(stencils[i_stencil]);
        while (stencils[i_stencil].size() < max_cells_per_stencil) {
            // Get the next ring of neighbors
            get_next_ring(neighbor_rings, i_cell);

            // Add the neighbors to the stencil as needed
            for (auto i_neighbor_cell : neighbor_rings.back()) {
                if (stencils[i_stencil].size() == max_cells_per_stencil) {
                    break;
                }
                stencils[i_stencil].push_back(i_neighbor_cell);
            }
        }
    }

    return stencils;
}

void TENO::compute_stencils_of_cell(u_int32_t i_cell,
                                    std::vector<u_int32_t> & v_offsets_stencil_groups,
                                    std::vector<u_int32_t> & v_offsets_stencils,
                                    std::vector<u_int32_t> & v_stencils) {
    std::vector<std::vector<u_int32_t>> stencils;

    // Compute the centered stencil
    std::vector<u_int32_t> stencil_centered = compute_stencil_of_cell_centered(i_cell);
    stencils.push_back(stencil_centered);

    // Compute the directional stencils
    std::vector<std::vector<u_int32_t>> stencils_directional = compute_stencils_of_cell_directional(i_cell);
    for (auto stencil : stencils_directional) {
        stencils.push_back(stencil);
    }

    // Update the global arrays
    for (auto stencil : stencils) {
        if ((stencil.size() < max_cells_per_stencil) && (stencil.size() > 0)) {
            throw std::runtime_error("Stencil is not full.");
        }
        for (auto i_cell : stencil) {
            v_stencils.push_back(i_cell);
        }
        v_offsets_stencils.push_back(v_stencils.size());
    }
    v_offsets_stencil_groups.push_back(v_offsets_stencils.size()-1);
}

void TENO::compute_stencils() {
    std::vector<u_int32_t> v_offsets_stencil_groups;
    std::vector<u_int32_t> v_offsets_stencils;
    std::vector<u_int32_t> v_stencils;
    
    v_offsets_stencils.push_back(0);
    v_offsets_stencil_groups.push_back(0);
    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        compute_stencils_of_cell(i_cell,
                                 v_offsets_stencil_groups,
                                 v_offsets_stencils,
                                 v_stencils);
    }

    // Allocate device arrays
    offsets_stencil_groups = Kokkos::View<u_int32_t *>("offsets_stencil_groups", v_offsets_stencil_groups.size());
    offsets_stencils = Kokkos::View<u_int32_t *>("offsets_stencils", v_stencils.size());
    stencils = Kokkos::View<u_int32_t *>("stencils", v_stencils.size());

    // Set up host mirrors
    h_offsets_stencil_groups = Kokkos::create_mirror_view(offsets_stencil_groups);
    h_offsets_stencils = Kokkos::create_mirror_view(offsets_stencils);
    h_stencils = Kokkos::create_mirror_view(stencils);

    // Fill host mirrors
    for (u_int32_t i = 0; i < v_offsets_stencil_groups.size(); ++i) {
        h_offsets_stencil_groups(i) = v_offsets_stencil_groups[i];
    }
    for (u_int32_t i = 0; i < v_offsets_stencils.size(); ++i) {
        h_offsets_stencils(i) = v_offsets_stencils[i];
    }
    for (u_int32_t i = 0; i < v_stencils.size(); ++i) {
        h_stencils(i) = v_stencils[i];
    }

    // Copy from host to device
    Kokkos::deep_copy(offsets_stencil_groups, h_offsets_stencil_groups);
    Kokkos::deep_copy(offsets_stencils, h_offsets_stencils);
    Kokkos::deep_copy(stencils, h_stencils);
}

void TENO::compute_reconstruction_matrices() {
    std::vector<u_int32_t> v_offsets_reconstruction_matrices;
    std::vector<rtype> v_reconstruction_matrices;
    std::vector<rtype> v_transformed_areas;

    v_offsets_reconstruction_matrices.push_back(0);
    for (u_int32_t i_cell = 0; i_cell < mesh->n_cells; ++i_cell) {
        if (mesh->n_nodes_of_cell(i_cell) != 3) {
            throw std::runtime_error("TENO has only been implemented for triangular cells.");
        }

        // Compute the transformation matrix for the target cell
        std::vector<rtype> J(N_DIM * N_DIM);
        std::vector<rtype> J_inv(N_DIM * N_DIM);
        std::vector<rtype> w0(N_DIM);
        std::vector<rtype> w1(N_DIM);
        std::vector<rtype> w2(N_DIM);
        FOR_I_DIM w0[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 0), i);
        FOR_I_DIM w1[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 1), i);
        FOR_I_DIM w2[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_cell, 2), i);
        triangle_J(w0.data(), w1.data(), w2.data(), J.data());
        invert_matrix<N_DIM>(J.data(), J_inv.data());

        u_int32_t i_first_stencil = h_offsets_stencil_groups(i_cell);
        u_int8_t group_size = h_offsets_stencil_groups(i_cell + 1) - i_first_stencil;
        for (u_int8_t i_stencil_loc = 0; i_stencil_loc < group_size; ++i_stencil_loc) {
            u_int32_t i_stencil = i_first_stencil + i_stencil_loc;
            u_int32_t stencil_offset = h_offsets_stencils(i_stencil);
            u_int32_t stencil_size = h_offsets_stencils(i_stencil + 1) - stencil_offset;
            
            // Handle the case where the stencil is empty
            if (stencil_size == 0) {
                v_offsets_reconstruction_matrices.push_back(v_reconstruction_matrices.size());
                continue;
            }
            
            // Compute the reconstruction matrix for the target cell
            std::vector<rtype> area_trans(stencil_size);
            std::vector<rtype> A(stencil_size * n_dof);
            std::vector<rtype> R(stencil_size * n_dof);
            std::vector<rtype> Y(n_dof * stencil_size);

            // Integrate the basis functions over each cell in the stencil
            for (u_int8_t i_neighbor_loc = 0; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                // Get the neighbor cell's vertex coordinates
                u_int32_t i_neighbor = h_stencils(stencil_offset + i_neighbor_loc);
                std::vector<rtype> v0(N_DIM);
                std::vector<rtype> v1(N_DIM);
                std::vector<rtype> v2(N_DIM);

                FOR_I_DIM v0[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor, 0), i);
                FOR_I_DIM v1[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor, 1), i);
                FOR_I_DIM v2[i] = mesh->h_node_coords(mesh->h_node_of_cell(i_neighbor, 2), i);

                // Compute the neighbor cell's transformation matrix, to be used for mapping
                // the quadrature points to physical space
                std::vector<rtype> J_neighbor(N_DIM * N_DIM);
                triangle_J(v0.data(), v1.data(), v2.data(), J_neighbor.data());

                // Transform the neighbor vertex coordinates to the target cell's local coordinates
                std::vector<rtype> v0_trans(N_DIM);
                std::vector<rtype> v1_trans(N_DIM);
                std::vector<rtype> v2_trans(N_DIM);

                FOR_I_DIM v0_trans[i] = v0[i] - w0[i];
                FOR_I_DIM v1_trans[i] = v1[i] - w0[i];
                FOR_I_DIM v2_trans[i] = v2[i] - w0[i];

                gemv<N_DIM>(J_inv.data(), v0_trans.data(), v0_trans.data());
                gemv<N_DIM>(J_inv.data(), v1_trans.data(), v1_trans.data());
                gemv<N_DIM>(J_inv.data(), v2_trans.data(), v2_trans.data());

                area_trans[i_neighbor_loc] = triangle_area<2>(v0_trans.data(), v1_trans.data(), v2_trans.data());

                // Get all the properly transformed quadrature points
                std::vector<rtype> quad_points(quadrature.h_points.extent(0) * N_DIM);
                for (u_int16_t i_quad = 0; i_quad < quadrature.h_points.extent(0); ++i_quad) {
                    // Get the quadrature point
                    std::vector<rtype> x(N_DIM);
                    FOR_I_DIM x[i] = quadrature.h_points(i_quad, i);
                    
                    // Transform the quadrature point into global coordinates
                    // using the neighbor cell's transformation matrix
                    gemv<N_DIM>(J_neighbor.data(), x.data(), x.data());
                    FOR_I_DIM x[i] += v0[i];

                    // Transform the quadrature point to the target cell's local coordinates
                    // using the target cell's transformation matrix
                    FOR_I_DIM x[i] -= w0[i];
                    gemv<N_DIM>(J_inv.data(), x.data(), x.data());

                    FOR_I_DIM quad_points[i_quad * N_DIM + i] = x[i];
                }

                for (u_int16_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                    size_t ind = i_neighbor_loc * n_dof + i_dof;
                    A[ind] = 0.0;
                    for (u_int16_t i_quad = 0; i_quad < quadrature.h_points.extent(0); ++i_quad) {
                        // Evaluate the basis function at the transformed point
                        rtype basis_value = basis_compute_2D(poly_indices(i_dof, 0),
                                                             poly_indices(i_dof, 1),
                                                             quad_points[i_quad * N_DIM + 0],
                                                             quad_points[i_quad * N_DIM + 1]);

                        // Add the weighted basis value to the integral
                        A[ind] += quadrature.h_weights(i_quad) * basis_value;
                    }
                    A[ind] *= area_trans[i_neighbor_loc];
                }
            }

            // Subtract the integral of the basis function over the target cell
            // Start at the last neighbor and work backwards so that the target cell
            // entries are not overwritten before they are used
            // i_neighbor is signed so that it can go negative and exit the loop
            for (int8_t i_neighbor_loc = stencil_size - 1; i_neighbor_loc >= 0; --i_neighbor_loc) {
                for (u_int16_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                    size_t ind_target   =                          i_dof; // i_target = 0
                    size_t ind_neighbor = i_neighbor_loc * n_dof + i_dof;
                    A[ind_neighbor] -= (area_trans[i_neighbor_loc] / area_trans[0]) * A[ind_target];
                }
            }

            // Compute the Moore-Penrose pseudoinverse of the reconstruction matrix
            // NOTES:
            // - In all cases, the first row of A is zero by definition.
            // - In the case where all cells in the stencil have identical areas in the
            //   target cell's local coordinates, the first column of A will also be 0, so we
            //   need to compute the pseudoinverse of the submatrix of A that excludes the first
            //   row and column.

            // Check for the special case where the first column of A is zero
            bool first_column_zero = true;
            for (u_int16_t i_neighbor_loc = 0; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                if (A[i_neighbor_loc * n_dof] > 1.0e-12) {
                    first_column_zero = false;
                    break;
                }
            }

            if (first_column_zero) {
                // Allocate a new B matrix to hold the submatrix of A, excluding the first row and column
                // NOTES:
                // - The existing R and Y matrices are fine, we just won't use their full memory allocation
                // - In the future we can just reuse A as well by shuffling elements,
                //   but we'll have to be careful not to overwrite in the process
                std::vector<rtype> B((stencil_size - 1) * (n_dof - 1));

                // Fill the B matrix with the submatrix of A that excludes the first row and column
                for (u_int8_t i_neighbor_loc = 1; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                    for (u_int16_t i_dof = 1; i_dof < n_dof; ++i_dof) {
                        size_t ind_A = i_neighbor_loc * n_dof + i_dof;
                        size_t ind_B = (i_neighbor_loc - 1) * (n_dof - 1) + (i_dof - 1);
                        B[ind_B] = A[ind_A];
                    }
                }

                // Compute the QR decomposition of B
                QR_householder_noQ(B.data(), R.data(), stencil_size - 1, n_dof - 1);

                // Now we need to solve the system R^T R B^+ = B^T
                // First, we use forward substitution to solve R^T Y = B^T
                // We ignore everything beyond the square part of R since it's zero
                forward_substitution(R.data(), B.data(), Y.data(),
                                     n_dof - 1, n_dof - 1, stencil_size - 1,
                                     true, true);
                
                // Next, we use back substitution to solve R B^+ = Y
                // Store B^+ in B since we don't need B anymore and it has the same number of elements
                // Again, we ignore everything beyond the square part of R since it's zero
                back_substitution(R.data(), Y.data(), B.data(),
                                  n_dof - 1, n_dof - 1, stencil_size - 1,
                                  false, false);
                
                // Finally, we fill A^+ with the submatrix B^+ and zeros for the first row and column
                // Store A^+ in A since we don't need A anymore and it has the same number of elements
                for (u_int8_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                    for (u_int16_t i_neighbor_loc = 0; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                        size_t ind_A = i_dof * stencil_size + i_neighbor_loc;
                        size_t ind_B = (i_dof - 1) * (stencil_size - 1) + (i_neighbor_loc - 1);
                        if ((i_dof == 0) || (i_neighbor_loc == 0)) {
                            A[ind_A] = 0.0;
                        } else {
                            A[ind_A] = B[ind_B];
                        }
                    }
                }
            } else {
                // Compute the QR decomposition of A
                QR_householder_noQ(A.data(), R.data(), stencil_size, n_dof);

                // Now we need to solve the system R^T R A^+ = A^T
                // First, we use forward substitution to solve R^T Y = A^T
                // We ignore everything beyond the square part of R since it's zero
                forward_substitution(R.data(), A.data(), Y.data(),
                                     n_dof, n_dof, stencil_size,
                                     true, true);

                // Next, we use back substitution to solve R A^+ = Y
                // Store A^+ in A since we don't need A anymore and it has the same number of elements
                // Again, we ignore everything beyond the square part of R since it's zero
                back_substitution(R.data(), Y.data(), A.data(),
                                  n_dof, n_dof, stencil_size,
                                  false, false);
            }

            // At this point, A contains the Moore-Penrose pseudoinverse of the reconstruction matrix.
            // We store it in the v_reconstruction_matrices vector and update the v_offsets_reconstruction_matrices vector
            // with the proper offset for the stencil.
            for (u_int16_t i_dof = 0; i_dof < n_dof; ++i_dof) {
                for (u_int8_t i_neighbor_loc = 0; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                    v_reconstruction_matrices.push_back(A[i_dof * stencil_size + i_neighbor_loc]);
                }
            }
            v_offsets_reconstruction_matrices.push_back(v_reconstruction_matrices.size());

            // Store the transformed areas for the stencil
            for (u_int8_t i_neighbor_loc = 0; i_neighbor_loc < stencil_size; ++i_neighbor_loc) {
                v_transformed_areas.push_back(area_trans[i_neighbor_loc]);
            }
        }
    }

    // Allocate device arrays
    offsets_reconstruction_matrices = Kokkos::View<u_int32_t *>("offsets_reconstruction_matrices", v_offsets_reconstruction_matrices.size());
    reconstruction_matrices = Kokkos::View<rtype *>("reconstruction_matrices", v_reconstruction_matrices.size());
    transformed_areas = Kokkos::View<rtype *>("transformed_areas", v_transformed_areas.size());

    // Set up host mirrors
    h_offsets_reconstruction_matrices = Kokkos::create_mirror_view(offsets_reconstruction_matrices);
    h_reconstruction_matrices = Kokkos::create_mirror_view(reconstruction_matrices);
    h_transformed_areas = Kokkos::create_mirror_view(transformed_areas);

    // Fill host mirrors
    for (u_int32_t i = 0; i < v_offsets_reconstruction_matrices.size(); ++i) {
        h_offsets_reconstruction_matrices(i) = v_offsets_reconstruction_matrices[i];
    }
    for (u_int32_t i = 0; i < v_reconstruction_matrices.size(); ++i) {
        h_reconstruction_matrices(i) = v_reconstruction_matrices[i];
    }
    for (u_int32_t i = 0; i < v_transformed_areas.size(); ++i) {
        h_transformed_areas(i) = v_transformed_areas[i];
    }

    // Copy from host to device
    Kokkos::deep_copy(offsets_reconstruction_matrices, h_offsets_reconstruction_matrices);
    Kokkos::deep_copy(reconstruction_matrices, h_reconstruction_matrices);
    Kokkos::deep_copy(transformed_areas, h_transformed_areas);
}

void TENO::compute_oscillation_indicator() {
    oscillation_indicator = Kokkos::View<rtype *>("oscillation_indicator", n_dof * n_dof);
    h_oscillation_indicator = Kokkos::create_mirror_view(oscillation_indicator);

    // i_dof indexes the basis function phi_i.
    // The polynomial indices for phi_i are taken as
    // poly_indices(i_dof, 0) and poly_indices(i_dof, 1)
    for (u_int8_t i_dof = 0; i_dof < n_dof; i_dof++) {

        // j_dof indexes the basis function phi_j.
        // The polynomial indices for phi_j are taken as
        // poly_indices(j_dof, 0) and poly_indices(j_dof, 1)
        for (u_int8_t j_dof = 0; j_dof < n_dof; j_dof++) {
            size_t ind = i_dof * n_dof + j_dof;
            h_oscillation_indicator(ind) = 0.0;

            // k_dof indexes the order of the derivative to be taken.
            // The derivative order in dimension i_dim is poly_indices(k_dof, i_dim)
            // We skip k_dof = 0 because this represents the zeroth derivative in all dimensions,
            // i.e. the basis function itself.
            for (u_int8_t k_dof = 1; k_dof < n_dof; k_dof++) {

                // i_quad indexes the quadrature points
                for (u_int8_t i_quad = 0; i_quad < quadrature.h_points.extent(0); ++i_quad) {
                    // Evaluate the requested derivative of phi_i at the quadrature point
                    // i_dim indexes the dimension of the derivative
                    rtype dphi_i = 1.0;
                    for (u_int8_t i_dim = 0; i_dim < N_DIM; ++i_dim) {
                        rtype dphi_i_dim = basis_derivative_1D(poly_indices(k_dof, i_dim),
                                                               poly_indices(i_dof, i_dim),
                                                               quadrature.h_points(i_quad, i_dim));
                        dphi_i *= dphi_i_dim;
                    }
                    // Evaluate the requested derivative of phi_j at the quadrature point
                    // i_dim indexes the dimension of the derivative
                    rtype dphi_j = 1.0;
                    for (u_int8_t i_dim = 0; i_dim < N_DIM; ++i_dim) {
                        rtype dphi_j_dim = basis_derivative_1D(poly_indices(k_dof, i_dim),
                                                               poly_indices(j_dof, i_dim),
                                                               quadrature.h_points(i_quad, i_dim));
                        dphi_j *= dphi_j_dim;
                    }
                    // The integrand is the product of the two derivatives
                    h_oscillation_indicator(ind) += quadrature.h_weights(i_quad) * dphi_i * dphi_j;
                }
            }
        }
    }

    // Copy from host to device
    Kokkos::deep_copy(oscillation_indicator, h_oscillation_indicator);
}

struct TENOFunctor {
    public:
        /**
         * @brief Construct a new TENOFunctor object
         * @param cells_of_face Cells of face.
         * @param face_solution Face solution.
         * @param solution Cell solution.
         */
        TENOFunctor(Kokkos::View<int32_t *[2]> cells_of_face,
                    Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution,
                    Kokkos::View<rtype *[N_CONSERVATIVE]> solution) :
                        cells_of_face(cells_of_face),
                        face_solution(face_solution),
                        solution(solution) {}
        
        /**
         * @brief Overloaded operator for TENO face reconstruction.
         * @param i_face Face index.
         */
        KOKKOS_INLINE_FUNCTION
        void operator()(const u_int32_t i_face) const {
            // Empty
        }
    
    private:
        Kokkos::View<int32_t *[2]> cells_of_face;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution;
        Kokkos::View<rtype *[N_CONSERVATIVE]> solution;
};

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                Kokkos::View<rtype *[2][N_CONSERVATIVE]> face_solution) {
    TENOFunctor recon_functor(mesh->cells_of_face,
                              face_solution,
                              solution);
    Kokkos::parallel_for(mesh->n_faces, recon_functor);
}
