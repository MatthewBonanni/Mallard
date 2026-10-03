/**
 * @file distributed_mesh_test.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief The scalable distributed setup reproduces the setup from a global mesh.
 * @version 0.4
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <vector>

#include "comm.h"
#include "distributed_mesh.h"
#include "distribution.h"
#include "mesh.h"
#include "mesh_block.h"
#include "partition.h"
#include "test_fixtures.h"

namespace {

std::vector<std::string> mesh_inputs() {
    if constexpr (N_DIM == 2) {
        return {"type = \"cartesian_tri\"\nNx = 9\nNy = 7\n", "type = \"wedge\"\nNx = 11\nNy = 6\n"};
    }
    return {"type = \"cartesian_mixed\"\nNx = 6\nNy = 3\nNz = 3\n", "type = \"cartesian_tet\"\nNx = 3\nNy = 3\nNz = 2\n"};
}

/**
 * @brief Reference: the local mesh built from the whole mesh, which every
 *        rank holds (the setup before DistributedMesh).
 */
std::shared_ptr<Mesh> build_local_mesh_from_global(Mesh & global, const std::vector<int> & owner, int halo_layers,
                                       Distribution & dist) {
    const int me = comm::rank();
    const uint32_t n_global = global.n_cells;
    if (owner.size() != n_global) throw std::invalid_argument("build_local_mesh: one owner per cell");

    // Vertex neighbors on the global mesh
    std::vector<std::vector<uint32_t>> cells_of_node(global.n_nodes);
    for (uint32_t c = 0; c < n_global; c++) {
        for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
            cells_of_node[global.h_node_of_cell(c, k)].push_back(c);
        }
    }

    // Owned cells, then halo layers by breadth-first search over vertex neighbors
    std::vector<int32_t> local_of(n_global, -1);
    dist = Distribution();
    dist.halo_layers = halo_layers;
    for (uint32_t c = 0; c < n_global; c++) {
        if (owner[c] != me) continue;
        local_of[c] = dist.global_cell.size();
        dist.global_cell.push_back(c);
        dist.layer.push_back(0);
    }
    dist.n_owned = dist.global_cell.size();
    size_t layer_begin = 0;
    for (int l = 1; l <= halo_layers; l++) {
        const size_t layer_end = dist.global_cell.size();
        std::vector<uint32_t> next;
        for (size_t i = layer_begin; i < layer_end; i++) {
            const uint32_t c = dist.global_cell[i];
            for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
                for (uint32_t nb : cells_of_node[global.h_node_of_cell(c, k)]) {
                    if (local_of[nb] == -1) {
                        local_of[nb] = -2;
                        next.push_back(nb);
                    }
                }
            }
        }
        std::sort(next.begin(), next.end());
        for (uint32_t c : next) {
            local_of[c] = dist.global_cell.size();
            dist.global_cell.push_back(c);
            dist.layer.push_back(l);
        }
        layer_begin = layer_end;
    }

    // Local nodes and cells
    std::unordered_map<uint32_t, uint32_t> local_node;
    std::vector<std::array<rtype, N_DIM>> nodes;
    std::vector<std::vector<uint32_t>> cells(dist.global_cell.size());
    for (size_t i = 0; i < dist.global_cell.size(); i++) {
        const uint32_t c = dist.global_cell[i];
        for (uint32_t k = 0; k < global.h_n_nodes_of_cell(c); k++) {
            const uint32_t gn = global.h_node_of_cell(c, k);
            auto [it, inserted] = local_node.emplace(gn, nodes.size());
            if (inserted) {
                std::array<rtype, N_DIM> x;
                for (int d = 0; d < N_DIM; d++) x[d] = global.h_node_coords(gn, d);
                nodes.push_back(x);
            }
            cells[i].push_back(it->second);
        }
    }

    // Boundary faces: global boundary zones, and faces towards non-local cells
    std::vector<const std::string *> zone_of_face(global.n_faces, nullptr);
    std::vector<std::string> zone_names;
    zone_names.reserve(global.n_face_zones());
    for (FaceZone & zone : *global.face_zones()) {
        if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
        zone_names.push_back(zone.get_name());
    }
    {
        size_t k = 0;
        for (FaceZone & zone : *global.face_zones()) {
            if (zone.get_type() != FaceZoneType::BOUNDARY) continue;
            for (uint32_t i = 0; i < zone.n_faces(); i++) zone_of_face[zone.h_faces(i)] = &zone_names[k];
            k++;
        }
    }
    std::vector<Mesh::BoundaryFace> boundary_faces;
    for (uint32_t c : dist.global_cell) {
        for (uint32_t k = 0; k < global.h_n_faces_of_cell(c); k++) {
            const uint32_t f = global.h_face_of_cell(c, k);
            const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
            const int32_t other = (c0 == static_cast<int32_t>(c)) ? c1 : c0;
            std::string zone;
            if (other < 0) {
                zone = zone_of_face[f] ? *zone_of_face[f] : "unassigned";
            } else if (local_of[other] < 0) {
                zone = PARTITION_ZONE;
            } else {
                continue;
            }
            std::vector<uint32_t> face_nodes;
            for (uint32_t j = 0; j < global.h_n_nodes_of_face(f); j++) {
                face_nodes.push_back(local_node.at(global.h_node_of_face(f, j)));
            }
            boundary_faces.push_back({std::move(face_nodes), zone});
        }
    }

    auto local = std::make_shared<Mesh>();
    // Global ids first: they order the faces and neighbor lists like the serial mesh's
    local->h_global_cell_id = dist.global_cell;
    local->n_global_cells = n_global;
    local->init_from_connectivity(nodes, cells, boundary_faces);
    local->n_owned_cells = dist.n_owned;
    local->n_reconstructed_cells = std::count_if(dist.layer.begin(), dist.layer.end(), [](uint8_t l) { return l <= 1; });
    local->n_complete_cells =
        std::count_if(dist.layer.begin(), dist.layer.end(), [&](uint8_t l) { return l < halo_layers; });

    std::vector<int> halo_owner;
    for (size_t i = dist.n_owned; i < dist.global_cell.size(); i++) halo_owner.push_back(owner[dist.global_cell[i]]);
    plan_halo_exchange(dist, halo_owner);
    return local;
}

using Point = std::array<rtype, N_DIM>;

/** @brief Face centroids of every boundary zone, sorted. */
std::map<std::string, std::vector<Point>> zone_faces(Mesh & mesh) {
    std::map<std::string, std::vector<Point>> zones;
    for (FaceZone & zone : *mesh.face_zones()) {
        auto & points = zones[zone.get_name()];
        for (uint32_t j = 0; j < zone.n_faces(); j++) {
            Point x;
            FOR_I_DIM x[i] = mesh.h_face_coords(zone.h_faces(j), i);
            points.push_back(x);
        }
        std::sort(points.begin(), points.end());
    }
    return zones;
}

void expect_same(Mesh & mesh, const Distribution & dist, Mesh & ref, const Distribution & ref_dist) {
    EXPECT_EQ(dist.n_owned, ref_dist.n_owned);
    ASSERT_EQ(dist.global_cell, ref_dist.global_cell);
    EXPECT_EQ(dist.layer, ref_dist.layer);
    EXPECT_EQ(dist.neighbors, ref_dist.neighbors);
    EXPECT_EQ(dist.send_cells, ref_dist.send_cells);
    EXPECT_EQ(dist.recv_cells, ref_dist.recv_cells);
    EXPECT_EQ(mesh.h_global_cell_id, ref.h_global_cell_id);
    EXPECT_EQ(mesh.n_global_cells, ref.n_global_cells);
    EXPECT_EQ(mesh.n_reconstructed(), ref.n_reconstructed());
    EXPECT_EQ(mesh.n_complete(), ref.n_complete());
    ASSERT_EQ(mesh.n_cells, ref.n_cells);
    EXPECT_EQ(mesh.n_nodes, ref.n_nodes);
    EXPECT_EQ(mesh.n_faces, ref.n_faces);
    for (uint32_t c = 0; c < mesh.n_cells; c++) {
        ASSERT_EQ(mesh.h_n_nodes_of_cell(c), ref.h_n_nodes_of_cell(c));
        for (uint32_t k = 0; k < mesh.h_n_nodes_of_cell(c); k++) {
            FOR_I_DIM ASSERT_EQ(mesh.h_node_coords(mesh.h_node_of_cell(c, k), i),
                                ref.h_node_coords(ref.h_node_of_cell(c, k), i));
        }
        EXPECT_EQ(mesh.h_cell_volume(c), ref.h_cell_volume(c));
        FOR_I_DIM EXPECT_EQ(mesh.h_cell_coords(c, i), ref.h_cell_coords(c, i));
    }
    EXPECT_EQ(zone_faces(mesh), zone_faces(ref));
}

} // namespace

/**
 * @brief The distributed setup (face matching by hashing, migration, halo
 *        layers by distributed search, and regrowing them) must build exactly
 *        the local meshes and exchange plans that the global-mesh setup builds
 *        for the same owners, so solutions stay rank-count independent.
 */
TEST(DistributedMeshTest, LocalMeshesMatchTheSetupFromTheGlobalMesh) {
    for (const std::string & mesh_input : mesh_inputs()) {
        SCOPED_TRACE(mesh_input);
        const toml::value input = parse_toml("[mesh]\n" + mesh_input);
        Mesh global;
        global.init(input);
        DistributedMesh distributed(read_mesh_block(input));

        // The dual graph of each block cell: its face neighbors in the global mesh
        for (uint32_t c = 0; c < distributed.n_block_cells(); c++) {
            const uint64_t g = distributed.first_cell() + c;
            std::vector<uint64_t> expected, graph(distributed.graph_neighbors().begin() + distributed.graph_offsets()[c],
                                                  distributed.graph_neighbors().begin() + distributed.graph_offsets()[c + 1]);
            for (uint32_t k = 0; k < global.h_n_faces_of_cell(g); k++) {
                const uint32_t f = global.h_face_of_cell(g, k);
                const int32_t c0 = global.h_cells_of_face(f, 0), c1 = global.h_cells_of_face(f, 1);
                if (c1 >= 0) expected.push_back(c0 == int32_t(g) ? c1 : c0);
            }
            std::sort(expected.begin(), expected.end());
            std::sort(graph.begin(), graph.end());
            ASSERT_EQ(graph, expected) << "cell " << g;
        }

        const std::vector<int> owner = partition_hilbert(distributed, comm::size());
        const std::vector<int32_t> all_owners = comm::allgatherv(std::vector<int32_t>(owner.begin(), owner.end()));
        distributed.distribute(owner);
        // One layer, then grown to three
        for (int layers : {1, 3}) {
            SCOPED_TRACE(layers);
            Distribution dist, ref_dist;
            auto mesh = distributed.build_local_mesh(layers, dist);
            auto ref = build_local_mesh_from_global(global, std::vector<int>(all_owners.begin(), all_owners.end()), layers, ref_dist);
            expect_same(*mesh, dist, *ref, ref_dist);
        }
    }
}

namespace {

/** @brief Test weights by global id: the first fifth of the cells (a band of the generated meshes) weighs 20. */
uint64_t band_weight(uint64_t g, uint64_t n) { return g < n / 5 ? 20 : 1; }

/**
 * @brief Weights of the block cells as a rebalance gets them: computed by the
 *        owners of a first partition and moved to the blocks.
 */
std::vector<uint64_t> weights_from_owners(DistributedMesh & distributed) {
    distributed.distribute(partition_hilbert(distributed, comm::size()));
    Distribution dist;
    distributed.build_local_mesh(1, dist);
    std::vector<uint64_t> owned(dist.n_owned);
    for (uint32_t c = 0; c < dist.n_owned; c++) owned[c] = band_weight(dist.global_cell[c], distributed.n_global_cells());
    return distributed.owned_to_block(owned);
}

} // namespace

/**
 * @brief The distributed sample sort must give the same order as sorting all
 *        cells along the curve on one rank, and parts must start where the
 *        weight prefix along that order crosses multiples of the total over
 *        the part count, with the weights moved from the owners.
 */
TEST(DistributedMeshTest, HilbertPartitionSplitsTheGlobalCurveOrderByWeight) {
    const toml::value input = parse_toml("[mesh]\n" + mesh_inputs()[0]);
    DistributedMesh distributed(read_mesh_block(input));
    const uint64_t n = distributed.n_global_cells();

    const auto centers = distributed.block_cell_centers();
    std::vector<double> flat;
    for (const auto & x : centers) flat.insert(flat.end(), x.begin(), x.end());
    std::vector<double> all;
    {
        // Gather every center (test-only: a global array)
        std::vector<std::vector<double>> send(comm::size(), flat);
        for (const auto & from : comm::alltoallv(send)) all.insert(all.end(), from.begin(), from.end());
    }
    ASSERT_EQ(all.size() / N_DIM, n);
    std::array<double, N_DIM> lo, hi;
    lo.fill(1e300);
    hi.fill(-1e300);
    for (uint64_t c = 0; c < n; c++) {
        FOR_I_DIM {
            lo[i] = std::min(lo[i], all[c * N_DIM + i]);
            hi[i] = std::max(hi[i], all[c * N_DIM + i]);
        }
    }
    std::vector<std::pair<uint64_t, uint64_t>> order(n);
    for (uint64_t c = 0; c < n; c++) {
        std::array<double, N_DIM> x;
        FOR_I_DIM x[i] = all[c * N_DIM + i];
        order[c] = {hilbert_key(x, lo, hi), c};
    }
    std::sort(order.begin(), order.end());

    for (const bool weighted : {false, true}) {
        SCOPED_TRACE(weighted ? "weighted" : "unweighted");
        const std::vector<uint64_t> weights = weighted ? weights_from_owners(distributed) : std::vector<uint64_t>{};
        const std::vector<int> owner = partition_hilbert(distributed, comm::size(), weights);
        uint64_t total = 0;
        for (uint64_t g = 0; g < n; g++) total += weighted ? band_weight(g, n) : 1;
        std::vector<int> expected(n);
        uint64_t prefix = 0;
        for (const auto & [key, g] : order) {
            expected[g] = (prefix * comm::size()) / total;
            prefix += weighted ? band_weight(g, n) : 1;
        }
        for (uint32_t c = 0; c < distributed.n_block_cells(); c++) {
            ASSERT_EQ(owner[c], expected[distributed.first_cell() + c]) << "cell " << distributed.first_cell() + c;
        }
    }
}

/** @brief dKaMinPar honors the weights: no part above (1 + epsilon) times the mean weight plus one cell. */
TEST(DistributedMeshTest, GraphPartitionBalancesTheWeights) {
    if (!have_graph_partitioner()) GTEST_SKIP() << "built without a graph partitioner";
    // Big enough parts for the heavy cells: on 32 cells per rank dKaMinPar 3.7 never finishes balancing them
    const toml::value input = parse_toml(N_DIM == 2 ? "[mesh]\ntype = \"cartesian_tri\"\nNx = 40\nNy = 30\n"
                                                    : "[mesh]\ntype = \"cartesian_tet\"\nNx = 8\nNy = 6\nNz = 5\n");
    DistributedMesh distributed(read_mesh_block(input));
    const std::vector<uint64_t> weights = weights_from_owners(distributed);
    const std::vector<int> owner = partition_graph(distributed, comm::size(), weights);
    std::vector<uint64_t> part_weight(comm::size(), 0);
    for (uint32_t c = 0; c < distributed.n_block_cells(); c++) part_weight[owner[c]] += weights[c];
    comm::allreduce(std::span<uint64_t>(part_weight), comm::Op::SUM);
    uint64_t total = 0, max_part = 0;
    for (uint64_t w : part_weight) {
        total += w;
        max_part = std::max(max_part, w);
    }
    const double mean = std::ceil(double(total) / comm::size());
    EXPECT_LE(double(max_part), (1.0 + GRAPH_PARTITION_EPSILON) * mean + 20.0);
}

/** @brief Relabeling a renumbered copy of the current partition gives back the current partition. */
TEST(DistributedMeshTest, RelabeledPartsStayWithTheirCurrentRanks) {
    const toml::value input = parse_toml("[mesh]\n" + mesh_inputs()[0]);
    DistributedMesh distributed(read_mesh_block(input));
    const std::vector<uint64_t> weights = weights_from_owners(distributed);
    const int p = comm::size();
    const std::vector<int> current = partition_hilbert(distributed, p, weights);
    std::vector<int> renumbered(current.size());
    for (size_t c = 0; c < current.size(); c++) renumbered[c] = (current[c] * 7 + 3) % p;
    if (std::gcd(7, p) != 1) GTEST_SKIP() << "7 must be invertible modulo the rank count";
    EXPECT_EQ(relabel_parts(current, renumbered, weights, p), current);
}

TEST(DistributedMeshTest, PeriodicHaloLayersFollowVertexNeighborsAcrossSeams) {
    // Each halo layer holds exactly the cells one vertex-neighbor step further,
    // including neighbors across the seams of a fully periodic box
    const std::string input = N_DIM == 2
        ? "[mesh]\ntype = \"cartesian_tri\"\nNx = 9\nNy = 7\nperiodic = [\"x\", \"y\"]\n"
        : "[mesh]\ntype = \"cartesian_tet\"\nNx = 4\nNy = 3\nNz = 3\nperiodic = [\"x\", \"y\", \"z\"]\n";
    const toml::value parsed = parse_toml(input);
    Mesh global;
    global.init(parsed);
    DistributedMesh distributed(read_mesh_block(parsed), Mesh::periodic_pairs(parsed));
    const std::vector<int> block_owner = partition_hilbert(distributed, comm::size());
    std::vector<uint64_t> pairs;
    for (uint32_t c = 0; c < block_owner.size(); c++) {
        pairs.insert(pairs.end(), {distributed.first_cell() + c, uint64_t(block_owner[c])});
    }
    pairs = comm::allgatherv(pairs);
    std::vector<int> owner(global.n_cells);
    for (size_t i = 0; i < pairs.size(); i += 2) owner[pairs[i]] = pairs[i + 1];
    distributed.distribute(block_owner);
    const int layers = 3;
    Distribution dist;
    distributed.build_local_mesh(layers, dist);

    std::vector<int> expected(global.n_cells, -1);
    std::vector<uint32_t> front;
    for (uint32_t c = 0; c < global.n_cells; c++) {
        if (owner[c] == comm::rank()) {
            expected[c] = 0;
            front.push_back(c);
        }
    }
    for (int l = 1; l <= layers; l++) {
        std::vector<uint32_t> next;
        for (uint32_t c : front) {
            for (uint32_t k = global.h_offsets_cells_of_cell(c); k < global.h_offsets_cells_of_cell(c + 1); k++) {
                const uint32_t nb = global.h_cells_of_cell(k);
                if (expected[nb] < 0) {
                    expected[nb] = l;
                    next.push_back(nb);
                }
            }
        }
        front = next;
    }
    const auto n_expected = std::count_if(expected.begin(), expected.end(), [](int l) { return l >= 0; });
    EXPECT_EQ(dist.global_cell.size(), size_t(n_expected));
    for (size_t i = 0; i < dist.global_cell.size(); i++) {
        EXPECT_EQ(int(dist.layer[i]), expected[dist.global_cell[i]]) << "cell " << dist.global_cell[i];
    }
}
