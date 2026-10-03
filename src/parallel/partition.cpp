/**
 * @file partition.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell partitioning for distributed runs.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include "partition.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

#include "comm.h"
#include "distributed_mesh.h"

#ifdef Mallard_HAS_KAMINPAR
#include <dkaminpar.h>
#endif

uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi) {
    constexpr int bits = 63 / N_DIM;
    constexpr uint32_t max_coord = (uint32_t(1) << bits) - 1;
    std::array<uint32_t, N_DIM> X;
    for (int d = 0; d < N_DIM; d++) {
        const double span = hi[d] - lo[d];
        const double t = span > 0.0 ? (x[d] - lo[d]) / span : 0.0;
        X[d] = static_cast<uint32_t>(std::clamp(t, 0.0, 1.0) * max_coord);
    }
    // Skilling, "Programming the Hilbert curve" (AIP Conf. Proc. 707, 2004): axes to transpose
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        const uint32_t P = Q - 1;
        for (int i = 0; i < N_DIM; i++) {
            if (X[i] & Q) {
                X[0] ^= P;
            } else {
                const uint32_t t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    for (int i = 1; i < N_DIM; i++) X[i] ^= X[i - 1];
    uint32_t t = 0;
    for (uint32_t Q = uint32_t(1) << (bits - 1); Q > 1; Q >>= 1) {
        if (X[N_DIM - 1] & Q) t ^= Q - 1;
    }
    for (int i = 0; i < N_DIM; i++) X[i] ^= t;
    // Interleave the transposed bits, most significant first
    uint64_t key = 0;
    for (int b = bits - 1; b >= 0; b--) {
        for (int i = 0; i < N_DIM; i++) key = (key << 1) | ((X[i] >> b) & 1u);
    }
    return key;
}


bool have_graph_partitioner() {
#ifdef Mallard_HAS_KAMINPAR
    return true;
#else
    return false;
#endif
}


namespace {

/** @brief Weight of block cell c: 1 without weights. */
uint64_t weight_of(const std::vector<uint64_t> & weights, size_t c) { return weights.empty() ? 1 : weights[c]; }

void check_weights(const DistributedMesh & mesh, const std::vector<uint64_t> & weights) {
    const bool bad = !weights.empty() && weights.size() != mesh.n_block_cells();
    if (comm::allreduce(uint32_t(bad), comm::Op::MAX) != 0) {
        throw std::invalid_argument("partition: one weight per block cell.");
    }
}

} // namespace

std::vector<int> partition_hilbert(const DistributedMesh & mesh, int n_parts, const std::vector<uint64_t> & weights) {
    check_weights(mesh, weights);
    const int p = comm::size();
    const auto centers = mesh.block_cell_centers();
    std::array<double, N_DIM> lo, hi;
    lo.fill(std::numeric_limits<double>::max());
    hi.fill(std::numeric_limits<double>::lowest());
    for (const auto & x : centers) {
        for (int d = 0; d < N_DIM; d++) {
            lo[d] = std::min(lo[d], x[d]);
            hi[d] = std::max(hi[d], x[d]);
        }
    }
    lo = comm::allreduce(lo, comm::Op::MIN);
    hi = comm::allreduce(hi, comm::Op::MAX);
    // (key, global id, weight): key and id are unique, so splitters are exact
    using Item = std::array<uint64_t, 3>;
    std::vector<Item> items(centers.size());
    for (size_t c = 0; c < centers.size(); c++) {
        items[c] = {hilbert_key(centers[c], lo, hi), mesh.first_cell() + c, weight_of(weights, c)};
    }
    std::sort(items.begin(), items.end());

    // Sample sort: evenly spaced samples from every rank pick p - 1 splitters.
    // Parts come from global positions, so splitters only balance the sort
    // itself; a bounded oversampling keeps the gathered samples O(p).
    constexpr int MAX_SAMPLES = 64;
    const int n_samples = std::min(p, MAX_SAMPLES);
    std::vector<uint64_t> samples;
    for (int k = 0; k < n_samples && !items.empty(); k++) {
        const Item & s = items[(items.size() * k) / n_samples];
        samples.push_back(s[0]);
        samples.push_back(s[1]);
    }
    const std::vector<uint64_t> all = comm::allgatherv(samples);
    using Key = std::pair<uint64_t, uint64_t>;
    std::vector<Key> sorted_samples;
    for (size_t i = 0; i < all.size(); i += 2) sorted_samples.push_back({all[i], all[i + 1]});
    std::sort(sorted_samples.begin(), sorted_samples.end());
    std::vector<Key> splitters;
    for (int k = 1; k < p && !sorted_samples.empty(); k++) {
        splitters.push_back(sorted_samples[(sorted_samples.size() * k) / p]);
    }
    std::vector<std::vector<uint64_t>> send(p);
    for (const Item & item : items) {
        const int dest = std::upper_bound(splitters.begin(), splitters.end(), Key{item[0], item[1]}) - splitters.begin();
        send[dest].insert(send[dest].end(), item.begin(), item.end());
    }
    items.clear();
    for (const auto & from : comm::alltoallv(send)) {
        for (size_t i = 0; i < from.size(); i += 3) items.push_back({from[i], from[i + 1], from[i + 2]});
    }
    std::sort(items.begin(), items.end());

    // Weight prefix along the curve -> part, sent to each cell's block rank
    uint64_t local_weight = 0;
    for (const Item & item : items) local_weight += item[2];
    const std::vector<uint64_t> rank_weights = comm::allgatherv(std::vector<uint64_t>{local_weight});
    uint64_t prefix = 0, total = 0;
    for (int r = 0; r < p; r++) {
        if (r < comm::rank()) prefix += rank_weights[r];
        total += rank_weights[r];
    }
    if (total == 0 || total > std::numeric_limits<uint64_t>::max() / uint64_t(n_parts)) {
        throw std::invalid_argument("partition_hilbert: the total weight must be positive and below 2^64 / parts.");
    }
    const auto & dist = mesh.cell_distribution();
    send.assign(p, {});
    for (const Item & item : items) {
        const uint64_t g = item[1];
        const int r = std::upper_bound(dist.begin(), dist.end(), g) - dist.begin() - 1;
        send[r].push_back(g);
        send[r].push_back((prefix * n_parts) / total);
        prefix += item[2];
    }
    std::vector<int> owner(mesh.n_block_cells());
    for (const auto & from : comm::alltoallv(send)) {
        for (size_t i = 0; i < from.size(); i += 2) owner[from[i] - mesh.first_cell()] = from[i + 1];
    }
    return owner;
}

std::vector<int> partition_graph(const DistributedMesh & mesh, int n_parts, const std::vector<uint64_t> & weights) {
    check_weights(mesh, weights);
#ifdef Mallard_HAS_KAMINPAR
    using kaminpar::dist::GlobalEdgeID;
    using kaminpar::dist::GlobalNodeID;
    using kaminpar::dist::GlobalNodeWeight;
    const auto & d = mesh.cell_distribution();
    std::vector<GlobalNodeID> vtxdist(d.begin(), d.end());
    std::vector<GlobalEdgeID> xadj(mesh.graph_offsets().begin(), mesh.graph_offsets().end());
    std::vector<GlobalNodeID> adjncy(mesh.graph_neighbors().begin(), mesh.graph_neighbors().end());
    std::vector<GlobalNodeWeight> vwgt(weights.begin(), weights.end());
    // The same partition on every call, so per-rank caches (TENO stencils) can be reused
    kaminpar::dKaMinPar::reseed(0);
    kaminpar::dKaMinPar partitioner(comm::world(), 1, kaminpar::dist::create_default_context());
    partitioner.set_output_level(kaminpar::OutputLevel::QUIET);
    partitioner.copy_graph(vtxdist, xadj, adjncy, vwgt);
    std::vector<kaminpar::dist::BlockID> blocks(mesh.n_block_cells());
    partitioner.compute_partition(n_parts, GRAPH_PARTITION_EPSILON, blocks);
    return std::vector<int>(blocks.begin(), blocks.end());
#else
    (void)mesh;
    (void)n_parts;
    throw std::runtime_error("This build has no graph partitioner (configure with Mallard_ENABLE_KAMINPAR=ON).");
#endif
}

std::vector<int> relabel_parts(const std::vector<int> & current, const std::vector<int> & proposed,
                               const std::vector<uint64_t> & weights, int n_parts) {
    const bool bad = current.size() != proposed.size() || (!weights.empty() && weights.size() != current.size());
    if (comm::allreduce(uint32_t(bad), comm::Op::MAX) != 0) {
        throw std::invalid_argument("relabel_parts: one current part, proposed part and weight per cell.");
    }
    // Weight shared by each (current, proposed) pair, summed over the ranks
    std::map<std::pair<int, int>, uint64_t> local;
    for (size_t c = 0; c < current.size(); c++) local[{current[c], proposed[c]}] += weight_of(weights, c);
    std::vector<uint64_t> flat;
    for (const auto & [pair, w] : local) flat.insert(flat.end(), {uint64_t(pair.first), uint64_t(pair.second), w});
    std::map<std::pair<int, int>, uint64_t> shared;
    const std::vector<uint64_t> all = comm::allgatherv(flat);
    for (size_t i = 0; i < all.size(); i += 3) shared[{int(all[i]), int(all[i + 1])}] += all[i + 2];

    // Heaviest pairs first; ties by part numbers, so every rank agrees
    std::vector<std::pair<uint64_t, std::pair<int, int>>> order;
    for (const auto & [pair, w] : shared) order.push_back({w, pair});
    std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::vector<int> label(n_parts, -1);
    std::vector<bool> taken(n_parts, false);
    for (const auto & [w, pair] : order) {
        const auto [from, to] = pair;
        if (label[to] >= 0 || taken[from]) continue;
        label[to] = from;
        taken[from] = true;
    }
    int next = 0;
    for (int q = 0; q < n_parts; q++) {
        if (label[q] >= 0) continue;
        while (taken[next]) next++;
        label[q] = next;
        taken[next] = true;
    }
    std::vector<int> relabeled(proposed.size());
    for (size_t c = 0; c < proposed.size(); c++) relabeled[c] = label[proposed[c]];
    return relabeled;
}
