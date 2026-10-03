/**
 * @file partition.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Cell partitioning for distributed runs.
 * @version 0.3
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#ifndef PARTITION_H
#define PARTITION_H

#include <array>
#include <cstdint>
#include <vector>

#include "common_typedef.h"

class DistributedMesh;

/**
 * @brief Position along the Hilbert curve of a point in the box [lo, hi]
 *        (any dimension; 63 / N_DIM bits per axis).
 */
uint64_t hilbert_key(const std::array<double, N_DIM> & x, const std::array<double, N_DIM> & lo,
                     const std::array<double, N_DIM> & hi);

/**
 * @brief Owner rank of every block cell of a distributed mesh: cells sorted
 *        along the Hilbert curve of their vertex averages (ties by global id)
 *        by a distributed sample sort, split into n_parts contiguous pieces of
 *        nearly equal weight (collective). A cell goes to the part its weight
 *        prefix along the curve starts in, so each part's weight is within the
 *        largest cell weight of the total over n_parts.
 * @param weights Weight of every block cell; empty for unit weights.
 */
std::vector<int> partition_hilbert(const DistributedMesh & mesh, int n_parts,
                                   const std::vector<uint64_t> & weights = {});

/** @brief Allowed imbalance of graph partitions: parts weigh at most (1 + epsilon) times the mean. */
inline constexpr double GRAPH_PARTITION_EPSILON = 0.03;

/**
 * @brief Owner rank of every block cell of a distributed mesh from dKaMinPar
 *        on its dual graph (collective). Requires Mallard_ENABLE_KAMINPAR.
 * @param weights Weight of every block cell; empty for unit weights.
 */
std::vector<int> partition_graph(const DistributedMesh & mesh, int n_parts,
                                 const std::vector<uint64_t> & weights = {});

/**
 * @brief Renumber the parts of a new partition so that as much weight as
 *        possible stays with its current part: (current, proposed) pairs are
 *        matched greedily by decreasing shared weight (collective).
 * @param current Current part of every block cell.
 * @param proposed New part of every block cell.
 * @param weights Weight of every block cell; empty for unit weights.
 * @return The proposed partition with renumbered parts.
 */
std::vector<int> relabel_parts(const std::vector<int> & current, const std::vector<int> & proposed,
                               const std::vector<uint64_t> & weights, int n_parts);

/** @brief Whether this build has a graph partitioner. */
bool have_graph_partitioner();

#endif // PARTITION_H
