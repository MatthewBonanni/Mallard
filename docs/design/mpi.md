# Design: distributed memory (MPI)

Status: proposal, for review. Tracks issue #2.

## Goals

- Run on one GPU per MPI rank, from one node (8 GPUs) to thousands of GPUs.
- **No rank ever holds the global mesh.** Reading, partitioning, setup and output are all distributed, so the mesh size is limited only by aggregate memory.
- Results independent of the rank count up to round-off (bitwise identical stencils and reconstructions; only the summation order of fluxes into a cell may differ).
- A single-rank build without MPI keeps working and stays the default for development and CI.
- Everything here is dimension-agnostic: it works on cells, faces as node lists and the cell graph, so 3D inherits it.

## Overview

```text
 parallel read        dual graph          partition         migrate           halo            precompute
 (HDF5, block    ->   (faces matched  ->  (dKaMinPar or ->  (cells, nodes, -> (k layers of  ->  (geometry, TENO
  of cells/rank)       by hashing)         Hilbert curve)    zone tags)        ghost cells)       stencils, BCs)
                                                                                                   |
                         time loop: exchange halo state -> reconstruct -> fluxes -> update owned cells
```

## Components

### 1. Communication layer

A thin `Comm` class wraps MPI (rank, size, `allreduce` min/sum/max, `alltoallv`, neighbor exchange). When MPI is disabled (`Mallard_ENABLE_MPI=OFF`), it is a one-rank stub, so the rest of the code has no `#ifdef`s. One rank per GPU: Kokkos maps devices by local rank (`--kokkos-map-device-id-by=mpi_rank`).

### 2. Mesh input

- **Format:** an HDF5 mesh file with global arrays: node coordinates, cell types and connectivity (CSR), boundary faces with zone ids, and zone names. Each rank reads a contiguous block of cells and the nodes it references, with collective parallel HDF5 I/O. HDF5 is already an optional dependency.
- **Converter:** `mallard-mesh-convert` turns Gmsh (2.2/4.1) files into this format. It is serial, since it is run once per mesh, and later can stream for meshes that don't fit in memory.
- Generated meshes (`cartesian`, `wedge`, ...) are produced directly in blocks per rank.
- Small cases may still read Gmsh on every rank, then partition; this is the default below a size threshold.

### 3. Distributed dual graph

Two cells are adjacent if they share a face. Each rank hashes every face of its cells (sorted global node ids) to an owner rank. One `alltoallv` brings the copies of each face together and pairs them. A second sends each pair back as a graph edge, and boundary faces are matched to zone tags the same way. The result is a distributed CSR graph in ParMETIS layout (`vtxdist`, `xadj`, `adjncy`). Cost is O(faces / ranks) per rank and two all-to-alls.

### 4. Partitioning

A `Partitioner` interface with two backends:

| Backend | When |
|---|---|
| **dKaMinPar** ([KaHIP/KaMinPar](https://github.com/KaHIP/KaMinPar), MIT, C++20; `FetchContent`) | Default when available. Distributed, scales to trillion-edge graphs, guarantees balance. Built with 64-bit ids for hero meshes. New dependency: oneTBB. |
| **Hilbert curve** (built in) | No-dependency fallback; also used everywhere to order cells within a rank for memory locality. |

Vertex weights model cost: a base weight per cell plus a term for the TENO stencil size. Dynamic rebalancing (troubled cells move with shocks) is out of scope; static weights are enough to start. ParMETIS or PT-Scotch can be added behind the same interface if needed.

### 5. Migration

One `alltoallv` sends each cell to its owner: its global id, type, node ids and boundary-zone tags. A second exchange fetches the coordinates of the nodes each rank now needs, from the ranks that read them.

### 6. Halo

- **Local numbering:**
  - Cells: `[owned interior | owned near partition boundary | halo layer 1 | ... | halo layer k]`. Owned cells are in Hilbert order within each group.
  - Faces: faces between owned cells, then faces between an owned and a halo cell, then faces between halo cells that reconstruction needs.
  - Every cell and node keeps its global id.
- **Depth k:** the number of vertex-neighbor layers any reconstruction needs:

  | Reconstruction | k |
  |---|---|
  | FO | 1 |
  | MUSCL | 2 |
  | TENO | stencil radius + 1 |

  TENO's central-stencil gather grows layer by layer until it has enough rows. So the halo is built with a default depth for the TENO order, stencils are computed, and any stencil that touches the halo frontier triggers one more layer for that region. This keeps stencils *identical* to the serial ones, which is what makes results rank-count independent.
- **Exchange plan:** for each neighbor rank, the list of local owned cells to send and of halo cells to receive, sorted by global id on both sides.

### 7. Time stepping

Per right-hand-side evaluation:

1. Post nonblocking receives and sends of the conservative state of halo cells, packed into contiguous device buffers. Sends use device pointers with GPU-aware MPI, or a host staging copy otherwise.
2. **Overlap:** reconstruct and compute fluxes for owned interior cells and their faces while messages are in flight.
3. Wait, unpack, then reconstruct the near-boundary owned cells and halo layer 1, and compute the remaining faces.

Faces between an owned and a halo cell are computed by both ranks. Each keeps only its owned side's contribution, since residuals of halo cells are never used. This costs a few redundant face fluxes but needs no reverse exchange.

Global reductions become `allreduce` calls: time step (min), NaN check, conservation sums, force monitors, average-pressure outlets.

### 8. Boundary conditions and precomputation

These are found during per-rank precomputation on owned plus halo cells, and with a sufficient halo they always resolve to local cells:
- transmissive image faces;
- TENO mirror images;
- Dirichlet face states;
- surface-zone membership.

### 9. Output and restart

- **Solution output:** per-rank VTU pieces plus a `.pvtu` index and the `.pvd` series to start. At scale, HDF5 with an XDMF index (one shared file per snapshot, collective writes).
- **Restart:** HDF5, written by global cell id, so a run can restart on a **different rank count**. This is essential for hero runs, which rarely get the same allocation twice.

## Testing

- **Correctness:** run the solver test suite on 1, 2, 3 and 4 ranks (oversubscribed CPUs, Serial backend) in CI with OpenMPI.
  - Stencils, mirror images and halos must match the serial ones exactly by global id.
  - Solutions must match the single-rank result to round-off.
- **Unit tests:** dual-graph construction, migration and halo construction on small meshes with known answers, including cells whose stencil reaches across several ranks.
- **Restart:** write on 3 ranks, read on 2, and get bit-identical state.
- **Scaling:**
  - Strong and weak scaling on one 8-GPU node, on the 2D Riemann problem and the double Mach reflection.
  - Then across nodes once inter-pod MPI is available on the cluster.

## Milestones

1. **Comm layer and build:** `Mallard_ENABLE_MPI`, one-rank stub, CI with `mpirun`.
2. **Correct multi-rank runs at small scale:** global Gmsh read on every rank, Hilbert partition, halo, exchange, reductions. Rank-count-independence tests for FO, MUSCL, TENO and viscous fluxes.
3. **Output and restart:** `.pvtu` output; HDF5 restart independent of the partition.
4. **Scalable setup:** HDF5 mesh format and converter, distributed read, distributed dual graph, dKaMinPar, migration.
5. **Performance:** communication/computation overlap, GPU-aware MPI, single-node 8-GPU scaling study, then launch-overhead work (CUDA graphs) where it matters at small per-rank sizes.
6. **Multi-node:** runs across nodes; HDF5/XDMF solution output.

## Open questions

1. **Halo depth vs. a second exchange.** The design reconstructs halo layer 1 redundantly, so one exchange per stage suffices. The alternative exchanges reconstructed face states at partition faces, which saves one halo layer and that layer's TENO work but adds a second message round per stage. At thousands of ranks, latency favors one exchange.
2. **NCCL instead of MPI for halo exchange on NVIDIA GPUs.** MPI stays the baseline. NCCL could be an optional backend behind `Comm` if GPU-aware MPI underperforms on the target cluster.
3. **Dynamic load balancing.** Troubled-cell cost moves with shocks. Repartitioning every N steps is possible later with the same migration code.
