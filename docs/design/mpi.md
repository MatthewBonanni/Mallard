# Design: distributed memory (MPI)

Status: accepted design (reviewed in #32), implemented through milestone 5 (see Milestones). Tracks issue #2.

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

The per-stage halo exchange and the time-step `allreduce` go through a `HaloExchange` interface with two backends:

- **MPI** (default): nonblocking point-to-point; device buffers passed directly with GPU-aware MPI, or staged through host memory otherwise.
- **NCCL** (optional, `Mallard_ENABLE_NCCL`): `ncclSend`/`ncclRecv` inside a group and `ncclAllReduce`, enqueued on the Kokkos execution stream, so the exchange is stream-ordered with the kernels (no host synchronization) and can be captured in a CUDA graph. The NCCL communicator is created from the MPI communicator. **Deferred** (milestone 5 measurements): with GPU-aware MPI over NVLink, or over InfiniBand with GPUDirect RDMA (3 us, 26 GB/s between GPUs on different nodes), an exchange costs about 0.1 ms per stage and is overlapped with the interior reconstruction, and a step launches about 30 kernels, so neither stream ordering nor CUDA graphs would recover much; the remaining strong-scaling loss is load imbalance (below).

Setup-time communication (partitioning, migration, I/O) always uses MPI.

### 2. Mesh input

- **Format:** an HDF5 mesh file with global arrays, global ids being row indices: node coordinates, cell connectivity (CSR of node ids; the cell type follows from the node count), boundary faces (CSR) with zone ids, and zone names (layout in `docs/input.md`). Each rank reads a contiguous block of cells, of nodes and of boundary faces, with collective parallel HDF5 I/O (independent reads when HDF5 is not parallel). Node coordinates a rank needs but did not read are fetched later from the rank that read them (section 5).
- **Converter:** `mallard-mesh-convert` turns Gmsh (2.2/4.1) files into this format, or writes a generated mesh described by an input file in parallel. A Gmsh file is read whole, since it is converted once per mesh; later the converter can stream meshes that don't fit in memory.
- Generated meshes (`cartesian`, `wedge`, ...) are produced directly in blocks per rank, numbered as the serial generators number them.
- A Gmsh file is read whole by every rank, which keeps only its block; meshes too large for that are converted first. Every source then goes through the same distributed path (`DistributedMesh`); only single-rank runs build the mesh directly.

### 3. Distributed dual graph

Two cells are adjacent if they share a face. Each rank hashes every face of its cells (sorted global node ids) to an owner rank. One `alltoallv` brings the copies of each face together and pairs them; the boundary faces of the mesh file go to the same ranks by the same hash. A second sends each pair back as a graph edge, and each unpaired face back as a boundary face with its zone (the first boundary face in global order that matches it, else `unassigned`). Faces shared by more than two cells, and boundary faces that are interior or not cell faces, are errors on every rank. The result is a distributed CSR graph in ParMETIS layout (`vtxdist`, `xadj`, `adjncy`). Cost is O(faces / ranks) per rank and two all-to-alls.

### 4. Partitioning

A `Partitioner` interface with two backends:

| Backend | When |
|---|---|
| **dKaMinPar** ([KaHIP/KaMinPar](https://github.com/KaHIP/KaMinPar), MIT, C++20; `FetchContent`) | Default when available. Distributed, scales to trillion-edge graphs, guarantees balance. Built with 64-bit ids for hero meshes. New dependency: oneTBB. |
| **Hilbert curve** (built in) | No-dependency fallback; also used everywhere to order cells within a rank for memory locality. |

The Hilbert backend sorts cells by the curve key of their vertex average (not their centroid, which needs the geometry a rank does not have yet) with a distributed sample sort, so its partitions differ from a global-mesh Hilbert partition but are equally valid; dKaMinPar works directly on the distributed dual graph (section 3).

Vertex weights model cost: a base weight per cell plus a term for the TENO stencil size. ParMETIS or PT-Scotch can be added behind the same interface if needed.

### 5. Migration

One `alltoallv` sends each cell to its owner: its global id, owner, node ids and boundary faces (local face index and zone). A second builds a **node directory**: the rank that read each node records the cells using it and their owners. Every node shared by several owners then sends each of them the cells there it does not own, which is halo layer 1. The coordinates of the nodes a rank needs are fetched from the ranks that read them once its halo is complete. The cells' blocks stay on the ranks that read them until setup ends, and serve the halo search.

### 6. Halo

- **Local numbering:**
  - Cells: `[owned interior | owned near partition boundary | halo layer 1 | ... | halo layer k]`. Owned cells are in Hilbert order within each group. (Implemented so far: owned cells, then each halo layer, each in global id order.)
  - Faces: faces between owned cells, then faces between an owned and a halo cell, then faces between halo cells that reconstruction needs.
  - Every cell and node keeps its global id.
- **Depth k:** the number of vertex-neighbor layers any reconstruction needs:

  | Reconstruction | k |
  |---|---|
  | FO | 1 |
  | MUSCL | 2 |
  | TENO | stencil radius + 1 |

  TENO's central-stencil gather grows layer by layer until it has enough rows. So the halo is built with a default depth, stencils are computed, and if any stencil reaches the halo frontier the halo is deepened on every rank and the setup after the mesh repeats. This keeps stencils *identical* to the serial ones, which is what makes results rank-count independent.
- **Construction:** a distributed breadth-first search over vertex neighbors. Layer 1 comes from the node directory (section 5); for each further layer, a rank asks the directory for the cells around the nodes of its previous layer that it has not asked about, then fetches the new cells from the ranks whose blocks hold them. Deepening the halo continues the search from the existing layers rather than starting over.
- **Exchange plan:** for each neighbor rank, the list of local owned cells to send and of halo cells to receive, sorted by global id on both sides.

### 7. Time stepping

Per right-hand-side evaluation:

1. Post nonblocking receives and sends of the conservative state of halo cells, packed into contiguous device buffers. Sends use device pointers with GPU-aware MPI, or a host staging copy otherwise.
2. **Overlap:** reconstruct the owned cells whose stencils and face neighbors are all owned (`FaceReconstruction::cells_independent_of_halo`, exact from the stencils rather than a distance bound) while messages are in flight, on a second execution space instance.
3. Wait, unpack, then reconstruct the near-boundary owned cells and halo layer 1 on the default instance, where they fill the device as the first instance drains. Then come the steps that need every cell: TENO's troubled-cell passes and the fluxes of the faces of owned cells (faces between two halo cells are skipped).

Implemented for TENO, the reconstruction that needs deep halos; FO and MUSCL exchange first and then reconstruct. The first stage of a step reuses the halo that the time-step computation has just filled, so an SSPRK3 step exchanges three times, not four.

Halo layer 1 is reconstructed redundantly so that a single exchange per stage suffices. The alternative, exchanging reconstructed face states at partition faces, would save that layer's work (a surface-to-volume fraction of the TENO cost) but adds a second, dependent message round per stage; at thousands of ranks the extra latency costs more than the redundant work. The choice is contained in the exchange plan and can be revisited with measurements.

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

### 10. Dynamic load balancing

Status: design (milestone 7). Default off until measured.

**Why.** TENO's troubled cells (near shocks) cost several times a smooth cell, and shocks cross a few ranks and move. On 8-16 A100s the troubled pass takes about 370 us per stage on the ranks a shock crosses and 20 us on the others, which is the remaining strong-scaling loss (milestone 5). Chemistry will add larger per-cell cost variation; it balances that by moving chemistry states between ranks without touching the mesh (`chemistry.md`, decision 9). This section moves the mesh, and takes chemistry's long-term cost as one more weight term.

**Invariant.** Results are bitwise identical with and without rebalancing, as they are on any rank count today. Nothing a rebalance changes enters the arithmetic: stencils, pseudo-inverses and face orders are functions of global ids, reductions are exact (min) or ordered by global keys, and state and per-cell data move as raw bits. Only *when* a rebalance happens depends on timings.

#### 10.1 Cost model

- **Per-cell cost** in smooth-cell units: `w_c = 1 + (kappa - 1) f_c`, where `f_c` is the fraction of the window's steps in which cell c was troubled. After each step a kernel over owned cells adds `sigma_c >= threshold` to a per-cell counter (one read per cell per step).
- **Weights** are integers, `W_c = round(16 w_c)`, so the partitioners see exactly the same input on every run with the same counts. Chemistry adds its measured per-cell cost here later.
- **Measured per-rank busy time** `B_r`: stepping time minus the host time blocked in the halo `MPI_Waitall` and in the time-step `allreduce`. `HaloExchange::start` already fences the device, so time outside those waits is the rank's own work.
- **kappa** (troubled / smooth cost ratio) is fitted at each check by least squares over ranks, `B_r = a N_r + b T_r` with `N_r` reconstructed cell-steps and `T_r` troubled cell-steps, `kappa = (a + b) / a`, clamped to [1, 100]. When troubled counts don't vary enough between ranks to separate a and b, the previous value is kept, starting from `troubled_cost` (input; default from the A100 measurements).

#### 10.2 Trigger

Every `rebalance_interval` steps (default 100) the ranks allreduce `(B_r, N_r, T_r)`:

- imbalance `I = max_r B_r / mean_r B_r`;
- predicted saving `S = (max_r B_r - 1.05 mean_r B_r) / interval` per step, over a horizon of the remaining steps (from the stop condition), capped at 10 intervals;
- predicted cost `C`: the last rebalance's wall time. Before the first one, `C = t_mesh + V / BW`: `t_mesh` is the setup's mesh phase (read, partition, migration, halo), which a rebalance repeats; `V` is the migration volume, the TENO records (section 10.4) and state of the cells expected to move, taken as the imbalance fraction `(I - 1) / I` of the owned cells; `BW` is the bandwidth the setup's `alltoallv` calls achieved (bytes and time are counted in `comm::exchange`).

Rebalance when `I > rebalance_threshold` (default 1.10), `S > 2 C`, and the rebalancing time so far plus `C` stays under `rebalance_max_cost` (default 1%) of the projected wall time of the run. The run summary reports rebalances, their time and the imbalance before and after.

#### 10.3 Repartitioning

The `DistributedMesh` (the blocks read at startup, the node directory and the dual graph) stays alive when rebalancing is enabled: it is host memory of O(cells / ranks), about 200 B per cell against about 10 KB per cell of TENO data. Owners send `W_c` to the block ranks by global id, and the partitioner returns new owners per block cell as at startup.

- **Hilbert** (built in, the incremental option): the sample sort is rerun with the weights attached, and the curve is split at weighted prefix sums, `part = floor((prefix_c + W_c / 2) p / W_total)`. Parts stay contiguous curve intervals, so only cells near the moved split points migrate, about the imbalance fraction of the cells.
- **dKaMinPar**: node weights through `copy_graph`, partitioned from scratch (the distributed API has no refinement of a given partition), then parts are relabeled to maximize overlap with the current owners: each rank counts the weight it keeps per (old, new) pair, the sparse counts are gathered and matched greedily by decreasing overlap. Migration is larger than Hilbert's, the edge cut smaller; the measurements pick the default.
- Diffusive moves on the graph (overloaded ranks shedding boundary cells to neighbors) would migrate the least but need their own quality control; deferred until measurements show the two options above aren't enough.

#### 10.4 Migration and rebuild

1. Old owners keep a host copy of the solution of their owned cells by global id.
2. `DistributedMesh::distribute(new owners)` and `build_local_mesh(halo_layers)` rebuild the local mesh, halo and exchange plan, as at startup. The halo depth stays that of startup: it depends on each cell's gather depth, which doesn't depend on the partition, so the migrated gather depths only need checking.
3. The solution goes from old to new owners by global id (one `alltoallv`).
4. **TENO data is migrated, not recomputed.** Setup costs about 330 us per reconstructed cell on one Mac core (2D TENO5 on quadrilaterals: 14 s for 42k cells per rank, twice that when the halo must deepen), while a step costs 5 us per cell on a CPU core and about 30 ns on an A100. Recomputing 125k cells per GPU would cost about 10,000 steps, which no imbalance repays. Instead every reconstructed cell has a record keyed by global id: its stencils as global cell ids, mirror faces as (global cell, local face), pseudo-inverses, smoothness matrix, basis moments, scale and gather depth. A rank keeps the records of the cells it still reconstructs and fetches the others from their previous owner (through the block rank, which knows it); every cell was owned, so reconstructed, by someone. Record sizes, mostly pseudo-inverses: 2D TENO5 (14 coefficients, about 35 large and 4 x 11 sector stencil cells) about 4 KB large + 2 KB sectors + 1 KB smoothness matrix and ids, so about 7-10 KB per cell; 3D TENO5 (55 coefficients, about 190 stencil cells) about 85 KB per cell. Moving 10% of 125k cells per GPU in 2D is about 100-125 MB per rank: about 5 ms at the 26 GB/s measured between A100s over InfiniBand (milestone 5), plus device-host copies at PCIe speed on each side when MPI isn't GPU-aware or records are packed on the host, so about 15 ms. In 3D the same move is about 1 GB, about 0.1 s. Both are small next to the mesh rebuild (0.13 s for 160k quadrilaterals on 4 Mac ranks) and far below recomputation. This shares the per-cell record format with the stencil cache (#89).
5. Partition-dependent solver state is rebuilt from the new mesh: boundary data (Dirichlet face lists and average-pressure outlets are cleared first), the RHS split, work arrays, halo buffers, output pieces (writers keep their counters), force-monitor faces. Old device data is freed before the new is allocated, so peak device memory stays that of one partition.

#### 10.5 Restart and output

Restart files are keyed by global id and don't change. A restarted run starts from the static partition and rebalances after its first window; storing `W_c` in the restart file would let it start balanced (later). `.pvtu` pieces follow the current partition.

#### 10.6 Tests

- Weighted partitioners: each part's weight within the tolerance of `W_total / p`, on weights concentrated in a corner (Hilbert, dKaMinPar); relabeling keeps an unchanged partition unchanged.
- A run with forced rebalances (synthetic weights at fixed steps) is bitwise identical to one without, on 2, 3 and 4 ranks, for TENO (with mirror faces and periodic seams), MUSCL and Navier-Stokes with Dirichlet and average-pressure boundaries.
- Migrated TENO records equal recomputed ones bitwise by global id.

### 11. Communication patterns at scale

Setup exchanges are dense `alltoallv` calls, whose count arrays alone are O(ranks) per rank and whose latency grows with the rank count. The face matching and migration genuinely talk to many ranks, but halo growth and node-coordinate fetches have sparse patterns (a rank's partition neighbors and the few ranks whose blocks hold its cells). Beyond about 10k ranks these should move to MPI neighborhood collectives or a sparse NBX exchange (nonblocking sends, `MPI_Ibarrier` to detect completion); planned with milestone 6.

### 12. Periodic boundaries

See `periodic.md`. Periodic node classes are currently found by gathering the
periodic zones on every rank, O(N^(2/3)) data per rank; the scalable
alternative (hashing snapped node coordinates, as for faces) is described
there.

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
4. **Scalable setup:** HDF5 mesh format and converter, distributed read, distributed dual graph, dKaMinPar, migration. Done: with generated meshes or HDF5 mesh files no rank holds the global mesh in a distributed run (Gmsh files are still read whole by every rank), and restart files are read by global id, each rank reading only its cells. Setup only (one FO step), CPU, peak memory per rank: 4.1M hexahedra took 6.0 GiB on every rank count before (each rank built the global mesh) and now 3.1 / 1.7 / 0.9 / 0.5 GiB on 2 / 4 / 8 / 16 ranks (setup 40 s -> 19 / 11 / 6.4 / 4.5 s); 16M quadrilaterals 8.7-11 GiB per rank before, now 6.5 / 3.3 / 1.7 / 0.9 GiB; 64M quadrilaterals set up on 16 ranks in 3.3 GiB each.
5. **Performance:** communication/computation overlap, GPU-aware MPI, the NCCL backend, single-node 8-GPU scaling study, then launch-overhead work (CUDA graphs, which the stream-ordered NCCL exchange allows) where it matters at small per-rank sizes. Done except NCCL and CUDA graphs, deferred (section 1).
   - **Profile.** Profiled at 8 GPUs on 1M cells. The TENO troubled-cell pass ran one heavy thread per cell, and a rank holds fewer troubled cells than one wave of threads. The pass therefore took one thread's latency (0.6-0.9 ms per stage) on ranks crossed by shocks, and about nothing on the others. The exchanges cost about 0.1 ms per stage each.
   - **Fixes.**
     - The troubled passes are split per (cell, sector), per (cell, face, characteristic variable), per (cell, face) and per cell, with the same arithmetic.
     - The exchange is overlapped with the interior reconstruction (section 7).
     - Flux and RHS work is limited to owned cells.
     - The first-stage exchange is reused from the time-step computation.
     - Results stay bitwise identical.
   - **Scaling.** 2D Riemann problem (configuration 3), TENO5 on quadrilaterals, HLLC, SSPRK3, double precision, A100-80GB GPUs. Times are seconds per 50 steps, with GPU-aware MPI and the Hilbert partition. Up to 8 GPUs share one node over NVLink. The 16-GPU runs use 4 GPUs on each of 4 nodes, over HDR InfiniBand with GPUDirect RDMA.

     | cells | 1 GPU | 2 | 4 | 8 | 16 | efficiency at 8 / 16 |
     |---|---|---|---|---|---|---|
     | 1M | 0.938 | 0.553 | 0.311 | 0.190 | 0.152 | 62% / 38% |
     | 4M | 3.50 | 1.93 | 1.00 | 0.534 | 0.316 | 82% / 69% |
     | 16M | | | | 1.93 | 1.01 | 95% from 8 to 16 |
     | 1M per GPU (weak) | 0.938 | 0.999 | 1.000 | 1.005 | 1.013 | 93% / 93% |

     - Before this milestone, 1M cells on 8 GPUs took 0.287 s (1.038 s on 1 GPU).
     - The 8 GPUs give the same times on one node as on 2 nodes x 4.
     - Host-staged MPI is 15-25% slower; the overlap recovers about 10% of it.
     - dKaMinPar partitions give the same times within a few percent, and 0.141 s on 1M cells at 16 GPUs.
     - What remains at small per-GPU sizes is the cost of troubled cells concentrated on the ranks crossed by shocks, and of the smooth TENO pass's last partial wave of threads. Dynamic load balancing (section 10) addresses the first.
6. **Multi-node:** runs across nodes; HDF5/XDMF solution output. Runs across nodes work, with GPUDirect RDMA (milestone 5 table). HDF5/XDMF output is still to do.
7. **Dynamic load balancing** (section 10): weighted partitioners; runtime migration and rebuild with TENO records migrated; cost measurement and trigger (`[parallel] rebalance`, off by default); bitwise tests; time to solution on 8 and 16 A100s for moving-shock cases.

## Decisions from review

1. One halo exchange per stage with a redundant halo-layer-1 reconstruction (section 7).
2. Optional NCCL backend for the halo exchange and reductions (section 1).
3. Dynamic load balancing deferred, with the setup structured to allow it (section 10); designed for milestone 7.
