/**
 * @file teno.cpp
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief TENO-E reconstruction implementation.
 * @version 0.2
 * @date 2026-10-02
 *
 * @copyright Copyright (c) 2026 Matthew Bonanni
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "comm.h"
#include "face_reconstruction.h"

#include "input.h"
#include "teno.h"

namespace {

/**
 * @brief Gauss-Legendre nodes and weights on [-1, 1] (Newton iteration).
 */
void gauss_legendre(int n, std::vector<double> & x, std::vector<double> & w) {
    x.resize(n);
    w.resize(n);
    for (int i = 0; i < n; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double dp = 0.0;
        for (int it = 0; it < 100; it++) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; k++) {
                const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            if (n == 1) { p1 = z; p0 = 1.0; }
            dp = n * (z * p1 - p0) / (z * z - 1.0);
            const double dz = p1 / dp;
            z -= dz;
            if (std::abs(dz) < 1e-15) break;
        }
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

/**
 * @brief Collapsed (Duffy) Gauss quadrature on the reference triangle
 *        (0,0), (1,0), (0,1); weights sum to 1/2. Exact for total degree 2n-2.
 */
struct TriangleRule {
    std::vector<double> xi, eta, w;
    explicit TriangleRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                const double s = 0.5 * (g[i] + 1.0);
                const double t = 0.5 * (g[j] + 1.0);
                xi.push_back(s);
                eta.push_back(t * (1.0 - s));
                w.push_back(0.25 * gw[i] * gw[j] * (1.0 - s));
            }
        }
    }
};

/**
 * @brief Integrate f(xi, eta) over a polygon (given as vertices in the scaled
 *        frame) by fan triangulation.
 */
template <typename F>
void integrate_polygon(const std::vector<double> & px, const std::vector<double> & py,
                       const TriangleRule & rule, F && f) {
    for (size_t k = 1; k + 1 < px.size(); k++) {
        const double ax = px[k] - px[0], ay = py[k] - py[0];
        const double bx = px[k + 1] - px[0], by = py[k + 1] - py[0];
        const double det = std::abs(ax * by - ay * bx);
        for (size_t q = 0; q < rule.w.size(); q++) {
            const double x = px[0] + rule.xi[q] * ax + rule.eta[q] * bx;
            const double y = py[0] + rule.xi[q] * ay + rule.eta[q] * by;
            f(x, y, rule.w[q] * det);
        }
    }
}

/**
 * @brief Least-squares pseudo-inverse P = R^-1 Q^T (n x m) of a full-column-rank
 *        m x n matrix A (row-major) via Householder QR.
 * @param max_condition Largest accepted ratio of the diagonal entries of R
 *        after column equilibration (an estimate of the condition number).
 * @return False if A is too ill conditioned.
 */
bool pseudo_inverse(std::vector<double> A, int m, int n, std::vector<double> & P, double max_condition) {
    // Equilibrate columns so the rank test is independent of monomial scaling
    std::vector<double> col_scale(n, 0.0);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) col_scale[j] += A[i * n + j] * A[i * n + j];
        col_scale[j] = std::sqrt(col_scale[j]);
        if (col_scale[j] == 0.0) return false;
    }
    if constexpr (N_DIM == 3) {
        // A column of round-off (e.g. the xy monomial over a stencil whose centroids
        // all lie on axis planes) would pass the rank test once equilibrated
        const double largest = *std::max_element(col_scale.begin(), col_scale.end());
        for (int j = 0; j < n; j++) {
            if (col_scale[j] < 1e-10 * largest) return false;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) A[i * n + j] /= col_scale[j];
    }
    std::vector<double> QT(m * m, 0.0);
    for (int i = 0; i < m; i++) QT[i * m + i] = 1.0;
    std::vector<double> v(m);
    double max_diag = 0.0;
    for (int k = 0; k < n; k++) {
        double norm = 0.0;
        for (int i = k; i < m; i++) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm == 0.0) return false;
        const double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        for (int i = 0; i < m; i++) v[i] = (i < k) ? 0.0 : A[i * n + k];
        v[k] -= alpha;
        double vnorm2 = 0.0;
        for (int i = k; i < m; i++) vnorm2 += v[i] * v[i];
        if (vnorm2 == 0.0) continue;
        for (int j = 0; j < n; j++) {
            double d = 0.0;
            for (int i = k; i < m; i++) d += v[i] * A[i * n + j];
            d *= 2.0 / vnorm2;
            for (int i = k; i < m; i++) A[i * n + j] -= d * v[i];
        }
        for (int j = 0; j < m; j++) {
            double d = 0.0;
            for (int i = k; i < m; i++) d += v[i] * QT[i * m + j];
            d *= 2.0 / vnorm2;
            for (int i = k; i < m; i++) QT[i * m + j] -= d * v[i];
        }
        max_diag = std::max(max_diag, std::abs(A[k * n + k]));
    }
    for (int k = 0; k < n; k++) {
        if (std::abs(A[k * n + k]) * max_condition < max_diag) return false;
    }
    P.assign(n * m, 0.0);
    for (int j = 0; j < m; j++) {
        for (int k = n - 1; k >= 0; k--) {
            double s = QT[k * m + j];
            for (int l = k + 1; l < n; l++) s -= A[k * n + l] * P[l * m + j];
            P[k * m + j] = s / A[k * n + k];
        }
    }
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < m; j++) P[k * m + j] /= col_scale[k];
    }
    return true;
}

// Cells per batch of the precomputation: one chunk of the packed stencils
constexpr uint32_t CHUNK_CELLS = 1u << teno::CHUNK_SHIFT;

/** @brief Precomputed data of one reconstructed cell, before packing. */
struct CellTables {
    uint8_t gather_depth = 0;
    rtype scale = 0.0;
    std::vector<rtype> basis_mean;                  // (l)
    std::vector<rtype> si;                          // upper triangle of (l, m)
    std::vector<int32_t> large_cells, large_faces;  // (s)
    std::vector<rtype> large_pinv;                  // (s, l)
    std::array<uint16_t, teno::MAX_FACES> small_size = {};
    std::vector<int32_t> small_cells, small_faces;  // the faces' sector stencils one after another
    std::vector<rtype> small_pinv;                  // (s, l)
};

using IndexRow = std::vector<int32_t> CellTables::*;
using ValueRow = std::vector<rtype> CellTables::*;

template <typename T>
using HostUnmanaged = Kokkos::View<T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/** @brief Rows [c0, c0 + h.extent(0)) of a per-cell device array, from the host. */
template <typename View>
void upload_rows(const View & dev, const uint32_t c0, const typename View::host_mirror_type & h) {
    if (h.extent(0) == 0) return;
    auto tmp = Kokkos::create_mirror_view_and_copy(typename View::memory_space(), h);
    const std::pair<size_t, size_t> rows(c0, c0 + h.extent(0));
    if constexpr (View::rank() == 1) {
        Kokkos::deep_copy(Kokkos::subview(dev, rows), tmp);
    } else {
        Kokkos::deep_copy(Kokkos::subview(dev, rows, Kokkos::ALL()), tmp);
    }
}

/** @brief Host copy of rows [c0, c1) of a per-cell device array. */
template <typename View>
typename View::host_mirror_type download_rows(const View & dev, const uint32_t c0, const uint32_t c1) {
    using Host = typename View::host_mirror_type;
    const std::pair<size_t, size_t> rows(c0, c1);
    Host h;
    if constexpr (View::rank() == 1) {
        h = Host("teno_rows", c1 - c0);
    } else {
        h = Host("teno_rows", c1 - c0, dev.extent(1));
    }
    if (c1 == c0) return h;
    auto tmp = Kokkos::create_mirror_view(typename View::memory_space(), h);
    if constexpr (View::rank() == 1) {
        Kokkos::deep_copy(tmp, Kokkos::subview(dev, rows));
    } else {
        Kokkos::deep_copy(tmp, Kokkos::subview(dev, rows, Kokkos::ALL()));
    }
    Kokkos::deep_copy(h, tmp);
    return h;
}

/**
 * @brief Moves one stencil family (large or sector) of CellTables into a
 *        teno::PackedStencils, one chunk of cells at a time, and back.
 */
class PackedRows {
    public:
        PackedRows(uint8_t shift, uint8_t width, IndexRow cells, IndexRow faces, ValueRow pinv)
            : shift(shift), width(width), cells(cells), faces(faces), pinv(pinv) {}

        /**
         * @brief Store the stencils of chunk tables (cells c0 onward, c0 a
         *        multiple of CHUNK_CELLS, after the previous chunk) in device
         *        memory appended to storage.
         */
        void add(const uint32_t c0, const std::vector<CellTables> & tables,
                 std::vector<Kokkos::View<char *>> & storage) {
            if (c0 != storage.size() * CHUNK_CELLS || c0 >> shift != slice_start.size()) {
                throw std::logic_error("TENO: stencil chunks must be stored in order.");
            }
            const uint32_t slice = 1u << shift;
            const uint32_t n = tables.size();
            uint32_t n_slots = 0;
            for (uint32_t a = 0; a < n; a += slice) {
                slice_start.push_back(n_slots);
                size_t largest = 0;
                for (uint32_t c = a; c < std::min(n, a + slice); c++) largest = std::max(largest, (tables[c].*cells).size());
                n_slots += largest;
            }
            const Layout layout(n_slots << shift, width);
            std::vector<char> buf(layout.bytes, 0);
            rtype * h_pinv = reinterpret_cast<rtype *>(buf.data());
            int32_t * h_cells = reinterpret_cast<int32_t *>(buf.data() + layout.cells);
            int32_t * h_faces = reinterpret_cast<int32_t *>(buf.data() + layout.faces);
            std::fill(h_cells, h_cells + layout.n_slots, -1);
            std::fill(h_faces, h_faces + layout.n_slots, -1);
            for (uint32_t c = 0; c < n; c++) {
                const CellTables & t = tables[c];
                const uint32_t start = slice_start[(c0 + c) >> shift];
                const uint32_t lane = (c0 + c) & (slice - 1);
                for (size_t s = 0; s < (t.*cells).size(); s++) {
                    h_cells[((start + s) << shift) + lane] = (t.*cells)[s];
                    h_faces[((start + s) << shift) + lane] = (t.*faces)[s];
                    for (uint32_t l = 0; l < width; l++) {
                        h_pinv[(((start + s) * width + l) << shift) + lane] = (t.*pinv)[s * width + l];
                    }
                }
            }
            Kokkos::View<char *> dev(Kokkos::view_alloc(Kokkos::WithoutInitializing, "teno_stencils"), buf.size());
            Kokkos::deep_copy(dev, HostUnmanaged<char>(buf.data(), buf.size()));
            chunks.push_back({reinterpret_cast<rtype *>(dev.data()), reinterpret_cast<int32_t *>(dev.data() + layout.cells),
                              reinterpret_cast<int32_t *>(dev.data() + layout.faces)});
            storage.push_back(dev);
        }

        /** @brief The device addressing of the stored chunks. */
        teno::PackedStencils finish(const std::string & label) const {
            teno::PackedStencils out;
            out.shift = shift;
            out.width = width;
            out.slice_start = Kokkos::View<uint32_t *>(label + "_slice_start", slice_start.size());
            Kokkos::deep_copy(out.slice_start, HostUnmanaged<const uint32_t>(slice_start.data(), slice_start.size()));
            out.chunks = Kokkos::View<teno::PackedStencils::Chunk *>(label + "_chunks", chunks.size());
            Kokkos::deep_copy(out.chunks,
                              HostUnmanaged<const teno::PackedStencils::Chunk>(chunks.data(), chunks.size()));
            return out;
        }

        /**
         * @brief Stencils of chunk tables (cells c0 onward) from the device;
         *        sizes: slots of each cell.
         */
        void download(const teno::PackedStencils & packed, const std::vector<uint32_t> & h_slice_start,
                      const Kokkos::View<char *> & storage, const uint32_t c0, const std::vector<uint16_t> & sizes,
                      std::vector<CellTables> & tables) const {
            const Layout layout(storage.extent(0) / (packed.width * sizeof(rtype) + 2 * sizeof(int32_t)), packed.width);
            std::vector<char> buf(layout.bytes);
            Kokkos::deep_copy(HostUnmanaged<char>(buf.data(), buf.size()), storage);
            const rtype * h_pinv = reinterpret_cast<const rtype *>(buf.data());
            const int32_t * h_cells = reinterpret_cast<const int32_t *>(buf.data() + layout.cells);
            const int32_t * h_faces = reinterpret_cast<const int32_t *>(buf.data() + layout.faces);
            const uint32_t slice = 1u << packed.shift;
            for (uint32_t c = 0; c < tables.size(); c++) {
                CellTables & t = tables[c];
                const uint32_t start = h_slice_start[(c0 + c) >> packed.shift];
                const uint32_t lane = (c0 + c) & (slice - 1);
                (t.*cells).resize(sizes[c]);
                (t.*faces).resize(sizes[c]);
                (t.*pinv).resize(sizes[c] * packed.width);
                for (uint16_t s = 0; s < sizes[c]; s++) {
                    (t.*cells)[s] = h_cells[((start + s) << packed.shift) + lane];
                    (t.*faces)[s] = h_faces[((start + s) << packed.shift) + lane];
                    for (uint32_t l = 0; l < packed.width; l++) {
                        (t.*pinv)[s * packed.width + l] = h_pinv[(((start + s) * packed.width + l) << packed.shift) + lane];
                    }
                }
            }
        }

    private:
        // Byte layout of a chunk: pseudo-inverse entries, then stencil cells, then mirror faces
        struct Layout {
            size_t n_slots, cells, faces, bytes;
            Layout(size_t n_slots, uint8_t width)
                : n_slots(n_slots),
                  cells(n_slots * width * sizeof(rtype)),
                  faces(cells + n_slots * sizeof(int32_t)),
                  bytes(faces + n_slots * sizeof(int32_t)) {}
        };
        uint8_t shift;
        uint8_t width;
        IndexRow cells, faces;
        ValueRow pinv;
        std::vector<uint32_t> slice_start;
        std::vector<teno::PackedStencils::Chunk> chunks;
};

PackedRows large_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, scheme.n_dof_large, &CellTables::large_cells, &CellTables::large_faces,
                      &CellTables::large_pinv);
}

PackedRows small_rows(const TENO & scheme) {
    return PackedRows(scheme.slice_shift, teno::NK_SMALL, &CellTables::small_cells, &CellTables::small_faces,
                      &CellTables::small_pinv);
}

/** @brief Moves per-cell tables, chunk by chunk, into TENO's device arrays. */
class TableBuilder {
    public:
        TableBuilder(TENO & scheme, const uint32_t n_cells)
            : scheme(scheme), large(large_rows(scheme)), small(small_rows(scheme)) {
            const uint8_t nk = scheme.n_dof_large;
            scheme.scale = Kokkos::View<rtype *>("teno_scale", n_cells);
            scheme.basis_mean = Kokkos::View<rtype **>("teno_basis_mean", n_cells, nk);
            scheme.si_matrix = Kokkos::View<rtype **>("teno_si_matrix", n_cells, nk * (nk + 1) / 2);
            scheme.stencil_large_size = Kokkos::View<uint16_t *>("teno_stencil_large_size", n_cells);
            scheme.stencil_small_size = Kokkos::View<uint16_t **>("teno_stencil_small_size", n_cells, teno::MAX_FACES);
            scheme.stencil_large_storage.clear();
            scheme.stencil_small_storage.clear();
            scheme.gather_depth.assign(n_cells, 0);
        }

        /** @brief Tables of cells [c0, c0 + tables.size()), the chunk after the previous one. */
        void add(const uint32_t c0, const std::vector<CellTables> & tables) {
            const uint32_t n = tables.size();
            Kokkos::View<rtype *>::host_mirror_type h_scale("teno_scale_rows", n);
            Kokkos::View<rtype **>::host_mirror_type h_mean("teno_basis_mean_rows", n, scheme.basis_mean.extent(1));
            Kokkos::View<rtype **>::host_mirror_type h_si("teno_si_rows", n, scheme.si_matrix.extent(1));
            Kokkos::View<uint16_t *>::host_mirror_type h_large_size("teno_large_size_rows", n);
            Kokkos::View<uint16_t **>::host_mirror_type h_small_size("teno_small_size_rows", n, teno::MAX_FACES);
            for (uint32_t c = 0; c < n; c++) {
                const CellTables & t = tables[c];
                scheme.gather_depth[c0 + c] = t.gather_depth;
                h_scale(c) = t.scale;
                for (size_t l = 0; l < t.basis_mean.size(); l++) h_mean(c, l) = t.basis_mean[l];
                for (size_t k = 0; k < t.si.size(); k++) h_si(c, k) = t.si[k];
                h_large_size(c) = t.large_cells.size();
                for (uint8_t k = 0; k < teno::MAX_FACES; k++) h_small_size(c, k) = t.small_size[k];
            }
            upload_rows(scheme.scale, c0, h_scale);
            upload_rows(scheme.basis_mean, c0, h_mean);
            upload_rows(scheme.si_matrix, c0, h_si);
            upload_rows(scheme.stencil_large_size, c0, h_large_size);
            upload_rows(scheme.stencil_small_size, c0, h_small_size);
            large.add(c0, tables, scheme.stencil_large_storage);
            small.add(c0, tables, scheme.stencil_small_storage);
        }

        void finish() {
            scheme.stencil_large = large.finish("teno_stencil_large");
            scheme.stencil_small = small.finish("teno_stencil_small");
        }

    private:
        TENO & scheme;
        PackedRows large, small;
};

/** @brief Host copy of the slice starts of packed stencils. */
std::vector<uint32_t> host_slice_start(const teno::PackedStencils & packed) {
    std::vector<uint32_t> h(packed.slice_start.extent(0));
    Kokkos::deep_copy(HostUnmanaged<uint32_t>(h.data(), h.size()), packed.slice_start);
    return h;
}

/** @brief Tables of the chunk of reconstructed cells starting at c0 from TENO's device arrays. */
void download_tables(const TENO & scheme, const std::vector<uint32_t> & large_slices,
                     const std::vector<uint32_t> & small_slices, const uint32_t c0, std::vector<CellTables> & tables) {
    const uint32_t c1 = c0 + tables.size();
    auto h_scale = download_rows(scheme.scale, c0, c1);
    auto h_mean = download_rows(scheme.basis_mean, c0, c1);
    auto h_si = download_rows(scheme.si_matrix, c0, c1);
    auto h_large_size = download_rows(scheme.stencil_large_size, c0, c1);
    auto h_small_size = download_rows(scheme.stencil_small_size, c0, c1);
    std::vector<uint16_t> n_large(tables.size()), n_small(tables.size(), 0);
    for (uint32_t c = 0; c < tables.size(); c++) {
        CellTables & t = tables[c];
        t.gather_depth = scheme.gather_depth[c0 + c];
        t.scale = h_scale(c);
        t.basis_mean.resize(h_mean.extent(1));
        for (size_t l = 0; l < t.basis_mean.size(); l++) t.basis_mean[l] = h_mean(c, l);
        t.si.resize(h_si.extent(1));
        for (size_t k = 0; k < t.si.size(); k++) t.si[k] = h_si(c, k);
        n_large[c] = h_large_size(c);
        for (uint8_t k = 0; k < teno::MAX_FACES; k++) {
            t.small_size[k] = h_small_size(c, k);
            n_small[c] += t.small_size[k];
        }
    }
    const size_t chunk = c0 / CHUNK_CELLS;
    large_rows(scheme).download(scheme.stencil_large, large_slices, scheme.stencil_large_storage[chunk], c0, n_large,
                                tables);
    small_rows(scheme).download(scheme.stencil_small, small_slices, scheme.stencil_small_storage[chunk], c0, n_small,
                                tables);
}

/**
 * @brief Run precompute(i, tables, failed_large, invalid_small) over the
 *        reconstructed cells chunk by chunk, moving each chunk's tables to
 *        the device arrays. Returns the largest central stencil.
 */
template <typename F>
uint16_t precompute_in_chunks(TENO & scheme, const uint32_t n_reconstructed, F && precompute, uint32_t & n_failed_large,
                              uint32_t & n_invalid_small) {
    TableBuilder builder(scheme, n_reconstructed);
    std::vector<CellTables> chunk;
    uint16_t ns_used = 0;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        uint32_t failed = 0, invalid = 0;
        Kokkos::parallel_reduce("teno_precompute",
                                Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>(
                                    c0, c0 + chunk.size()),
                                [&](const uint32_t i, uint32_t & failed_large, uint32_t & invalid_small) {
            precompute(i, chunk[i - c0], failed_large, invalid_small);
        }, failed, invalid);
        n_failed_large += failed;
        n_invalid_small += invalid;
        for (const CellTables & t : chunk) ns_used = std::max<uint16_t>(ns_used, t.large_cells.size());
        if (n_failed_large == 0) builder.add(c0, chunk);
    }
    if (n_failed_large == 0) builder.finish();
    return ns_used;
}

} // namespace

TENO::TENO() {
    type = FaceReconstructionType::TENO;
}

TENO::~TENO() {
    // Empty
}

void TENO::read_options(const toml::value & input) {
    const int order = toml::find_or<int>(input, "order", 5);
    // The small stencils are degree 2, so the central polynomial must be at least degree 2
    if (order < 3 || order > teno::MAX_DEGREE + 1) {
        throw std::runtime_error("TENO order must be between 3 and " +
                                 std::to_string(teno::MAX_DEGREE + 1) + " (use MUSCL for second order).");
    }
    degree = order - 1;
    n_dof_large = teno::n_dof(degree);
    stencil_factor = find_real_or(input, "stencil_factor", 2.0);
    n_stencil_large = static_cast<uint16_t>(std::ceil(stencil_factor * n_dof_large));
    n_stencil_small = toml::find_or<int>(input, "small_stencil_size", (N_DIM == 2) ? 10 : 2 * teno::NK_SMALL);
    sigma_threshold = find_real_or(input, "troubled_threshold", 1.0e-3);
    sigma_upper = find_real_or(input, "troubled_upper", 1.0e-2);
    C_T = find_real_or(input, "C_T", -1.0);
    characteristic = toml::find_or<bool>(input, "characteristic", true);
    max_condition = find_real_or(input, "max_condition", 1.0e8);
    bound_preserving = toml::find_or<bool>(input, "bound_preserving", false);
    cache_file = toml::find_or<std::string>(input, "cache_file", "");
    // One file per rank, made for this partition
    if (!cache_file.empty() && comm::size() > 1) {
        cache_file += ".r" + std::to_string(comm::rank()) + "-of-" + std::to_string(comm::size());
    }
}

void TENO::init(const toml::value & input) {
    read_options(input);
    const int order = degree + 1;
    const int n_gp = std::max(1, std::min<int>(teno::MAX_FACE_QUAD, (order + 1) / 2));
    quadrature_face = GaussLegendre(n_gp);
    if constexpr (N_DIM == 3) init_face_quadrature_3d(order);

    Kokkos::Timer timer;
    cache_loaded = !cache_file.empty() && load_cache();
    if (!cache_loaded) compute_stencils_and_matrices();
    allocate_scratch();
    print();
    std::cout << "> Precomputation time: " << timer.seconds() << " s" << std::endl;
}

void TENO::print() const {
    std::cout << LOG_SEPARATOR << std::endl;
    std::cout << "Face reconstruction: " << FACE_RECONSTRUCTION_NAMES.at(type) << std::endl;
    std::cout << "> Order: " << (int)degree + 1 << " (polynomial degree " << (int)degree << ")" << std::endl;
    std::cout << "> Large stencil size: " << n_stencil_large << " + target" << std::endl;
    std::cout << "> Small stencil size: " << n_stencil_small << " + target" << std::endl;
    std::cout << "> Face quadrature points: " << (int)n_face_quadrature_points() << std::endl;
    std::cout << "> Troubled-cell threshold: " << sigma_threshold << std::endl;
    if (C_T > 0.0) {
        std::cout << "> C_T: " << C_T << std::endl;
    } else {
        std::cout << "> C_T: adaptive" << std::endl;
    }
    std::cout << "> Characteristic decomposition: " << (characteristic ? "yes" : "no") << std::endl;
    std::cout << "> Bound-preserving scaling in troubled cells: " << (bound_preserving ? "yes" : "no") << std::endl;
    std::cout << LOG_SEPARATOR << std::endl;
}

uint8_t TENO::n_face_quadrature_points() const {
    if constexpr (N_DIM == 3) return face_quad_weights.extent(1);
    return quadrature_face.h_points.extent(0);
}

void TENO::compute_stencils_and_matrices() {
    if constexpr (N_DIM == 3) {
        compute_stencils_and_matrices_3d();
        return;
    }
    const uint32_t n_cells = mesh->n_cells;
    const uint8_t r = degree;
    const uint8_t nk = n_dof_large;
    const uint16_t ns = n_stencil_large;
    const uint16_t nss = n_stencil_small;
    const uint16_t ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk));
    const uint16_t nss_max = 2 * nss;

    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }

    // Vertex-neighbor adjacency
    std::vector<std::vector<uint32_t>> cells_of_node(mesh->n_nodes);
    for (uint32_t i = 0; i < n_cells; i++) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            cells_of_node[mesh->h_node_of_cell(i, k)].push_back(i);
        }
    }
    std::vector<std::vector<uint32_t>> neighbors(n_cells);
    Kokkos::parallel_for("teno_neighbors", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t i) {
        std::vector<uint32_t> & nb = neighbors[i];
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            for (uint32_t c : cells_of_node[mesh->h_node_of_cell(i, k)]) {
                if (c != i) nb.push_back(c);
            }
        }
        std::sort(nb.begin(), nb.end(),
                  [&](uint32_t a, uint32_t b) { return mesh->h_global_cell(a) < mesh->h_global_cell(b); });
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    });

    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    const TriangleRule rule(r + 2);
    uint32_t n_failed_large = 0;
    uint32_t n_invalid_small = 0;

    auto precompute = [&](const uint32_t i, CellTables & t, uint32_t & failed_large, uint32_t & invalid_small) {
        const double x0 = mesh->h_cell_coords(i, 0);
        const double y0 = mesh->h_cell_coords(i, 1);
        const double h = std::sqrt(mesh->h_cell_volume(i));
        t.scale = h;

        // Stencil entries are interior cells, or mirror images of interior cells
        // across a straight boundary segment (face >= 0) carrying the boundary
        // condition's ghost state. Mirrors give boundary cells full, centered
        // stencils instead of one-sided extrapolation, which is unstable at
        // inflow boundaries.
        struct Entry {
            uint32_t cell;
            int32_t face;
            double x, y;
        };
        auto mirror = [&](int32_t face, double px, double py, double & qx, double & qy) {
            if (face < 0) {
                qx = px;
                qy = py;
                return;
            }
            const double nx = mesh->h_face_normals(face, 0) / mesh->h_face_area(face);
            const double ny = mesh->h_face_normals(face, 1) / mesh->h_face_area(face);
            const double d = (px - mesh->h_face_coords(face, 0)) * nx + (py - mesh->h_face_coords(face, 1)) * ny;
            qx = px - 2.0 * d * nx;
            qy = py - 2.0 * d * ny;
        };

        // Mean of each monomial over a stencil entry, in the frame of target cell i
        auto monomial_means = [&](const Entry & e, uint8_t deg, std::vector<double> & means) {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(e.cell);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(e.cell, k);
                double qx, qy;
                mirror(e.face, mesh->h_node_coords(node, 0), mesh->h_node_coords(node, 1), qx, qy);
                px[k] = (qx - x0) / h;
                py[k] = (qy - y0) / h;
            }
            const uint8_t n = teno::n_dof(deg);
            means.assign(n, 0.0);
            double area = 0.0;
            rtype phi[teno::MAX_NK];
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                teno::monomials(deg, x, y, phi);
                for (uint8_t l = 0; l < n; l++) means[l] += w * phi[l];
                area += w;
            });
            for (uint8_t l = 0; l < n; l++) means[l] /= area;
        };

        std::vector<double> mean0;
        monomial_means(Entry{i, -1, x0, y0}, r, mean0);
        t.basis_mean.assign(mean0.begin(), mean0.end());

        auto point_in_cell = [&](uint32_t c, double px, double py) {
            const uint32_t n = mesh->h_n_nodes_of_cell(c);
            int sign = 0;
            for (uint32_t k = 0; k < n; k++) {
                const uint32_t a = mesh->h_node_of_cell(c, k), b = mesh->h_node_of_cell(c, (k + 1) % n);
                const double cross = (mesh->h_node_coords(b, 0) - mesh->h_node_coords(a, 0)) * (py - mesh->h_node_coords(a, 1)) -
                                     (mesh->h_node_coords(b, 1) - mesh->h_node_coords(a, 1)) * (px - mesh->h_node_coords(a, 0));
                const double scale = 1e-12 * h * h;
                const int s = (cross > scale) - (cross < -scale);
                if (s == 0) return false;  // On an edge: treat as outside (mirror of a boundary cell)
                if (sign == 0) sign = s;
                if (s != sign) return false;
            }
            return true;
        };

        // Candidates by vertex-neighbor layers plus their mirror images across
        // nearby boundary lines, sorted by distance
        int layers_used = 0;
        bool truncated = false;
        auto gather = [&](size_t n_min, int max_layers) {
            std::vector<uint32_t> layer = {i}, cells = {i}, next;
            std::vector<Entry> entries;
            for (int depth = 0; depth < max_layers && entries.size() < n_min; depth++) {
                next.clear();
                for (uint32_t c : layer) {
                    // The outermost halo layer misses neighbors on other ranks
                    if (c >= mesh->n_complete()) truncated = true;
                    for (uint32_t nb : neighbors[c]) {
                        if (std::find(cells.begin(), cells.end(), nb) == cells.end()) {
                            cells.push_back(nb);
                            next.push_back(nb);
                        }
                    }
                }
                if (next.empty()) break;
                layer = next;
                layers_used = std::max(layers_used, depth + 1);
                // Straight boundary lines touched by the gathered cells. One line can
                // carry several conditions (e.g. inflow then wall), so each image takes
                // its state from the line's face nearest to it.
                struct Line {
                    double nx, ny;
                    std::vector<int32_t> faces;
                };
                std::vector<Line> lines;
                for (uint32_t c : cells) {
                    for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                        const uint32_t f = mesh->h_face_of_cell(c, k);
                        if (mesh->h_cells_of_face(f, 1) >= 0 || h_face_bc(f) < 0) continue;
                        if (h_bcs(h_face_bc(f)).type == BoundaryType::PARTITION) continue;
                        const double nx = mesh->h_face_normals(f, 0) / mesh->h_face_area(f);
                        const double ny = mesh->h_face_normals(f, 1) / mesh->h_face_area(f);
                        Line * match = nullptr;
                        for (Line & line : lines) {
                            const int32_t g = line.faces[0];
                            const double off = (mesh->h_face_coords(f, 0) - mesh->h_face_coords(g, 0)) * line.nx +
                                               (mesh->h_face_coords(f, 1) - mesh->h_face_coords(g, 1)) * line.ny;
                            if (std::abs(nx * line.nx + ny * line.ny - 1.0) < 1e-10 && std::abs(off) < 1e-10 * h) {
                                match = &line;
                                break;
                            }
                        }
                        if (match == nullptr) {
                            lines.push_back(Line{nx, ny, {}});
                            match = &lines.back();
                        }
                        if (std::find(match->faces.begin(), match->faces.end(), (int32_t)f) == match->faces.end()) {
                            match->faces.push_back(f);
                        }
                    }
                }
                entries.clear();
                for (uint32_t c : cells) {
                    if (c != i) entries.push_back(Entry{c, -1, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1)});
                    for (const Line & line : lines) {
                        Entry e{c, line.faces[0], 0.0, 0.0};
                        mirror(e.face, mesh->h_cell_coords(c, 0), mesh->h_cell_coords(c, 1), e.x, e.y);
                        // Images that land inside the domain (non-convex boundaries) are not ghosts
                        bool inside = false;
                        for (uint32_t other : cells) {
                            if (point_in_cell(other, e.x, e.y)) {
                                inside = true;
                                break;
                            }
                        }
                        if (inside) continue;
                        // The ghost state comes from the line's face nearest to the image
                        double best = std::numeric_limits<double>::max();
                        for (int32_t f : line.faces) {
                            const double dx = mesh->h_face_coords(f, 0) - 0.5 * (e.x + mesh->h_cell_coords(c, 0));
                            const double dy = mesh->h_face_coords(f, 1) - 0.5 * (e.y + mesh->h_cell_coords(c, 1));
                            if (dx * dx + dy * dy < best) {
                                best = dx * dx + dy * dy;
                                e.face = f;
                            }
                        }
                        entries.push_back(e);
                    }
                }
            }
            auto dist2 = [&](const Entry & e) {
                return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0);
            };
            std::stable_sort(entries.begin(), entries.end(),
                             [&](const Entry & a, const Entry & b) { return dist2(a) < dist2(b); });
            return entries;
        };

        // Least-squares system rows: mean of psi_l over each stencil entry
        auto build_pinv = [&](const std::vector<Entry> & stencil, uint8_t deg, std::vector<double> & P) {
            const uint8_t n = teno::n_dof(deg);
            const int m = stencil.size();
            std::vector<double> A(m * n);
            std::vector<double> means;
            for (int s = 0; s < m; s++) {
                monomial_means(stencil[s], deg, means);
                for (uint8_t l = 0; l < n; l++) A[s * n + l] = means[l] - mean0[l];
            }
            return pseudo_inverse(A, m, n, P, max_condition);
        };

        // Large central stencil, grown until the least-squares system has full rank
        // (anisotropic cells can have too few distinct rows/columns)
        std::vector<Entry> candidates = gather(ns_max, 64);
        std::vector<double> P;
        bool ok = false;
        // Never split a group of equidistant candidates: keeps stencils
        // independent of cell numbering, so mirror-symmetric meshes give
        // mirror-symmetric reconstructions
        auto dist2 = [&](const Entry & e) { return (e.x - x0) * (e.x - x0) + (e.y - y0) * (e.y - y0); };
        auto splits_tie = [&](const std::vector<Entry> & list, size_t n) {
            return n < list.size() && std::abs(dist2(list[n]) - dist2(list[n - 1])) < 1e-10 * h * h;
        };
        uint16_t n_used = ns;
        for (; n_used <= std::min<size_t>(ns_max, candidates.size()); n_used++) {
            if (splits_tie(candidates, n_used)) continue;
            std::vector<Entry> stencil(candidates.begin(), candidates.begin() + n_used);
            if (build_pinv(stencil, r, P)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            // A stencil cut off by the halo is retried once the halo is deep enough
            if (!truncated) failed_large++;
        } else {
            t.large_pinv.resize(n_used * nk);
            for (uint16_t s = 0; s < n_used; s++) {
                t.large_cells.push_back(candidates[s].cell);
                t.large_faces.push_back(candidates[s].face);
                for (uint8_t l = 0; l < nk; l++) t.large_pinv[s * nk + l] = P[l * n_used + s];
            }
        }

        // Small sector stencils, one per face
        std::vector<Entry> wide = gather(8 * nss, 6);
        t.gather_depth = layers_used;
        const uint32_t n_faces = mesh->h_n_faces_of_cell(i);
        for (uint32_t k = 0; k < n_faces; k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            const uint32_t na = mesh->h_node_of_face(f, 0);
            const uint32_t nb = mesh->h_node_of_face(f, 1);
            const double ax = mesh->h_node_coords(na, 0) - x0, ay = mesh->h_node_coords(na, 1) - y0;
            const double bx = mesh->h_node_coords(nb, 0) - x0, by = mesh->h_node_coords(nb, 1) - y0;
            const double det = ax * by - ay * bx;
            std::vector<Entry> sector;
            for (const Entry & e : wide) {
                const double dx = e.x - x0;
                const double dy = e.y - y0;
                const double alpha = (dx * by - dy * bx) / det;
                const double beta = (ax * dy - ay * dx) / det;
                if (alpha >= -1e-10 && beta >= -1e-10) sector.push_back(e);
            }
            size_t n_sector = nss;
            while (n_sector < nss_max && splits_tie(sector, n_sector)) n_sector++;
            if (sector.size() < nss || splits_tie(sector, n_sector)) {
                invalid_small++;
                continue;
            }
            sector.resize(n_sector);
            if (!build_pinv(sector, 2, P)) {
                invalid_small++;
                continue;
            }
            t.small_size[k] = n_sector;
            for (uint16_t s = 0; s < n_sector; s++) {
                t.small_cells.push_back(sector[s].cell);
                t.small_faces.push_back(sector[s].face);
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) t.small_pinv.push_back(P[l * n_sector + s]);
            }
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m
        {
            const uint32_t n_nodes = mesh->h_n_nodes_of_cell(i);
            std::vector<double> px(n_nodes), py(n_nodes);
            for (uint32_t k = 0; k < n_nodes; k++) {
                const uint32_t node = mesh->h_node_of_cell(i, k);
                px[k] = (mesh->h_node_coords(node, 0) - x0) / h;
                py[k] = (mesh->h_node_coords(node, 1) - y0) / h;
            }
            std::vector<double> M(nk * nk, 0.0);
            std::vector<double> d(nk);
            auto falling = [](int a, int p) {
                double c = 1.0;
                for (int t = 0; t < p; t++) c *= (a - t);
                return c;
            };
            integrate_polygon(px, py, rule, [&](double x, double y, double w) {
                for (int bp = 0; bp <= r; bp++) {
                    for (int bq = 0; bp + bq <= r; bq++) {
                        if (bp + bq == 0) continue;
                        for (uint8_t l = 0; l < nk; l++) {
                            uint8_t a, b;
                            teno::exponents(l, a, b);
                            d[l] = (a >= bp && b >= bq)
                                       ? falling(a, bp) * falling(b, bq) * std::pow(x, a - bp) * std::pow(y, b - bq)
                                       : 0.0;
                        }
                        for (uint8_t l = 0; l < nk; l++) {
                            for (uint8_t m = 0; m < nk; m++) M[l * nk + m] += w * d[l] * d[m];
                        }
                    }
                }
            });
            for (uint8_t l = 0; l < nk; l++) {
                for (uint8_t m = l; m < nk; m++) t.si.push_back(M[l * nk + m]);
            }
        }
    };
    // Outer halo cells are never reconstructed; their neighborhoods are cut off
    const uint16_t ns_used =
        precompute_in_chunks(*this, mesh->n_reconstructed(), precompute, n_failed_large, n_invalid_small);

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    std::cout << "TENO: " << n_invalid_small << " small sector stencils unavailable (boundaries)." << std::endl;

    std::cout << "TENO: largest central stencil " << ns_used << " cells (nominal " << ns << ")." << std::endl;
}

namespace {

/**
 * @brief Collapsed (Stroud conical) Gauss quadrature on the reference
 *        tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1); weights sum to 1/6.
 */
struct TetRule {
    std::vector<double> x, y, z, w;
    explicit TetRule(int n) {
        std::vector<double> g, gw;
        gauss_legendre(n, g, gw);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    const double u = 0.5 * (g[i] + 1.0), v = 0.5 * (g[j] + 1.0), s = 0.5 * (g[k] + 1.0);
                    x.push_back(u);
                    y.push_back(v * (1.0 - u));
                    z.push_back(s * (1.0 - u) * (1.0 - v));
                    w.push_back(0.125 * gw[i] * gw[j] * gw[k] * (1.0 - u) * (1.0 - u) * (1.0 - v));
                }
            }
        }
    }
};

using Point3 = std::array<double, 3>;

/**
 * @brief Integrate f(point, weight) over a cell given by its node coordinates
 *        (Gmsh/VTK order, either orientation) via its tetrahedral decomposition.
 */
template <typename F>
void integrate_cell(const std::vector<Point3> & nodes, const TetRule & rule, F && f) {
    std::vector<std::array<Point3, 4>> tets;
    cell_tetrahedra(nodes, tets);
    for (const auto & t : tets) {
        double e[3][3];
        for (int a = 0; a < 3; a++) {
            for (int d = 0; d < 3; d++) e[a][d] = t[a + 1][d] - t[0][d];
        }
        const double det = std::abs(e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1]) -
                                    e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0]) +
                                    e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0]));
        for (size_t q = 0; q < rule.w.size(); q++) {
            Point3 p;
            for (int d = 0; d < 3; d++) p[d] = t[0][d] + rule.x[q] * e[0][d] + rule.y[q] * e[1][d] + rule.z[q] * e[2][d];
            f(p, rule.w[q] * det);
        }
    }
}

double det3(const Point3 & a, const Point3 & b, const Point3 & c) {
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
}

} // namespace

void TENO::compute_stencils_and_matrices_3d() {
    const uint32_t n_cells = mesh->n_cells;
    const uint8_t r = degree;
    const uint8_t nk = n_dof_large;
    const uint16_t ns = n_stencil_large;
    const uint16_t nss = n_stencil_small;
    // Shells of equidistant candidates are large on 3D lattices; leave room to finish one
    const uint16_t ns_max = static_cast<uint16_t>(std::ceil(3.5 * nk)) + 64;
    const uint16_t nss_max = 2 * nss;

    for (uint32_t i = 0; i < n_cells; i++) {
        if (mesh->h_n_faces_of_cell(i) > teno::MAX_FACES) {
            throw std::runtime_error("TENO supports cells with at most " +
                                     std::to_string(teno::MAX_FACES) + " faces.");
        }
    }

    std::vector<std::vector<uint32_t>> cells_of_node(mesh->n_nodes);
    for (uint32_t i = 0; i < n_cells; i++) {
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            cells_of_node[mesh->h_node_of_cell(i, k)].push_back(i);
        }
    }
    std::vector<std::vector<uint32_t>> neighbors(n_cells);
    Kokkos::parallel_for("teno_neighbors", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t i) {
        std::vector<uint32_t> & nb = neighbors[i];
        for (uint32_t k = 0; k < mesh->h_n_nodes_of_cell(i); k++) {
            for (uint32_t c : cells_of_node[mesh->h_node_of_cell(i, k)]) {
                if (c != i) nb.push_back(c);
            }
        }
        std::sort(nb.begin(), nb.end(),
                  [&](uint32_t a, uint32_t b) { return mesh->h_global_cell(a) < mesh->h_global_cell(b); });
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    });

    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);

    // Collapsed Gauss with n points per direction is exact to degree 2n - 3 on a tet
    const TetRule rule((r + 4) / 2);
    const TetRule rule_si(r + 1);
    uint32_t n_failed_large = 0;
    uint32_t n_invalid_small = 0;

    auto node = [&](uint32_t n) {
        Point3 p;
        for (int d = 0; d < 3; d++) p[d] = mesh->h_node_coords(n, d);
        return p;
    };
    auto unit_normal = [&](uint32_t f) {
        Point3 n;
        for (int d = 0; d < 3; d++) n[d] = mesh->h_face_normals(f, d) / mesh->h_face_area(f);
        return n;
    };

    // Central moments of every cell in its own scaled frame, mean of
    // ((x - x_c) / h_c)^a ((y - y_c) / h_c)^b ((z - z_c) / h_c)^c, so that
    // monomial means over unmirrored stencil entries follow by binomial
    // expansion instead of quadrature. Only total degrees below nm are stored.
    const int nm = 2 * r - 1;
    std::vector<int> moment_slot(nm * nm * nm, -1);
    int n_moments = 0;
    for (int a = 0; a < nm; a++) {
        for (int b = 0; a + b < nm; b++) {
            for (int c = 0; a + b + c < nm; c++) moment_slot[(a * nm + b) * nm + c] = n_moments++;
        }
    }
    auto moment = [&](int a, int b, int c) { return moment_slot[(a * nm + b) * nm + c]; };
    std::vector<double> moments(static_cast<size_t>(n_cells) * n_moments, 0.0);
    Kokkos::parallel_for("teno_moments", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_cells),
                         [&](const uint32_t c) {
        Point3 xc;
        for (int d = 0; d < 3; d++) xc[d] = mesh->h_cell_coords(c, d);
        const double hc = std::cbrt(mesh->h_cell_volume(c));
        std::vector<Point3> p(mesh->h_n_nodes_of_cell(c));
        for (size_t k = 0; k < p.size(); k++) {
            const Point3 q = node(mesh->h_node_of_cell(c, k));
            for (int d = 0; d < 3; d++) p[k][d] = (q[d] - xc[d]) / hc;
        }
        double * m = &moments[static_cast<size_t>(c) * n_moments];
        double vol = 0.0;
        integrate_cell(p, rule_si, [&](const Point3 & x, double w) {
            double px[12], py[12], pz[12];
            px[0] = py[0] = pz[0] = 1.0;
            for (int k = 1; k < nm; k++) {
                px[k] = px[k - 1] * x[0];
                py[k] = py[k - 1] * x[1];
                pz[k] = pz[k - 1] * x[2];
            }
            for (int a = 0; a < nm; a++) {
                for (int b = 0; a + b < nm; b++) {
                    for (int cc = 0; a + b + cc < nm; cc++) m[moment(a, b, cc)] += w * px[a] * py[b] * pz[cc];
                }
            }
            vol += w;
        });
        for (int k = 0; k < n_moments; k++) m[k] /= vol;
    });
    std::vector<std::array<uint8_t, 3>> expo_all(nk);
    for (uint8_t l = 0; l < nk; l++) teno::exponents(l, expo_all[l][0], expo_all[l][1], expo_all[l][2]);
    double binom[12][12] = {};
    for (int n = 0; n < 12; n++) {
        binom[n][0] = 1.0;
        for (int k = 1; k <= n; k++) binom[n][k] = binom[n - 1][k - 1] + (k < n ? binom[n - 1][k] : 0.0);
    }

    auto precompute = [&](const uint32_t i, CellTables & t, uint32_t & failed_large, uint32_t & invalid_small) {
        Point3 x0;
        for (int d = 0; d < 3; d++) x0[d] = mesh->h_cell_coords(i, d);
        const double h = std::cbrt(mesh->h_cell_volume(i));
        t.scale = h;

        // Stencil entries: interior cells, or mirror images of interior cells
        // across a planar boundary (face >= 0) carrying the boundary
        // condition's ghost state
        struct Entry {
            uint32_t cell;
            int32_t face;
            Point3 x;
        };
        auto mirror = [&](int32_t face, const Point3 & p) {
            if (face < 0) return p;
            const Point3 n = unit_normal(face);
            double d = 0.0;
            for (int k = 0; k < 3; k++) d += (p[k] - mesh->h_face_coords(face, k)) * n[k];
            Point3 q;
            for (int k = 0; k < 3; k++) q[k] = p[k] - 2.0 * d * n[k];
            return q;
        };
        auto scaled_nodes = [&](uint32_t c, int32_t face) {
            std::vector<Point3> p(mesh->h_n_nodes_of_cell(c));
            for (size_t k = 0; k < p.size(); k++) {
                const Point3 q = mirror(face, node(mesh->h_node_of_cell(c, k)));
                for (int d = 0; d < 3; d++) p[k][d] = (q[d] - x0[d]) / h;
            }
            return p;
        };

        auto monomial_means = [&](const Entry & e, uint8_t deg, std::vector<double> & means) {
            const uint8_t n = teno::n_dof(deg);
            means.assign(n, 0.0);
            if (e.face < 0) {
                // ((x - x0) / h)^a = (d + s xi)^a with d = (x_c - x0) / h, s = h_c / h
                const double s = std::cbrt(mesh->h_cell_volume(e.cell)) / h;
                double dpow[3][12], spow[12];
                spow[0] = 1.0;
                for (int k = 1; k < nm; k++) spow[k] = spow[k - 1] * s;
                for (int d = 0; d < 3; d++) {
                    dpow[d][0] = 1.0;
                    const double dd = (mesh->h_cell_coords(e.cell, d) - x0[d]) / h;
                    for (int k = 1; k < nm; k++) dpow[d][k] = dpow[d][k - 1] * dd;
                }
                const double * m = &moments[static_cast<size_t>(e.cell) * n_moments];
                for (uint8_t l = 0; l < n; l++) {
                    const auto & ex = expo_all[l];
                    double sum = 0.0;
                    for (int ka = 0; ka <= ex[0]; ka++) {
                        const double ta = binom[ex[0]][ka] * dpow[0][ex[0] - ka];
                        for (int kb = 0; kb <= ex[1]; kb++) {
                            const double tb = ta * binom[ex[1]][kb] * dpow[1][ex[1] - kb];
                            for (int kc = 0; kc <= ex[2]; kc++) {
                                sum += tb * binom[ex[2]][kc] * dpow[2][ex[2] - kc] * spow[ka + kb + kc] *
                                       m[moment(ka, kb, kc)];
                            }
                        }
                    }
                    means[l] = sum;
                }
                return;
            }
            double vol = 0.0;
            rtype phi[teno::MAX_NK];
            integrate_cell(scaled_nodes(e.cell, e.face), rule, [&](const Point3 & p, double w) {
                teno::monomials(deg, p[0], p[1], p[2], phi);
                for (uint8_t l = 0; l < n; l++) means[l] += w * phi[l];
                vol += w;
            });
            for (uint8_t l = 0; l < n; l++) means[l] /= vol;
        };

        std::vector<double> mean0;
        monomial_means(Entry{i, -1, x0}, r, mean0);
        t.basis_mean.assign(mean0.begin(), mean0.end());

        // Strictly inside cell c (points on a face count as outside)
        auto point_in_cell = [&](uint32_t c, const Point3 & p) {
            for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                const uint32_t f = mesh->h_face_of_cell(c, k);
                const double sign = (mesh->h_cells_of_face(f, 0) == (int32_t)c) ? 1.0 : -1.0;
                double d = 0.0;
                for (int q = 0; q < 3; q++) d += (p[q] - mesh->h_face_coords(f, q)) * sign * mesh->h_face_normals(f, q);
                if (d > -1e-12 * h * mesh->h_face_area(f)) return false;
            }
            return true;
        };

        auto dist2 = [&](const Entry & e) {
            double s = 0.0;
            for (int d = 0; d < 3; d++) s += (e.x[d] - x0[d]) * (e.x[d] - x0[d]);
            return s;
        };

        // Candidates by vertex-neighbor layers plus their mirror images across
        // nearby boundary planes, sorted by distance
        int layers_used = 0;
        bool truncated = false;
        auto gather = [&](size_t n_min, int max_layers) {
            std::vector<uint32_t> layer = {i}, cells = {i}, next;
            std::vector<Entry> entries;
            for (int depth = 0; depth < max_layers && entries.size() < n_min; depth++) {
                next.clear();
                for (uint32_t c : layer) {
                    // The outermost halo layer misses neighbors on other ranks
                    if (c >= mesh->n_complete()) truncated = true;
                    for (uint32_t nb : neighbors[c]) {
                        if (std::find(cells.begin(), cells.end(), nb) == cells.end()) {
                            cells.push_back(nb);
                            next.push_back(nb);
                        }
                    }
                }
                if (next.empty()) break;
                layer = next;
                layers_used = std::max(layers_used, depth + 1);
                // Planar boundaries touched by the gathered cells; each image takes
                // its state from the plane's face nearest to it
                struct Plane {
                    Point3 n;
                    std::vector<int32_t> faces;
                };
                std::vector<Plane> planes;
                for (uint32_t c : cells) {
                    for (uint32_t k = 0; k < mesh->h_n_faces_of_cell(c); k++) {
                        const uint32_t f = mesh->h_face_of_cell(c, k);
                        if (mesh->h_cells_of_face(f, 1) >= 0 || h_face_bc(f) < 0) continue;
                        if (h_bcs(h_face_bc(f)).type == BoundaryType::PARTITION) continue;
                        const Point3 n = unit_normal(f);
                        Plane * match = nullptr;
                        for (Plane & plane : planes) {
                            const int32_t g = plane.faces[0];
                            double off = 0.0, cos = 0.0;
                            for (int d = 0; d < 3; d++) {
                                off += (mesh->h_face_coords(f, d) - mesh->h_face_coords(g, d)) * plane.n[d];
                                cos += n[d] * plane.n[d];
                            }
                            if (std::abs(cos - 1.0) < 1e-10 && std::abs(off) < 1e-10 * h) {
                                match = &plane;
                                break;
                            }
                        }
                        if (match == nullptr) {
                            planes.push_back(Plane{n, {}});
                            match = &planes.back();
                        }
                        if (std::find(match->faces.begin(), match->faces.end(), (int32_t)f) == match->faces.end()) {
                            match->faces.push_back(f);
                        }
                    }
                }
                entries.clear();
                for (uint32_t c : cells) {
                    Point3 xc;
                    for (int d = 0; d < 3; d++) xc[d] = mesh->h_cell_coords(c, d);
                    if (c != i) entries.push_back(Entry{c, -1, xc});
                    for (const Plane & plane : planes) {
                        Entry e{c, plane.faces[0], mirror(plane.faces[0], xc)};
                        bool inside = false;
                        for (uint32_t other : cells) {
                            if (point_in_cell(other, e.x)) {
                                inside = true;
                                break;
                            }
                        }
                        if (inside) continue;
                        double best = std::numeric_limits<double>::max();
                        for (int32_t f : plane.faces) {
                            double d2 = 0.0;
                            for (int d = 0; d < 3; d++) d2 += std::pow(mesh->h_face_coords(f, d) - 0.5 * (e.x[d] + xc[d]), 2);
                            if (d2 < best) {
                                best = d2;
                                e.face = f;
                            }
                        }
                        entries.push_back(e);
                    }
                }
            }
            std::stable_sort(entries.begin(), entries.end(),
                             [&](const Entry & a, const Entry & b) { return dist2(a) < dist2(b); });
            return entries;
        };

        // Means of all degree-r monomials per entry; lower degrees are a prefix
        std::map<std::pair<uint32_t, int32_t>, std::vector<double>> means_cache;
        auto build_pinv = [&](const std::vector<Entry> & stencil, uint8_t deg, std::vector<double> & P) {
            const uint8_t n = teno::n_dof(deg);
            const int m = stencil.size();
            std::vector<double> A(m * n);
            for (int s = 0; s < m; s++) {
                auto key = std::make_pair(stencil[s].cell, stencil[s].face);
                auto it = means_cache.find(key);
                if (it == means_cache.end()) {
                    std::vector<double> means;
                    monomial_means(stencil[s], r, means);
                    it = means_cache.emplace(key, std::move(means)).first;
                }
                for (uint8_t l = 0; l < n; l++) A[s * n + l] = it->second[l] - mean0[l];
            }
            return pseudo_inverse(A, m, n, P, max_condition);
        };
        auto splits_tie = [&](const std::vector<Entry> & list, size_t n) {
            return n < list.size() && std::abs(dist2(list[n]) - dist2(list[n - 1])) < 1e-10 * h * h;
        };

        // Large central stencil, grown until the least-squares system has full rank
        std::vector<Entry> candidates = gather(ns_max, 64);
        std::vector<double> P;
        bool ok = false;
        uint16_t n_used = ns;
        for (; n_used <= std::min<size_t>(ns_max, candidates.size()); n_used++) {
            if (splits_tie(candidates, n_used)) continue;
            std::vector<Entry> stencil(candidates.begin(), candidates.begin() + n_used);
            if (build_pinv(stencil, r, P)) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            // A stencil cut off by the halo is retried once the halo is deep enough
            if (!truncated) failed_large++;
        } else {
            t.large_pinv.resize(n_used * nk);
            for (uint16_t s = 0; s < n_used; s++) {
                t.large_cells.push_back(candidates[s].cell);
                t.large_faces.push_back(candidates[s].face);
                for (uint8_t l = 0; l < nk; l++) t.large_pinv[s * nk + l] = P[l * n_used + s];
            }
        }

        // Small sector stencils, one per face: entries whose direction from the
        // centroid lies in the cone spanned by the face's vertices
        std::vector<Entry> wide = gather(8 * nss, 6);
        t.gather_depth = layers_used;
        const uint32_t n_faces = mesh->h_n_faces_of_cell(i);
        for (uint32_t k = 0; k < n_faces; k++) {
            const uint32_t f = mesh->h_face_of_cell(i, k);
            std::vector<Point3> v(mesh->h_n_nodes_of_face(f));
            for (size_t a = 0; a < v.size(); a++) {
                const Point3 p = node(mesh->h_node_of_face(f, a));
                for (int d = 0; d < 3; d++) v[a][d] = p[d] - x0[d];
            }
            std::vector<Entry> sector;
            for (const Entry & e : wide) {
                Point3 dx;
                for (int d = 0; d < 3; d++) dx[d] = e.x[d] - x0[d];
                bool in = false;
                for (size_t a = 1; a + 1 < v.size() && !in; a++) {
                    const double det = det3(v[0], v[a], v[a + 1]);
                    const double alpha = det3(dx, v[a], v[a + 1]) / det;
                    const double beta = det3(v[0], dx, v[a + 1]) / det;
                    const double gamma = det3(v[0], v[a], dx) / det;
                    in = alpha >= -1e-10 && beta >= -1e-10 && gamma >= -1e-10;
                }
                if (in) sector.push_back(e);
            }
            size_t n_sector = nss;
            while (n_sector < nss_max && splits_tie(sector, n_sector)) n_sector++;
            if (sector.size() < nss || splits_tie(sector, n_sector)) {
                invalid_small++;
                continue;
            }
            sector.resize(n_sector);
            if (!build_pinv(sector, 2, P)) {
                invalid_small++;
                continue;
            }
            t.small_size[k] = n_sector;
            for (uint16_t s = 0; s < n_sector; s++) {
                t.small_cells.push_back(sector[s].cell);
                t.small_faces.push_back(sector[s].face);
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) t.small_pinv.push_back(P[l * n_sector + s]);
            }
        }

        // Smoothness-indicator matrix: M_lm = sum_{1<=|beta|<=r} int D^beta phi_l D^beta phi_m
        {
            std::vector<double> M(nk * nk, 0.0);
            std::vector<double> d(nk);
            std::vector<std::array<uint8_t, 3>> expo(nk);
            for (uint8_t l = 0; l < nk; l++) teno::exponents(l, expo[l][0], expo[l][1], expo[l][2]);
            auto falling = [](int a, int p) {
                double c = 1.0;
                for (int t = 0; t < p; t++) c *= (a - t);
                return c;
            };
            // Integrals of monomial products over the cell are its central moments
            // (the scaled volume is 1)
            const double * mom = &moments[static_cast<size_t>(i) * n_moments];
            for (int b0 = 0; b0 <= r; b0++) {
                for (int b1 = 0; b0 + b1 <= r; b1++) {
                    for (int b2 = 0; b0 + b1 + b2 <= r; b2++) {
                        if (b0 + b1 + b2 == 0) continue;
                        for (uint8_t l = 0; l < nk; l++) {
                            const auto & e = expo[l];
                            d[l] = (e[0] >= b0 && e[1] >= b1 && e[2] >= b2)
                                       ? falling(e[0], b0) * falling(e[1], b1) * falling(e[2], b2)
                                       : 0.0;
                        }
                        for (uint8_t l = 0; l < nk; l++) {
                            if (d[l] == 0.0) continue;
                            for (uint8_t m = 0; m < nk; m++) {
                                if (d[m] == 0.0) continue;
                                const auto & el = expo[l];
                                const auto & em = expo[m];
                                M[l * nk + m] += d[l] * d[m] *
                                                 mom[moment(el[0] + em[0] - 2 * b0, el[1] + em[1] - 2 * b1,
                                                            el[2] + em[2] - 2 * b2)];
                            }
                        }
                    }
                }
            }
            for (uint8_t l = 0; l < nk; l++) {
                for (uint8_t m = l; m < nk; m++) t.si.push_back(M[l * nk + m]);
            }
        }
    };
    // Outer halo cells are never reconstructed; their neighborhoods are cut off
    const uint16_t ns_used =
        precompute_in_chunks(*this, mesh->n_reconstructed(), precompute, n_failed_large, n_invalid_small);

    if (n_failed_large > 0) {
        throw std::runtime_error("TENO: could not build a full-rank large stencil for " +
                                 std::to_string(n_failed_large) + " cells (mesh too small for this order?).");
    }
    std::cout << "TENO: " << n_invalid_small << " small sector stencils unavailable (boundaries)." << std::endl;
    std::cout << "TENO: largest central stencil " << ns_used << " cells (nominal " << ns << ")." << std::endl;
}

/**
 * @brief Per-cell TENO-E reconstruction of degree DEG to the quadrature points
 *        of all of the cell's faces. SmoothPass finishes the cells below the
 *        troubled threshold and queues the others with their central
 *        coefficients. TroubledFacePass runs the stencil selection on that
 *        queue only, so the register-heavy path does not slow the smooth
 *        cells, with one thread per (cell, face) so that the few troubled
 *        cells along shocks still fill the device; TroubledFinishPass then
 *        checks admissibility over the whole cell.
 */
template <uint8_t DEG>
struct TENOFunctor {
    static constexpr uint8_t NK = teno::n_dof(DEG);
    struct SmoothPass {};
    struct TroubledFacePass {};
    struct TroubledFinishPass {};
    struct GradientPass {};

    rtype sigma_threshold;
    rtype sigma_upper;
    rtype C_T;
    bool characteristic;
    bool bound_preserving;
    rtype gamma;

    Kokkos::View<uint32_t *> offsets_faces_of_cell;
    Kokkos::View<uint32_t *> faces_of_cell;
    Kokkos::View<int32_t *[2]> cells_of_face;
    Kokkos::View<uint32_t *> offsets_nodes_of_face;
    Kokkos::View<uint32_t *> nodes_of_face;
    Kokkos::View<rtype *[N_DIM]> node_coords;
    Kokkos::View<rtype *[N_DIM]> cell_coords;
    Kokkos::View<rtype *[N_DIM]> face_coords;
    Kokkos::View<rtype *[N_DIM]> face_normals;
    Kokkos::View<rtype **> quad_points;
    Kokkos::View<rtype ***> face_quad_points;   // 3D: (face, q, dim)
    Kokkos::View<rtype **> face_quad_weights;   // 3D: (face, q), zero on padding points
    BoundaryData boundaries;

    Kokkos::View<rtype *> scale;
    Kokkos::View<rtype **> basis_mean;
    Kokkos::View<uint16_t *> stencil_large_size;
    teno::PackedStencils stencil_large;
    Kokkos::View<uint16_t **> stencil_small_size;
    teno::PackedStencils stencil_small;
    Kokkos::View<rtype **> si_matrix;
    Kokkos::View<rtype *> sigma_out;
    Kokkos::View<rtype ***> coeffs;           // (cell, l, var): central coefficients of queued cells
    Kokkos::View<uint32_t *> troubled_cells;  // queue of troubled cells
    Kokkos::View<uint32_t> n_troubled;

    Kokkos::View<rtype *[N_CONSERVATIVE]> W;
    Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution;
    Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients;  // GradientPass output

    /**
     * @brief State of a stencil entry: cell c, or its mirror across boundary face f.
     */
    KOKKOS_INLINE_FUNCTION
    void entry_W(const int32_t c, const int32_t f, rtype * W_e) const {
        FOR_I_CONSERVATIVE W_e[i] = W(c, i);
        if (f >= 0) {
            rtype n[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            rtype W_c[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_c[i] = W_e[i];
            rtype d = 0.0;
            FOR_I_DIM d += (face_coords(f, i) - cell_coords(c, i)) * n[i];
            boundaries.ghost_W_at(f, W_c, n, 2.0 * Kokkos::fabs(d), W_e);
            const BoundaryCondition & bc = boundaries.bcs(boundaries.face_bc(f));
            if (bc.type == BoundaryType::FARFIELD) {
                // The characteristic state holds at the face; outside lies the free stream
                FOR_I_CONSERVATIVE W_e[i] = bc.data[i];
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void entry_conservatives(const int32_t c, const int32_t f, rtype * U) const {
        rtype W_e[N_CONSERVATIVE];
        entry_W(c, f, W_e);
        U[0] = W_e[0];
        FOR_I_DIM U[1 + i] = W_e[0] * W_e[1 + i];
        U[N_DIM + 1] = W_e[N_DIM + 1] / (gamma - 1.0) + 0.5 * W_e[0] * dot<N_DIM>(W_e + 1, W_e + 1);
    }

    KOKKOS_INLINE_FUNCTION
    void conservatives(const int32_t c, rtype * U) const {
        rtype W_c[N_CONSERVATIVE];
        FOR_I_CONSERVATIVE W_c[i] = W(c, i);
        U[0] = W_c[0];
        FOR_I_DIM U[1 + i] = W_c[0] * W_c[1 + i];
        U[N_DIM + 1] = W_c[N_DIM + 1] / (gamma - 1.0) + 0.5 * W_c[0] * dot<N_DIM>(W_c + 1, W_c + 1);
    }

    KOKKOS_INLINE_FUNCTION
    rtype smoothness(const rtype coeffs[][N_CONSERVATIVE], const uint8_t n, const uint8_t var,
                     const uint32_t i_cell) const {
        rtype si = 0.0;
        for (uint8_t l = 0; l < n; l++) {
            // Entry (l, m) is stored at upper_index(min(l, m), max(l, m), NK)
            rtype row = 0.0;
            uint16_t k = l;
            for (uint8_t m = 0; m < l; m++) {
                row += si_matrix(i_cell, k) * coeffs[m][var];
                k += NK - m - 1;
            }
            for (uint8_t m = l; m < n; m++) row += si_matrix(i_cell, k++) * coeffs[m][var];
            si += coeffs[l][var] * row;
        }
        return si;
    }

    KOKKOS_INLINE_FUNCTION
    void to_primitives(const rtype * U, rtype * Wq) const {
        Wq[0] = U[0];
        FOR_I_DIM Wq[1 + i] = U[1 + i] / U[0];
        Wq[N_DIM + 1] = (gamma - 1.0) * (U[N_DIM + 1] - 0.5 * U[0] * dot<N_DIM>(Wq + 1, Wq + 1));
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t n_quad() const {
        if constexpr (N_DIM == 2) {
            return quad_points.extent(0);
        } else {
            return face_quad_weights.extent(1);
        }
    }

    /**
     * @brief Monomials (minus their cell means) at quadrature point q of face f,
     *        in the scaled frame of cell i_cell. False for 3D padding points.
     */
    KOKKOS_INLINE_FUNCTION
    bool face_point_basis(const uint32_t f, const uint8_t q, const uint32_t i_cell, rtype * psi) const {
        const rtype h = scale(i_cell);
        if constexpr (N_DIM == 2) {
            const rtype xc = cell_coords(i_cell, 0), yc = cell_coords(i_cell, 1);
            const uint32_t node_0 = nodes_of_face(offsets_nodes_of_face(f));
            const uint32_t node_1 = nodes_of_face(offsets_nodes_of_face(f) + 1);
            const rtype s_q = 0.5 * quad_points(q, 0);
            const rtype x = face_coords(f, 0) + s_q * (node_coords(node_1, 0) - node_coords(node_0, 0));
            const rtype y = face_coords(f, 1) + s_q * (node_coords(node_1, 1) - node_coords(node_0, 1));
            teno::monomials(DEG, (x - xc) / h, (y - yc) / h, psi);
        } else {
            if (face_quad_weights(f, q) == 0.0) return false;
            rtype xi[N_DIM];
            FOR_I_DIM xi[i] = (face_quad_points(f, q, i) - cell_coords(i_cell, i)) / h;
            teno::monomials(DEG, xi, psi);
        }
        for (uint8_t l = 0; l < NK; l++) psi[l] -= basis_mean(i_cell, l);
        return true;
    }

    /**
     * @brief Gradients of W at the centroid from the central (large-stencil)
     *        polynomial of the conservative variables, whose linear monomials
     *        carry the first derivatives at the centroid.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(GradientPass, const uint32_t i_cell) const {
        rtype U0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        rtype dU[N_CONSERVATIVE][N_DIM] = {};
        const rtype inv_h = 1.0 / scale(i_cell);
        const teno::PackedStencils::Row stencil = stencil_large.row(i_cell);
        for (uint16_t s = 0; s < stencil_large_size(i_cell); s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            FOR_I_DIM {
                const rtype P = stencil.pinv(s, i) * inv_h;
                for (uint8_t v = 0; v < N_CONSERVATIVE; v++) dU[v][i] += P * (U[v] - U0[v]);
            }
        }
        const rtype rho = W(i_cell, 0);
        rtype u[N_DIM];
        FOR_I_DIM u[i] = W(i_cell, 1 + i);
        FOR_I_DIM {
            gradients(i_cell, 0, i) = dU[0][i];
            rtype work = dU[N_DIM + 1][i] - 0.5 * dot<N_DIM>(u, u) * dU[0][i];
            for (uint8_t k = 0; k < N_DIM; k++) {
                const rtype du = (dU[1 + k][i] - u[k] * dU[0][i]) / rho;
                gradients(i_cell, 1 + k, i) = du;
                work -= rho * u[k] * du;
            }
            gradients(i_cell, N_DIM + 1, i) = (gamma - 1.0) * work;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(SmoothPass, const uint32_t i_cell) const {
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);

        const uint16_t ns = stencil_large_size(i_cell);
        // One pass over the central stencil: large-stencil coefficients (conservative
        // variables) and the troubled-cell measure, the variance of the relative
        // density jumps (Welford's update)
        rtype aK[NK][N_CONSERVATIVE] = {};
        rtype g_mean = 0.0, g_m2 = 0.0;
        const teno::PackedStencils::Row stencil = stencil_large.row(i_cell);
        for (uint16_t s = 0; s < ns; s++) {
            rtype U[N_CONSERVATIVE];
            entry_conservatives(stencil.cell(s), stencil.face(s), U);
            const rtype g = Kokkos::fabs(U[0] - W0[0]) / W0[0];
            const rtype delta = g - g_mean;
            g_mean += delta / (s + 1);
            g_m2 += delta * (g - g_mean);
            for (uint8_t l = 0; l < NK; l++) {
                const rtype P = stencil.pinv(s, l);
                FOR_I_CONSERVATIVE aK[l][i] += P * (U[i] - U0[i]);
            }
        }
        const rtype sigma = g_m2 / ns;
        sigma_out(i_cell) = sigma;
        if (sigma >= sigma_threshold) {
            for (uint8_t l = 0; l < NK; l++) {
                FOR_I_CONSERVATIVE coeffs(i_cell, l, i) = aK[l][i];
            }
            troubled_cells(Kokkos::atomic_fetch_add(&n_troubled(), 1u)) = i_cell;
            return;
        }

        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;
        bool admissible = true;
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = (cells_of_face(f, 0) == (int32_t)i_cell) ? 0 : 1;
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype psi[NK];
                if (!face_point_basis(f, q, i_cell, psi)) continue;
                rtype U_f[N_CONSERVATIVE];
                FOR_I_CONSERVATIVE U_f[i] = U0[i];
                for (uint8_t l = 0; l < NK; l++) {
                    FOR_I_CONSERVATIVE U_f[i] += aK[l][i] * psi[l];
                }
                rtype Wq[N_CONSERVATIVE];
                to_primitives(U_f, Wq);
                admissible = admissible && (Wq[0] > 0.0) && (Wq[N_DIM + 1] > 0.0) && Kokkos::isfinite(Wq[N_DIM + 1]);
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
            }
        }
        if (!admissible) {
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const uint8_t side = (cells_of_face(f, 0) == (int32_t)i_cell) ? 0 : 1;
                for (uint8_t q = 0; q < n_quad; q++) {
                    FOR_I_CONSERVATIVE face_solution(f, q, side, i) = W0[i];
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    bool is_padding(const uint32_t f, const uint8_t q) const {
        if constexpr (N_DIM == 2) {
            return false;
        } else {
            return face_quad_weights(f, q) == 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    uint8_t side_of(const uint32_t f, const uint32_t i_cell) const {
        return (cells_of_face(f, 0) == (int32_t)i_cell) ? 0 : 1;
    }

    /**
     * @brief Stencil selection and evaluation on face k of queued cell j, one
     *        thread per (cell, face). Leaves the conservative face states in
     *        face_solution for TroubledFinishPass.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledFacePass, const uint32_t idx) const {
        const uint32_t j = idx / teno::MAX_FACES;
        const uint8_t k = idx % teno::MAX_FACES;
        if (j >= n_troubled()) return;
        const uint32_t i_cell = troubled_cells(j);
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;
        if (k >= n_faces) return;
        constexpr rtype eps = 1.0e-12;
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);

        rtype aK[NK][N_CONSERVATIVE];
        for (uint8_t l = 0; l < NK; l++) {
            FOR_I_CONSERVATIVE aK[l][i] = coeffs(i_cell, l, i);
        }
        const rtype sigma = sigma_out(i_cell);
        const rtype cutoff = (C_T > 0.0) ? C_T : teno::adaptive_CT(sigma, sigma_threshold, sigma_upper);

        // Every face's selection weighs all sector stencils
        rtype aS[teno::MAX_FACES][teno::NK_SMALL][N_CONSERVATIVE] = {};
        bool valid[teno::MAX_FACES] = {};
        const teno::PackedStencils::Row stencil = stencil_small.row(i_cell);
        uint16_t start = 0;  // first slot of sector s in the cell's row
        for (uint8_t s = 0; s < n_faces; s++) {
            const uint16_t n_small = stencil_small_size(i_cell, s);
            valid[s] = n_small > 0;
            for (uint16_t e = 0; e < n_small; e++) {
                rtype U[N_CONSERVATIVE];
                entry_conservatives(stencil.cell(start + e), stencil.face(start + e), U);
                for (uint8_t l = 0; l < teno::NK_SMALL; l++) {
                    const rtype P = stencil.pinv(start + e, l);
                    FOR_I_CONSERVATIVE aS[s][l][i] += P * (U[i] - U0[i]);
                }
            }
            start += n_small;
        }

        const uint32_t f = faces_of_cell(f_begin + k);
        const int32_t c0 = cells_of_face(f, 0);
        const int32_t c1 = cells_of_face(f, 1);

        // Stencil selection per characteristic variable
        rtype L[N_CONSERVATIVE][N_CONSERVATIVE], R[N_CONSERVATIVE][N_CONSERVATIVE];
        if (characteristic) {
            const int32_t nb = (c0 == (int32_t)i_cell) ? c1 : c0;
            rtype W_avg[N_CONSERVATIVE];
            FOR_I_CONSERVATIVE W_avg[i] = (nb >= 0) ? 0.5 * (W0[i] + W(nb, i)) : W0[i];
            rtype n[N_DIM], n_vec[N_DIM];
            FOR_I_DIM n_vec[i] = face_normals(f, i);
            unit<N_DIM>(n_vec, n);
            teno::eigenvectors(W_avg, n, gamma, L, R);
        } else {
            FOR_I_CONSERVATIVE {
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) {
                    L[i][m] = (i == m) ? 1.0 : 0.0;
                    R[i][m] = L[i][m];
                }
            }
        }
        auto project = [&](const rtype * a, rtype * c) {
            FOR_I_CONSERVATIVE {
                c[i] = 0.0;
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) c[i] += L[i][m] * a[m];
            }
        };
        rtype cK[NK][N_CONSERVATIVE];
        rtype cS[teno::MAX_FACES][teno::NK_SMALL][N_CONSERVATIVE];
        rtype c0_char[N_CONSERVATIVE];
        project(U0, c0_char);
        for (uint8_t l = 0; l < NK; l++) project(aK[l], cK[l]);
        for (uint8_t s = 0; s < n_faces; s++) {
            if (!valid[s]) continue;
            for (uint8_t l = 0; l < teno::NK_SMALL; l++) project(aS[s][l], cS[s][l]);
        }
        bool use_large[N_CONSERVATIVE];
        rtype w_small[teno::MAX_FACES][N_CONSERVATIVE];
        for (uint8_t var = 0; var < N_CONSERVATIVE; var++) {
            // gamma_k = 1 / (SI_k + eps)^6, normalized by the largest one so
            // that the weights cannot overflow (even in single precision)
            const rtype si_K = smoothness(cK, NK, var, i_cell) + eps;
            rtype si_small[teno::MAX_FACES] = {};
            rtype si_min = si_K;
            for (uint8_t s = 0; s < n_faces; s++) {
                if (!valid[s]) continue;
                si_small[s] = smoothness(cS[s], teno::NK_SMALL, var, i_cell) + eps;
                si_min = Kokkos::fmin(si_min, si_small[s]);
            }
            const rtype gK = Kokkos::pow(si_min / si_K, 6.0);
            rtype g_small[teno::MAX_FACES] = {};
            rtype sum_small = 0.0;
            for (uint8_t s = 0; s < n_faces; s++) {
                if (!valid[s]) continue;
                g_small[s] = Kokkos::pow(si_min / si_small[s], 6.0);
                sum_small += g_small[s];
            }
            use_large[var] = (sum_small == 0.0) || (gK / (gK + sum_small) >= cutoff);
            if (!use_large[var]) {
                rtype n_kept = 0.0;
                for (uint8_t s = 0; s < n_faces; s++) {
                    const bool keep = valid[s] && (g_small[s] / sum_small >= cutoff);
                    w_small[s][var] = keep ? 1.0 : 0.0;
                    n_kept += w_small[s][var];
                }
                for (uint8_t s = 0; s < n_faces; s++) w_small[s][var] /= n_kept;
            }
        }

        const uint8_t side = side_of(f, i_cell);
        const uint8_t n_quad = this->n_quad();
        for (uint8_t q = 0; q < n_quad; q++) {
            rtype psi[NK];
            if (!face_point_basis(f, q, i_cell, psi)) continue;
            rtype v_char[N_CONSERVATIVE];
            for (uint8_t var = 0; var < N_CONSERVATIVE; var++) {
                v_char[var] = c0_char[var];
                if (use_large[var]) {
                    for (uint8_t l = 0; l < NK; l++) v_char[var] += cK[l][var] * psi[l];
                } else {
                    for (uint8_t s = 0; s < n_faces; s++) {
                        if (!valid[s] || w_small[s][var] == 0.0) continue;
                        rtype p_s = 0.0;
                        for (uint8_t l = 0; l < teno::NK_SMALL; l++) p_s += cS[s][l][var] * psi[l];
                        v_char[var] += w_small[s][var] * p_s;
                    }
                }
            }
            FOR_I_CONSERVATIVE {
                rtype U_f = 0.0;
                for (uint8_t m = 0; m < N_CONSERVATIVE; m++) U_f += R[i][m] * v_char[m];
                face_solution(f, q, side, i) = U_f;
            }
        }
    }

    /**
     * @brief Conservative face state left by TroubledFacePass; the cell mean
     *        on padding points.
     */
    KOKKOS_INLINE_FUNCTION
    void troubled_face_U(const uint32_t f, const uint8_t q, const uint8_t side, const rtype * U0, rtype * U) const {
        if (is_padding(f, q)) {
            FOR_I_CONSERVATIVE U[i] = U0[i];
        } else {
            FOR_I_CONSERVATIVE U[i] = face_solution(f, q, side, i);
        }
    }

    /**
     * @brief Admissibility check and optional bound-preserving limiting of
     *        queued cell j over all its faces, then the primitive face states.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(TroubledFinishPass, const uint32_t j) const {
        if (j >= n_troubled()) return;
        const uint32_t i_cell = troubled_cells(j);
        rtype U0[N_CONSERVATIVE], W0[N_CONSERVATIVE];
        conservatives(i_cell, U0);
        FOR_I_CONSERVATIVE W0[i] = W(i_cell, i);
        const uint8_t n_quad = this->n_quad();
        const uint32_t f_begin = offsets_faces_of_cell(i_cell);
        const uint8_t n_faces = offsets_faces_of_cell(i_cell + 1) - f_begin;

        bool admissible = true;
        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = side_of(f, i_cell);
            for (uint8_t q = 0; q < n_quad; q++) {
                if (is_padding(f, q)) continue;
                rtype U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                troubled_face_U(f, q, side, U0, U_f);
                to_primitives(U_f, Wq);
                admissible = admissible && (Wq[0] > 0.0) && (Wq[N_DIM + 1] > 0.0) && Kokkos::isfinite(Wq[N_DIM + 1]);
            }
        }

        rtype theta = 1.0;
        if (bound_preserving) {
            // Scale the high-order deviation so density and pressure at every face
            // point stay within the range of the cell and its face neighbors
            constexpr uint8_t P = N_DIM + 1;
            rtype lo[2] = {W0[0], W0[P]}, hi[2] = {W0[0], W0[P]};
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const int32_t c0 = cells_of_face(f, 0), c1 = cells_of_face(f, 1);
                const int32_t nb = (c0 == (int32_t)i_cell) ? c1 : c0;
                if (nb < 0) continue;
                lo[0] = Kokkos::fmin(lo[0], W(nb, 0));
                hi[0] = Kokkos::fmax(hi[0], W(nb, 0));
                lo[1] = Kokkos::fmin(lo[1], W(nb, P));
                hi[1] = Kokkos::fmax(hi[1], W(nb, P));
            }
            for (uint8_t k = 0; k < n_faces; k++) {
                const uint32_t f = faces_of_cell(f_begin + k);
                const uint8_t side = side_of(f, i_cell);
                for (uint8_t q = 0; q < n_quad; q++) {
                    if (is_padding(f, q)) continue;
                    rtype U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                    troubled_face_U(f, q, side, U0, U_f);
                    to_primitives(U_f, Wq);
                    const rtype vals[2] = {Wq[0], Wq[P]};
                    const rtype centers[2] = {W0[0], W0[P]};
                    for (uint8_t v = 0; v < 2; v++) {
                        const rtype d = vals[v] - centers[v];
                        if (d > 0.0) theta = Kokkos::fmin(theta, (hi[v] - centers[v]) / d);
                        if (d < 0.0) theta = Kokkos::fmin(theta, (lo[v] - centers[v]) / d);
                    }
                }
            }
            if (theta < 1.0) {
                theta = Kokkos::fmax(0.0, theta);
                admissible = true;
                for (uint8_t k = 0; k < n_faces; k++) {
                    const uint32_t f = faces_of_cell(f_begin + k);
                    const uint8_t side = side_of(f, i_cell);
                    for (uint8_t q = 0; q < n_quad; q++) {
                        rtype U_face[N_CONSERVATIVE], U_f[N_CONSERVATIVE], Wq[N_CONSERVATIVE];
                        troubled_face_U(f, q, side, U0, U_face);
                        FOR_I_CONSERVATIVE U_f[i] = U0[i] + theta * (U_face[i] - U0[i]);
                        to_primitives(U_f, Wq);
                        admissible = admissible && (Wq[0] > 0.0) && (Wq[P] > 0.0);
                    }
                }
            }
        }
        const bool limited = theta < 1.0;

        for (uint8_t k = 0; k < n_faces; k++) {
            const uint32_t f = faces_of_cell(f_begin + k);
            const uint8_t side = side_of(f, i_cell);
            for (uint8_t q = 0; q < n_quad; q++) {
                rtype Wq[N_CONSERVATIVE];
                if (!admissible) {
                    FOR_I_CONSERVATIVE Wq[i] = W0[i];
                } else {
                    rtype U_f[N_CONSERVATIVE];
                    troubled_face_U(f, q, side, U0, U_f);
                    if (limited) {
                        FOR_I_CONSERVATIVE U_f[i] = U0[i] + theta * (U_f[i] - U0[i]);
                        to_primitives(U_f, Wq);
                    } else if (is_padding(f, q)) {
                        FOR_I_CONSERVATIVE Wq[i] = W0[i];
                    } else {
                        to_primitives(U_f, Wq);
                    }
                }
                FOR_I_CONSERVATIVE face_solution(f, q, side, i) = Wq[i];
            }
        }
    }
};

template <uint8_t DEG>
void TENO::launch_reconstruction(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                                 Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    using Functor = TENOFunctor<DEG>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, troubled, troubled_coeffs, troubled_cells, n_troubled,
                    solution, face_solution, {}};
    using Dynamic = Kokkos::Schedule<Kokkos::Dynamic>;
    // The troubled passes cover all cells and exit past the queue length, so the
    // count never has to be read back to the host
    Kokkos::deep_copy(n_troubled, 0u);
    Kokkos::parallel_for("teno_smooth",
                         Kokkos::RangePolicy<typename Functor::SmoothPass, Dynamic>(0, mesh->n_reconstructed()), functor);
    Kokkos::parallel_for("teno_troubled_faces",
                         Kokkos::RangePolicy<typename Functor::TroubledFacePass, Dynamic>(
                             0, mesh->n_reconstructed() * teno::MAX_FACES),
                         functor);
    Kokkos::parallel_for("teno_troubled_finish",
                         Kokkos::RangePolicy<typename Functor::TroubledFinishPass, Dynamic>(0, mesh->n_reconstructed()),
                         functor);
}

template <uint8_t DEG>
void TENO::launch_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    using Functor = TENOFunctor<DEG>;
    Functor functor{sigma_threshold, sigma_upper, C_T, characteristic, bound_preserving, boundaries.gamma,
                    mesh->offsets_faces_of_cell, mesh->faces_of_cell, mesh->cells_of_face,
                    mesh->offsets_nodes_of_face, mesh->nodes_of_face, mesh->node_coords,
                    mesh->cell_coords, mesh->face_coords, mesh->face_normals,
                    quadrature_face.points, face_quad_points, face_quad_weights, boundaries,
                    scale, basis_mean, stencil_large_size, stencil_large, stencil_small_size, stencil_small,
                    si_matrix, troubled, troubled_coeffs, troubled_cells, n_troubled,
                    solution, {}, gradients};
    Kokkos::parallel_for("teno_gradients", Kokkos::RangePolicy<typename Functor::GradientPass>(0, n_cells), functor);
}

bool TENO::cell_gradients(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                          Kokkos::View<rtype *[N_CONSERVATIVE][N_DIM]> gradients, const uint32_t n_cells) {
    switch (degree) {
        case 2: launch_gradients<2>(solution, gradients, n_cells); break;
        case 3: launch_gradients<3>(solution, gradients, n_cells); break;
        case 4: launch_gradients<4>(solution, gradients, n_cells); break;
        case 5: launch_gradients<5>(solution, gradients, n_cells); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
    return true;
}

void TENO::calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> solution,
                            Kokkos::View<rtype **[2][N_CONSERVATIVE]> face_solution) {
    switch (degree) {
        case 2: launch_reconstruction<2>(solution, face_solution); break;
        case 3: launch_reconstruction<3>(solution, face_solution); break;
        case 4: launch_reconstruction<4>(solution, face_solution); break;
        case 5: launch_reconstruction<5>(solution, face_solution); break;
        default: throw std::runtime_error("TENO: unsupported degree.");
    }
}

namespace {

// Version 3 stores each reconstructed cell's tables at their actual stencil sizes
constexpr char TENO_CACHE_MAGIC[16] = "MALLARD-TENO-3";
constexpr char TENO_CACHE_FAMILY[] = "MALLARD-TENO-";

struct Fnv1a {
    uint64_t h = 1469598103934665603ULL;
    void add(const void * data, size_t n) {
        const unsigned char * p = static_cast<const unsigned char *>(data);
        for (size_t i = 0; i < n; i++) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    template <typename T>
    void add(const T & value) { add(&value, sizeof(T)); }
};

struct CacheHeader {
    char magic[sizeof(TENO_CACHE_MAGIC)] = {};
    uint64_t options_key = 0;
    uint64_t cache_key = 0;
    uint8_t halo_layers = 0;
    uint32_t n_reconstructed = 0;
};

template <typename T>
void put(std::vector<char> & buf, const T * data, size_t n) {
    const char * p = reinterpret_cast<const char *>(data);
    buf.insert(buf.end(), p, p + n * sizeof(T));
}

template <typename T>
bool get(std::istream & in, T * data, size_t n) {
    in.read(reinterpret_cast<char *>(data), n * sizeof(T));
    return in.good();
}

void write_header(std::ostream & out, const CacheHeader & h) {
    std::vector<char> buf;
    put(buf, TENO_CACHE_MAGIC, sizeof(TENO_CACHE_MAGIC));
    put(buf, &h.options_key, 1);
    put(buf, &h.cache_key, 1);
    put(buf, &h.halo_layers, 1);
    put(buf, &h.n_reconstructed, 1);
    out.write(buf.data(), buf.size());
}

bool read_header(std::istream & in, CacheHeader & h) {
    return get(in, h.magic, sizeof(h.magic)) && get(in, &h.options_key, 1) && get(in, &h.cache_key, 1) &&
           get(in, &h.halo_layers, 1) && get(in, &h.n_reconstructed, 1);
}

/** @brief Append one cell's record: its fixed-size data, then its stencils at their actual sizes. */
void serialize(const CellTables & t, std::vector<char> & buf) {
    const uint16_t n_large = t.large_cells.size();
    put(buf, &t.gather_depth, 1);
    put(buf, &t.scale, 1);
    put(buf, &n_large, 1);
    put(buf, t.small_size.data(), t.small_size.size());
    put(buf, t.basis_mean.data(), t.basis_mean.size());
    put(buf, t.si.data(), t.si.size());
    put(buf, t.large_cells.data(), n_large);
    put(buf, t.large_faces.data(), n_large);
    put(buf, t.large_pinv.data(), t.large_pinv.size());
    put(buf, t.small_cells.data(), t.small_cells.size());
    put(buf, t.small_faces.data(), t.small_faces.size());
    put(buf, t.small_pinv.data(), t.small_pinv.size());
}

bool deserialize(std::istream & in, const uint8_t nk, CellTables & t) {
    uint16_t n_large = 0;
    if (!get(in, &t.gather_depth, 1) || !get(in, &t.scale, 1) || !get(in, &n_large, 1) ||
        !get(in, t.small_size.data(), t.small_size.size())) {
        return false;
    }
    size_t n_small = 0;
    for (uint16_t n : t.small_size) n_small += n;
    t.basis_mean.resize(nk);
    t.si.resize(nk * (nk + 1) / 2);
    t.large_cells.resize(n_large);
    t.large_faces.resize(n_large);
    t.large_pinv.resize(size_t(n_large) * nk);
    t.small_cells.resize(n_small);
    t.small_faces.resize(n_small);
    t.small_pinv.resize(n_small * teno::NK_SMALL);
    return get(in, t.basis_mean.data(), t.basis_mean.size()) && get(in, t.si.data(), t.si.size()) &&
           get(in, t.large_cells.data(), n_large) && get(in, t.large_faces.data(), n_large) &&
           get(in, t.large_pinv.data(), t.large_pinv.size()) && get(in, t.small_cells.data(), n_small) &&
           get(in, t.small_faces.data(), n_small) && get(in, t.small_pinv.data(), t.small_pinv.size());
}

} // namespace

void TENO::allocate_scratch() {
    const uint32_t n_reconstructed = mesh->n_reconstructed();
    troubled = Kokkos::View<rtype *>("teno_sigma", mesh->n_cells);
    troubled_coeffs = Kokkos::View<rtype ***>("teno_troubled_coeffs", n_reconstructed, n_dof_large, N_CONSERVATIVE);
    troubled_cells = Kokkos::View<uint32_t *>("teno_troubled_cells", n_reconstructed);
    n_troubled = Kokkos::View<uint32_t>("teno_n_troubled");
}

TENO::Stencils TENO::large_stencils() const {
    Stencils out;
    out.offsets.push_back(0);
    const uint32_t n_reconstructed = scale.extent(0);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        for (const CellTables & t : chunk) {
            out.cells.insert(out.cells.end(), t.large_cells.begin(), t.large_cells.end());
            out.faces.insert(out.faces.end(), t.large_faces.begin(), t.large_faces.end());
            out.offsets.push_back(out.cells.size());
        }
    }
    return out;
}

uint64_t TENO::options_key() const {
    Fnv1a hash;
    hash.add(sizeof(rtype));
    hash.add(degree);
    hash.add(n_stencil_small);
    hash.add(stencil_factor);
    hash.add(max_condition);
    hash.add(comm::size());
    hash.add(comm::rank());
    return hash.h;
}

uint64_t TENO::cache_key() const {
    Fnv1a hash;
    hash.add(options_key());
    hash.add(mesh->n_cells);
    hash.add(mesh->n_faces);
    hash.add(mesh->n_reconstructed());
    hash.add(mesh->n_complete());
    hash.add(mesh->h_node_coords.data(), mesh->h_node_coords.span() * sizeof(rtype));
    hash.add(mesh->h_offsets_nodes_of_cell.data(), mesh->h_offsets_nodes_of_cell.span() * sizeof(uint32_t));
    hash.add(mesh->h_nodes_of_cell.data(), mesh->h_nodes_of_cell.span() * sizeof(uint32_t));
    hash.add(mesh->h_faces_of_cell.data(), mesh->h_faces_of_cell.span() * sizeof(uint32_t));
    // A rank's part of a distributed mesh: which global cells it holds, in which order
    hash.add(mesh->h_global_cell_id.data(), mesh->h_global_cell_id.size() * sizeof(uint64_t));
    auto h_face_bc = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.face_bc);
    hash.add(h_face_bc.data(), h_face_bc.span() * sizeof(int32_t));
    auto h_bcs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), boundaries.bcs);
    for (size_t b = 0; b < h_bcs.extent(0); b++) hash.add(h_bcs(b).type);
    return hash.h;
}

uint8_t TENO::cached_halo_layers(const toml::value & input) {
    TENO teno;
    teno.read_options(input);
    if (teno.cache_file.empty()) return 0;
    std::ifstream in(teno.cache_file, std::ios::binary);
    CacheHeader header;
    if (!read_header(in, header) || std::string(header.magic) != TENO_CACHE_MAGIC ||
        header.options_key != teno.options_key()) {
        return 0;
    }
    return header.halo_layers;
}

void TENO::save_cache(const uint8_t halo_layers) const {
    if (cache_file.empty() || cache_loaded) return;
    // Written aside and renamed, so an interrupted run never leaves a truncated cache
    const std::string partial = cache_file + ".partial";
    std::ofstream out(partial, std::ios::binary);
    if (!out.good()) {
        std::cout << "TENO: could not write cache file " << cache_file << "." << std::endl;
        return;
    }
    const uint32_t n_reconstructed = scale.extent(0);
    CacheHeader header;
    header.options_key = options_key();
    header.cache_key = cache_key();
    header.halo_layers = halo_layers;
    header.n_reconstructed = n_reconstructed;
    write_header(out, header);
    const auto large_slices = host_slice_start(stencil_large);
    const auto small_slices = host_slice_start(stencil_small);
    std::vector<CellTables> chunk;
    std::vector<char> buf;
    for (uint32_t c0 = 0; c0 < n_reconstructed; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        download_tables(*this, large_slices, small_slices, c0, chunk);
        buf.clear();
        for (const CellTables & t : chunk) serialize(t, buf);
        out.write(buf.data(), buf.size());
    }
    out.close();
    std::error_code error;
    if (out.good()) std::filesystem::rename(partial, cache_file, error);
    if (!out.good() || error) {
        std::filesystem::remove(partial, error);
        std::cout << "TENO: could not write cache file " << cache_file << "." << std::endl;
        return;
    }
    std::cout << "TENO: wrote stencil cache " << cache_file << " ("
              << std::filesystem::file_size(cache_file, error) / 1e6 << " MB)" << std::endl;
}

bool TENO::load_cache() {
    std::ifstream in(cache_file, std::ios::binary);
    if (!in.good()) return false;
    CacheHeader header;
    const bool read = read_header(in, header);
    const std::string magic(header.magic, strnlen(header.magic, sizeof(header.magic)));
    if (!read || magic != TENO_CACHE_MAGIC) {
        std::cout << "TENO: " << cache_file
                  << (magic.rfind(TENO_CACHE_FAMILY, 0) == 0 ? " has an older format" : " is not a TENO cache")
                  << "; recomputing." << std::endl;
        return false;
    }
    const uint32_t n_reconstructed = mesh->n_reconstructed();
    if (header.options_key != options_key() || header.n_reconstructed != n_reconstructed ||
        header.cache_key != cache_key()) {
        std::cout << "TENO: cache " << cache_file << " does not match this case; recomputing." << std::endl;
        return false;
    }
    TableBuilder builder(*this, n_reconstructed);
    std::vector<CellTables> chunk;
    bool ok = true;
    for (uint32_t c0 = 0; c0 < n_reconstructed && ok; c0 += CHUNK_CELLS) {
        chunk.assign(std::min(CHUNK_CELLS, n_reconstructed - c0), CellTables());
        for (CellTables & t : chunk) ok = ok && deserialize(in, n_dof_large, t);
        if (ok) builder.add(c0, chunk);
    }
    if (!ok) {
        std::cout << "TENO: cache " << cache_file << " is truncated; recomputing." << std::endl;
        return false;
    }
    builder.finish();
    std::cout << "TENO: loaded stencil cache " << cache_file << std::endl;
    return true;
}
