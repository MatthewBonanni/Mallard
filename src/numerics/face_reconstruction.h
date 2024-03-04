/**
 * @file face_reconstruction.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Face reconstruction class declaration.
 * @version 0.1
 * @date 2023-12-24
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef FACE_RECONSTRUCTION_H
#define FACE_RECONSTRUCTION_H

#include <memory>

#include "common_typedef.h"
#include "mesh.h"

enum class FaceReconstructionType {
    FirstOrder,
    WENO3_JS,
    WENO5_JS,
};

static const std::unordered_map<std::string, FaceReconstructionType> FACE_RECONSTRUCTION_TYPES = {
    {"FO", FaceReconstructionType::FirstOrder},
    {"WENO3_JS", FaceReconstructionType::WENO3_JS},
    {"WENO5_JS", FaceReconstructionType::WENO5_JS}
};

static const std::unordered_map<FaceReconstructionType, std::string> FACE_RECONSTRUCTION_NAMES = {
    {FaceReconstructionType::FirstOrder, "FO"},
    {FaceReconstructionType::WENO3_JS, "WENO3_JS"},
    {FaceReconstructionType::WENO5_JS, "WENO5_JS"}
};

/**
 * @brief Face reconstruction class.
 */
class FaceReconstruction {
    public:
        /**
         * @brief Construct a new Face Reconstruction object
         */
        FaceReconstruction();

        /**
         * @brief Destroy the Face Reconstruction object
         */
        virtual ~FaceReconstruction();

        /**
         * @brief Initialize the face reconstruction.
         */
        virtual void init();

        /**
         * @brief Print the face reconstruction.
         */
        void print() const;

        /**
         * @brief Set the cell conservatives.
         * @param cell_conservatives Pointer to the cell conservatives.
         */
        void set_cell_conservatives(Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives);

        /**
         * @brief Set the face conservatives.
         * @param face_conservatives Pointer to the face conservatives.
         */
        void set_face_conservatives(Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives);

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        virtual void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> * solution,
                                      Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution) = 0;
    protected:
        FaceReconstructionType type;
        Kokkos::View<rtype *[N_CONSERVATIVE]> * cell_conservatives;
        Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_conservatives;
        std::shared_ptr<Mesh> mesh;
    private:
};

class FirstOrder : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new First Order object
         */
        FirstOrder();

        /**
         * @brief Destroy the First Order object
         */
        ~FirstOrder();

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> * solution,
                              Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution) override;
    protected:
    private:
};

class WENO3_JS : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new WENO object
         */
        WENO3_JS();

        /**
         * @brief Destroy the WENO object
         */
        ~WENO3_JS();

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> * solution,
                              Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution) override;
    protected:
    private:
};

class WENO5_JS : public FaceReconstruction {
    public:
        /**
         * @brief Construct a new WENO object
         */
        WENO5_JS();

        /**
         * @brief Destroy the WENO object
         */
        ~WENO5_JS();

        /**
         * @brief Reconstruct the face values.
         * @param solution Pointer to the solution.
         * @param face_solution Pointer to the face solution.
         */
        void calc_face_values(Kokkos::View<rtype *[N_CONSERVATIVE]> * solution,
                              Kokkos::View<rtype *[2][N_CONSERVATIVE]> * face_solution) override;
    protected:
    private:
};

#endif // FACE_RECONSTRUCTION_H