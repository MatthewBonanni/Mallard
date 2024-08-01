/**
 * @file main.cpp
 * @brief Main file for Mallard.
 * @version 0.1
 * @date 2023-12-17
 * 
 * @mainpage Mallard
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 */

#include <iostream>

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include "common/common_io.h"
#include "solver/solver.h"

/**
 * @brief Entry point of the program.
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return Exit status.
 */
int main(int argc, char* argv[]) {
    // Check if there are at least two arguments (program name and at least one argument)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " -i input.toml" << std::endl;
        return 1;
    }

    std::string inputFileName;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                inputFileName = argv[i + 1];
                ++i;
            } else {
                std::cerr << "Error: Missing input file after -i flag." << std::endl;
                return 1;
            }
        }
    }

    int status = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (mpi_rank == 0) {
        std::cout << LOG_SEPARATOR << std::endl;
        print_logo();
        std::cout << LOG_SEPARATOR << std::endl;
        std::cout << "Computational configuration:" << std::endl;

        // Print MPI information
        std::cout << std::endl;
        std::cout << "MPI:" << std::endl;
        std::cout << "> Number of MPI processes: " << mpi_size << std::endl;

        // Print Kokkos execution space information
        std::cout << std::endl;
        std::cout << "Kokkos:" << std::endl;
        std::cout << "> The default execution space is: "
                  << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        std::cout << "> This space has concurrency: "
                  << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;
        std::cout << "> The default host execution space is: "
                  << typeid(Kokkos::DefaultHostExecutionSpace).name() << std::endl;
        std::cout << "> This space has concurrency: "
                  << Kokkos::DefaultHostExecutionSpace::concurrency() << std::endl;
        
        std::cout << std::endl;
        #ifdef Mallard_USE_DOUBLES
            std::cout << "Mallard has been compiled with DOUBLE precision." << std::endl;
        #else   
            std::cout << "Mallard has been compiled with SINGLE precision." << std::endl;
        #endif
    }

    // Create solver object
    Solver solver;
    status = solver.init(inputFileName);
    if (status != 0) {
        std::cerr << "Error: Solver initialization failed." << std::endl;
        return status;
    }

    // Run solver
    status = solver.run();
    if (status != 0) {
        std::cerr << "Error: Solver run failed." << std::endl;
        return status;
    }

    }

    // Finalize
    Kokkos::finalize();
    MPI_Finalize();

    return status;
}