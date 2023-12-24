/**
 * @file time_integrator.h
 * @author Matthew Bonanni(mbonanni001@gmail.com)
 * @brief Time integrator class declarations.
 * @version 0.1
 * @date 2023-12-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef TIME_INTEGRATOR_H
#define TIME_INTEGRATOR_H

class TimeIntegrator;

#include "solver/solver.h"

class TimeIntegrator {
    public:
        /**
         * @brief Construct a new TimeIntegrator object
         */
        TimeIntegrator();

        /**
         * @brief Destroy the TimeIntegrator object
         */
        virtual ~TimeIntegrator();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        virtual void take_step(const double& dt,
                               std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                               std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                               void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                                std::vector<std::array<double, 4>> * rhs));
    protected:
    private:
};

class ForwardEuler : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new ForwardEuler object
         */
        ForwardEuler();

        /**
         * @brief Destroy the ForwardEuler object
         */
        ~ForwardEuler();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                        std::vector<std::array<double, 4>> * rhs)) override;
    protected:
    private:
};

class RK4 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new RK4 object
         */
        RK4();

        /**
         * @brief Destroy the RK4 object
         */
        ~RK4();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                        std::vector<std::array<double, 4>> * rhs)) override;
    protected:
    private:
        std::vector<double> coeffs;
};

class RK4LowMemory : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new RK4LowMemory object
         */
        RK4LowMemory();

        /**
         * @brief Destroy the RK4LowMemory object
         */
        ~RK4LowMemory();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                        std::vector<std::array<double, 4>> * rhs)) override;
    protected:
    private:
};

class SSPRK3 : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new SSPRK3 object
         */
        SSPRK3();

        /**
         * @brief Destroy the SSPRK3 object
         */
        ~SSPRK3();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                        std::vector<std::array<double, 4>> * rhs)) override;
    protected:
    private:
        std::vector<double> coeffs;
};

class SSPRK3LowMemory : public TimeIntegrator {
    public:
        /**
         * @brief Construct a new SSPRK3LowMemory object
         */
        SSPRK3LowMemory();

        /**
         * @brief Destroy the SSPRK3LowMemory object
         */
        ~SSPRK3LowMemory();

        /**
         * @brief Take a single time step.
         * @param dt Time step size.
         * @param solution_pointers Pointers to solution vectors.
         * @param rhs_pointers Pointers to rhs vectors.
         * @param calc_rhs Function to calculate rhs.
         */
        void take_step(const double& dt,
                       std::vector<std::vector<std::array<double, 4>> *> & solution_pointers,
                       std::vector<std::vector<std::array<double, 4>> *> & rhs_pointers,
                       void (*calc_rhs)(std::vector<std::array<double, 4>> * solution,
                                        std::vector<std::array<double, 4>> * rhs)) override;
    protected:
    private:
};

#endif // TIME_INTEGRATOR_H