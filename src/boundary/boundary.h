/**
 * @file boundary.h
 * @author Matthew Bonanni (mbonanni001@gmail.com)
 * @brief Boundary class declaration.
 * @version 0.1
 * @date 2023-12-20
 * 
 * @copyright Copyright (c) 2023 Matthew Bonanni
 * 
 */

#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <toml++/toml.h>

#include "common/common.h"
#include "mesh/zone.h"
#include "mesh/mesh.h"
#include "physics/physics.h"

enum class BoundaryType {
    WALL_ADIABATIC = 1,
    UPT = 2,
    P_OUT = 3
};

static const std::unordered_map<std::string, BoundaryType> BOUNDARY_TYPES = {
    {"wall_adiabatic", BoundaryType::WALL_ADIABATIC},
    {"upt", BoundaryType::UPT},
    {"p_out", BoundaryType::P_OUT}
};

static const std::unordered_map<BoundaryType, std::string> BOUNDARY_NAMES = {
    {BoundaryType::WALL_ADIABATIC, "wall_adiabatic"},
    {BoundaryType::UPT, "upt"},
    {BoundaryType::P_OUT, "p_out"}
};

class Boundary {
    public:
        /**
         * @brief Construct a new Boundary object
         */
        Boundary();

        /**
         * @brief Destroy the Boundary object
         */
        virtual ~Boundary();

        /**
         * @brief Set the zone.
         * @param zone Pointer to the zone.
         */
        void set_zone(FaceZone * zone);

        /**
         * @brief Set the mesh.
         * @param mesh Pointer to the mesh.
         */
        void set_mesh(std::shared_ptr<Mesh> mesh);

        /**
         * @brief Set the physics.
         * @param physics Pointer to the physics.
         */
        void set_physics(std::shared_ptr<Physics> physics);

        /**
         * @brief Print the boundary.
         */
        virtual void print();

        /**
         * @brief Initialize the boundary.
         * @param input TOML input parameter table.
         */
        virtual void init(const toml::table & input);

        /**
         * @brief Compute and apply the boundary flux.
         * @param solution Pointer to the solution.
         * @param rhs Pointer to the right hand side.
         */
        virtual void apply(StateVector * solution,
                           StateVector * rhs);
        
    protected:
        FaceZone * zone;
        std::shared_ptr<Mesh> mesh;
        BoundaryType type;
        std::shared_ptr<Physics> physics;
    private:
};

#endif // BOUNDARY_H