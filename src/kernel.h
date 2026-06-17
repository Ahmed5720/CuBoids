#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <vector>

struct BoidsParams {
    float dt = 0.4f;

    float rule1Distance = 5.0f;
    float rule2Distance = 3.0f;
    float rule3Distance = 5.0f;

    float rule1Scale = 0.01f;
    float rule2Scale = 0.10f;
    float rule3Scale = 0.10f;

    float maxSpeed = 1.0f;

    // mesh-boundary avoidance (mirrors SPH's pressure repulsion)
    float boundaryDistance = 5.0f;
    float boundaryScale = 0.20f;
};

struct SPHParams {
    float dt = 0.06f;

    float h = 4.0f;
    float mass = 1.0f;
    float restDensity = 0.05f;
    float gasConst = 250.0f;
    float viscosity = 0.5f;
    float tension = 2.0f;
    float gravity = -9.8f;     // magnitude; applied as -gravity

    //deriveSPHConstants() fills them in
    float h2 = 0.0f, h6 = 0.0f, h9 = 0.0f;
    float poly6 = 0.0f, spikyGrad = 0.0f, spikyLap = 0.0f;
    float selfDens = 0.0f, massPoly6Product = 0.0f;
};

inline void deriveSPHConstants(SPHParams& p) {
    constexpr float kPi = 3.14159265358979323846f;
    p.h2 = p.h * p.h;
    p.h6 = p.h2 * p.h2 * p.h2;
    p.h9 = p.h6 * p.h2 * p.h;
    p.poly6 = 315.0f / (64.0f * kPi * p.h9);
    p.spikyGrad = -45.0f / (kPi * p.h6);
    p.spikyLap = 45.0f / (kPi * p.h6);
    p.massPoly6Product = p.mass * p.poly6;
    p.selfDens = p.mass * p.poly6 * p.h6;
}

enum class SimulationType { Boids, SPH };


namespace Simulator{
    void initBoidsSimulation(int numBoids, const BoidsParams& params, const glm::vec3* boundaryPos, int numBoundary);
    void initSPHSimulation(int N, const SPHParams& params, const glm::vec3* boundaryPos, int numBoundary);
    void setBoidsParams(const BoidsParams& params);
    void setSPHParams(const SPHParams& params);
    void stepBoidsSimulationCoherentGrid(float dt);
    void stepSPHSimulationCoherentGrid(float dt);

    void updateBoundaryParticles(glm::vec3* boundary, int size);
    void copyToVBO(float *vbodptr_positions, float *vbodptr_velocities);
    
    void endSimulation();
    void unitTest();

    void stepBoidsSimulationScatteredGrid(float dt);
    void stepBoidsSimulationNaive(float dt);
}
