#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f


int numFluid;
int numParticles; // total of fluid + object (fake particles)


// Kernel state (pointers are device pointers) 


// Host-side copies (read freely from host code: init, ImGui, grid sizing math).
BoidsParams h_boidsParams;
SPHParams   h_sphParams;

// Device-side mirrors — these are what every __device__/__global__ function reads.
__constant__ BoidsParams d_boidsParams;
__constant__ SPHParams   d_sphParams;



int numObjects;
dim3 threadsPerBlock(blockSize);

float* dev_pressures;
float* dev_densities;
glm::vec3* dev_forces;
glm::vec3 *dev_pos;
// ping-pong buffers.
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;
glm::vec3 *dev_vel_coherent;
glm::vec3 *dev_pos_coherent;




int *dev_particleArrayIndices; 
int *dev_particleGridIndices; 
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

void Simulator::setBoidsParams(const BoidsParams& params) {
    h_boidsParams = params;
    cudaMemcpyToSymbol(d_boidsParams, &h_boidsParams, sizeof(BoidsParams));
}

void Simulator::setSPHParams(const SPHParams& params) {
    h_sphParams = params;
    deriveSPHConstants(h_sphParams);
    cudaMemcpyToSymbol(d_sphParams, &h_sphParams, sizeof(SPHParams));
}


__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}


__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

__global__ void kernInitLattice(int N, glm::vec3* pos, glm::vec3* vel,
    int perSide, float spacing, glm::vec3 origin) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) return;

    int i = index % perSide;
    int j = (index / perSide) % perSide;
    int k = index / (perSide * perSide);

  
    glm::vec3 jitter = 0.01f * spacing * generateRandomVec3(1.0f, index);
    glm::vec3 offset = {0,50,-50 };
    pos[index] = origin + offset + glm::vec3(i, j, k) * spacing + jitter;
    vel[index] = glm::vec3(0.0f);   // start at rest
}

/**
* Initialize memory, update some globals
*/
void Simulator::initBoidsSimulation(int numBoids,  const BoidsParams& params,
    const glm::vec3* boundaryPos, int numBoundary)
{
    numFluid = numBoids;
    numObjects = numBoids;
    numParticles = numBoids + numBoundary;
    setBoidsParams(params);

    dim3 fullBlocksPerGrid((numParticles + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel1, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel_coherent, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_pos_coherent, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel2, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_particleArrayIndices, numParticles * sizeof(int));
    cudaMalloc((void**)&dev_particleGridIndices, numParticles * sizeof(int));
    checkCUDAErrorWithLine("boids cudaMalloc failed!");

    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

    kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> > (1, numBoids, dev_pos, scene_scale);

    if (numBoundary > 0)
        cudaMemcpy(dev_pos + numFluid, boundaryPos, numBoundary * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("copying boundaryPos failed!");

    gridCellWidth = std::max({ h_boidsParams.rule1Distance, h_boidsParams.rule2Distance,
                                h_boidsParams.rule3Distance, h_boidsParams.boundaryDistance });
    int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
    gridSideCount = 2 * halfSideCount;
    gridCellCount = gridSideCount * gridSideCount * gridSideCount;

    cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
    cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("grid cudaMalloc failed!");

    gridInverseCellWidth = 1.0f / gridCellWidth;
    gridMinimum = glm::vec3(-gridCellWidth * halfSideCount); // assign, don't accumulate

    cudaDeviceSynchronize();
}

void Simulator::initSPHSimulation(int N, const SPHParams& params, const glm::vec3* boundaryPos, int numBoundary)
{
    numObjects = N;
    numFluid = N;
    numParticles = numFluid + numBoundary;
    setSPHParams(params);

    dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

    cudaMalloc((void**)&dev_pos, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel1, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel_coherent, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_pos_coherent, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_vel2, numParticles * sizeof(glm::vec3));
    cudaMalloc((void**)&dev_particleArrayIndices, numParticles * sizeof(int));
    cudaMalloc((void**)&dev_particleGridIndices, numParticles * sizeof(int));
    cudaMalloc((void**)&dev_pressures, numParticles * sizeof(float));
    cudaMalloc((void**)&dev_densities, numParticles * sizeof(float));
    cudaMalloc((void**)&dev_forces, numParticles * sizeof(glm::vec3));
    checkCUDAErrorWithLine("SPH cudaMalloc failed!");

    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

    float spacing = 0.5f * h_sphParams.h;
    int   perSide = (int)ceilf(cbrtf((float)N));
    float blockWidth = (perSide - 1) * spacing;
    glm::vec3 latticeOrigin(-0.5f * blockWidth, -scene_scale + 10.0f, -0.5f * blockWidth);

    kernInitLattice << <fullBlocksPerGrid, blockSize >> > (
        numObjects, dev_pos, dev_vel1, perSide, spacing, latticeOrigin);

    if (numBoundary > 0)
        cudaMemcpy(dev_pos + numFluid, boundaryPos, numBoundary * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("copying boundaryPos failed!");

    gridCellWidth = h_sphParams.h;
    int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
    gridSideCount = 2 * halfSideCount;
    gridCellCount = gridSideCount * gridSideCount * gridSideCount;

    cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
    cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
    checkCUDAErrorWithLine("grid cudaMalloc failed!");

    gridInverseCellWidth = 1.0f / gridCellWidth;
    gridMinimum = glm::vec3(-gridCellWidth * halfSideCount);

    cudaDeviceSynchronize();
}


void Simulator::updateBoundaryParticles(glm::vec3* boundary, int numBoundary)
{
    if (numBoundary > 0)
        cudaMemcpy(dev_pos + numFluid, boundary, numBoundary * sizeof(glm::vec3), cudaMemcpyHostToDevice);
}

// Copy the boid positions into the VBO so that they can be drawn by OpenGL.
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Simulator::copyToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numFluid + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numFluid, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numFluid, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

// move toward center of mass
__device__ glm::vec3 rule1(int iSelf, const glm::vec3 *pos, const glm::vec3 *vel, int start, int end)
{
    glm::vec3 percieved_center = { 0,0,0 };
    int neighbors = 0;
    for (int i = start; i < end; i++)
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < d_boidsParams.rule1Distance)
        {
            percieved_center += pos[i];
            neighbors++;
        }
    percieved_center /= float(neighbors);
    return neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos[iSelf]) * d_boidsParams.rule1Scale;
}
// try to keep a small distance away from other boids
__device__ glm::vec3 rule2(int iSelf, const glm::vec3* pos, const glm::vec3* vel, int start, int end)
{
    glm::vec3 c = { 0,0,0 };
    for (int i = start; i < end; i++)
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < d_boidsParams.rule2Distance)
            c -= (pos[i] - pos[iSelf]);
    return c * d_boidsParams.rule2Scale;
}
// try to match velocity with that of other boids
__device__ glm::vec3 rule3(int iSelf, const glm::vec3* pos, const glm::vec3* vel, int start, int end)
{
    glm::vec3 percieved_velocity = { 0,0,0 };
    int neighbors = 0;
    for (int i = start; i < end; i++)
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < d_boidsParams.rule3Distance)
        {
            percieved_velocity += vel[i];
            neighbors++;
        }
    percieved_velocity /= float(neighbors);
    return neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * d_boidsParams.rule3Scale;
}

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 vel_change = { 0,0,0 };
    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
    vel_change += rule1(iSelf, pos, vel, 0, N);
    // Rule 2: boids try to stay a distance d away from each other
    vel_change += rule2(iSelf, pos, vel, 0, N);
    // Rule 3: boids try to match the speed of surrounding boids
    vel_change += rule3(iSelf, pos, vel, 0, N);
    return vel_change;
}



// For each of the `N` bodies, update its position based on its current velocity.

__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2 to avoid a read after write hazard
   /* int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
        return;
    glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
    float speed = glm::length(newVel);
    if (speed >  boidsParams.maxSpeed)
        newVel = glm::normalize(newVel) * boidsParams.maxSpeed;
    vel2[index] = newVel;*/
}

// For each of the `N` bodies, update its position based on its current velocity.

__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}


//  is this the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // Label each boid with the index of its grid cell.
    // Sets up a parallel array of integer indices as pointers to the actual
    // boid data in pos and vel1/vel2
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N)
        return;
    glm::vec3 xyz = ((pos[idx] - gridMin) * inverseCellWidth);
    gridIndices[idx] = gridIndex3Dto1D(xyz[0], xyz[1], xyz[2], gridResolution); 
    indices[idx] = idx;

}

__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int Cell = particleGridIndices[index];

    if (index <= 0) {
        gridCellStartIndices[Cell] = index;

        if (particleGridIndices[index + 1] != Cell)
            gridCellEndIndices[Cell] = index;
    }
    else if (index >= (N - 1)) {
        if (particleGridIndices[index - 1] != Cell)
            gridCellStartIndices[Cell] = index;

        gridCellEndIndices[Cell] = index;
    }
    else {
        if (particleGridIndices[index - 1] != Cell)
            gridCellStartIndices[Cell] = index;

        if (particleGridIndices[index + 1] != Cell)
            gridCellEndIndices[Cell] = index;
    }

}

// must also shuffle densities and pressures for them to be coherent
__global__ void kernShufflePosAndVel(int N, int* particleArrayIndices, glm::vec3* pos_coherent, glm::vec3* vel_coherent, glm::vec3* pos, glm::vec3* vel)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= N)
        return;
    int sorted = particleArrayIndices[idx];
    pos_coherent[idx] = pos[sorted];
    vel_coherent[idx] = vel[sorted];
    /*densities[idx] = densities[sorted];
    forces[idx] = forces[sorted];
    pressures[idx] = pressures[sorted]; */

}
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // -Updates a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int cnt = 0;
 
    if (idx >= N)
        return;
    //int pIdx = particleArrayIndices[idx];
    //glm::vec3 newVel = vel1[pIdx];// +computeVelocityChange(N, idx, pos, vel1);
    //glm::vec3 xyz = ((pos[pIdx] - gridMin) * inverseCellWidth);
    //int cellX = (int)xyz.x;
    //int cellY = (int)xyz.y;
    //int cellZ = (int)xyz.z;
    //glm::vec3 vel_change = { 0,0,0 };
    //int rule1neighbors = 0;
    //int rule3neighbors = 0;
    //glm::vec3 percieved_velocity = { 0,0,0 };
    //glm::vec3 percieved_center = { 0,0,0 };
    //glm::vec3 c = { 0,0,0 };
    //for (int dx = -1; dx <= 1; dx++)
    //    for (int dy = -1; dy <= 1; dy++)
    //        for (int dz = -1; dz <= 1; dz++)
    //        {   
    //           //  bounds check
    //            if (cellX + dx < 0 || cellX + dx >= gridResolution) continue;
    //            if (cellY + dy < 0 || cellY + dy >= gridResolution) continue;
    //            if (cellZ + dz < 0 || cellZ + dz >= gridResolution) continue;

    //            int gridIdx = gridIndex3Dto1D(cellX+dx, cellY+dy, cellZ + dz, gridResolution);
    //            int start = gridCellStartIndices[gridIdx];
    //            int end = gridCellEndIndices[gridIdx];

    //            if (start == -1) // empty cell
    //                continue;
    //        

    //            for (int i = start; i <= end; i++)
    //            {   
    //                int neighbor = particleArrayIndices[i];
    //                if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule1Distance)
    //                {
    //                    percieved_center += pos[neighbor];
    //                    rule1neighbors++;
    //                }
    //                if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule2Distance)
    //                    c -= (pos[neighbor] - pos[pIdx]);
    //                if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule3Distance)
    //                {
    //                    percieved_velocity += vel1[neighbor];
    //                    rule3neighbors++;
    //                }
    //        
    //            }
    //     
    //        }

    //if (rule1neighbors > 0)
    //    percieved_center /= (float)rule1neighbors;

    //if (rule3neighbors > 0)
    //    percieved_velocity /= (float)rule3neighbors;
    //vel_change += rule1neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos[pIdx]) * rule1Scale;
    //vel_change += c * rule2Scale;
    //vel_change += rule3neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * rule3Scale;
    //newVel += vel_change;
    //float speed = glm::length(newVel);
    //if (speed > maxSpeed)
    //    newVel = glm::normalize(newVel) * maxSpeed;
    //vel2[pIdx] = newVel;
    
}


__global__ void kernUpdateSPHDensitiesAndPressure(int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices, int* particleArrayIndices,
    float* pressures, float* densities, glm::vec3* pos_coherent)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N)
        return;

    glm::vec3 xyz = ((pos_coherent[idx] - gridMin) * inverseCellWidth);
    int cellX = (int)xyz.x;
    int cellY = (int)xyz.y;
    int cellZ = (int)xyz.z;
    float density = d_sphParams.selfDens; // initialize particle density with its self density

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                // bounds check
                if (cellX + dx < 0 || cellX + dx >= gridResolution) continue;
                if (cellY + dy < 0 || cellY + dy >= gridResolution) continue;
                if (cellZ + dz < 0 || cellZ + dz >= gridResolution) continue;

                int gridIdx = gridIndex3Dto1D(cellX + dx, cellY + dy, cellZ + dz, gridResolution);
                int start = gridCellStartIndices[gridIdx];
                int end = gridCellEndIndices[gridIdx];

                if (start == -1) // empty cell
                    continue;


                for (int i = start; i <= end; i++)
                {
                    // int neighbor = particleArrayIndices[i]; // no longer needed
                    float dist2 = glm::dot(pos_coherent[i] - pos_coherent[idx], pos_coherent[i] - pos_coherent[idx]);
                    if (i != idx && dist2 < d_sphParams.h2)
                    {
                        density += d_sphParams.massPoly6Product * __powf(d_sphParams.h2 - dist2, 3.0f);
                    }

                }


            }
    densities[idx] = density;
    pressures[idx] = d_sphParams.gasConst * (density - d_sphParams.restDensity);
    
}

__global__ void kernUpdateSPHForces(int N, int numFluid, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices, int* particleArrayIndices,
    const float* pressures, const float* densities,
    glm::vec3* forces, glm::vec3* pos_coherent, glm::vec3* vel_coherent)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) return;

    // if none fluid particle (boundary representative particle)
    if (particleArrayIndices[idx] >= numFluid) { forces[idx] = glm::vec3(0.0f); return; }
    glm::vec3 xyz = (pos_coherent[idx] - gridMin) * inverseCellWidth;
    int cellX = (int)xyz.x, cellY = (int)xyz.y, cellZ = (int)xyz.z;

    glm::vec3 force(0.0f);

    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++) {
                if (cellX + dx < 0 || cellX + dx >= gridResolution) continue;
                if (cellY + dy < 0 || cellY + dy >= gridResolution) continue;
                if (cellZ + dz < 0 || cellZ + dz >= gridResolution) continue;

                int gridIdx = gridIndex3Dto1D(cellX + dx, cellY + dy, cellZ + dz, gridResolution);
               // int gridIdx = gridIndex3Dto1D(cellX, cellY, cellZ, gridResolution);
                int start = gridCellStartIndices[gridIdx];
                int end = gridCellEndIndices[gridIdx];
                if (start == -1) continue;

                for (int i = start; i <= end; i++) {
                    if (i == idx) continue;

                    glm::vec3 rij = pos_coherent[i] - pos_coherent[idx];   
                    float dist2 = glm::dot(rij, rij);
                    if (dist2 >= d_sphParams.h2 || dist2 < 1e-8f) continue;

                    float invDist = rsqrtf(dist2);
                    float dist = dist2 * invDist;
                    glm::vec3 dir = rij * invDist;
                    float falloff = d_sphParams.h - dist;

                    // Pressure force (repulsive when pressures > 0).
                    // dir is neighbor and spikyGrad < 0, so +dir pushes self away.
                    float pcoeff = d_sphParams.mass * (pressures[idx] + pressures[i])
                        / (2.0f * densities[i])
                        * d_sphParams.spikyGrad * (falloff * falloff);
                    force += dir * pcoeff;

                    // Viscosity force: damps relative velocity (spikyLap > 0).
                    glm::vec3 dv = vel_coherent[i] - vel_coherent[idx];
                    force += d_sphParams.viscosity * d_sphParams.mass * (dv / densities[i]) * d_sphParams.spikyLap * falloff;
                }
            }

    forces[idx] = force;
}

__global__ void kernUpdateSPHPosition(int N, int numFluid, float dt,
    int* particleArrayIndices,
    glm::vec3* pos_coherent, glm::vec3* vel_coherent,
    glm::vec3* forces, float* densities,
    glm::vec3* pos_out, glm::vec3* vel_out)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) return;

    // if boundary representative object skip
    int orig = particleArrayIndices[idx];
    if (orig >= numFluid) return;

    glm::vec3 accel = forces[idx] / densities[idx] + glm::vec3(0.0f,0.0f, - d_sphParams.gravity);
    glm::vec3 vel = vel_coherent[idx] + accel * dt;
    glm::vec3 pos = pos_coherent[idx] + vel * dt;

    // simple box collision against the scene bounds
    const float bound = scene_scale;
    const float damp = -0.5f;
    if (pos.x < -bound) { pos.x = -bound; vel.x *= damp; }
    if (pos.x > bound) { pos.x = bound; vel.x *= damp; }
    if (pos.y < -bound) { pos.y = -bound; vel.y *= damp; }
    if (pos.y > bound) { pos.y = bound; vel.y *= damp; }
    if (pos.z < -bound) { pos.z = -bound; vel.z *= damp; }
    if (pos.z > bound) { pos.z = bound; vel.z *= damp; }

    // scatter back to original (un-sorted) layout
    pos_out[orig] = pos;
    vel_out[orig] = vel;
}
__global__ void kernUpdateBoidsVelNeighborSearchCoherent(int numParticles, int numFluid, int gridResolution, glm::vec3 gridMin, float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices, int* particleArrayIndices,
  glm::vec3 *pos_coherent, glm::vec3 *vel_coherent, glm::vec3 *vel2) {
  // this basically copies the scattered one except that we can directly get vel1 and pos from coherent_vel1 and coherent_pos so we can skip the indirection of getting index from particleArrayIndex first

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= numParticles)
        return;
    
    int pIdx = particleArrayIndices[idx];
    if (pIdx >= numFluid) return;


    glm::vec3 newVel = vel_coherent[idx];
    glm::vec3 xyz = ((pos_coherent[idx] - gridMin) * inverseCellWidth);
    int cellX = (int)xyz.x;
    int cellY = (int)xyz.y;
    int cellZ = (int)xyz.z;
    glm::vec3 vel_change = { 0,0,0 };
    int rule1neighbors = 0;
    int rule3neighbors = 0;
    glm::vec3 percieved_velocity = { 0,0,0 };
    glm::vec3 percieved_center = { 0,0,0 };
    glm::vec3 c = { 0,0,0 };
    glm::vec3 boundaryPush(0.0f);
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                // bounds check
                if (cellX + dx < 0 || cellX + dx >= gridResolution) continue;
                if (cellY + dy < 0 || cellY + dy >= gridResolution) continue;
                if (cellZ + dz < 0 || cellZ + dz >= gridResolution) continue;

                int gridIdx = gridIndex3Dto1D(cellX + dx, cellY + dy, cellZ + dz, gridResolution);
                int start = gridCellStartIndices[gridIdx];
                int end = gridCellEndIndices[gridIdx];

                if (start == -1) // empty cell
                    continue;


                for (int i = start; i <= end; i++)
                {
                   // int neighbor = particleArrayIndices[i]; // no longer needed 
                    bool boundary = particleArrayIndices[i] >= numFluid;
                    if (boundary)
                    {
                        // apply repulsion rule
                        glm::vec3 diff = pos_coherent[idx] - pos_coherent[i];
                        float dist = glm::length(diff);
                        if (dist > 1e-5f && dist < d_boidsParams.boundaryDistance)
                        {
                            float falloff = (d_boidsParams.boundaryDistance - dist) / d_boidsParams.boundaryDistance;
                            boundaryPush += (diff / dist) * falloff;
                        }
                        continue;
                    }
                    if (glm::distance(pos_coherent[idx], pos_coherent[i]) < d_boidsParams.rule1Distance)
                    {
                        percieved_center += pos_coherent[i];
                        rule1neighbors++;
                    }
                    if (glm::distance(pos_coherent[idx], pos_coherent[i]) < d_boidsParams.rule2Distance)
                        c -= (pos_coherent[i] - pos_coherent[idx]);
                    if (glm::distance(pos_coherent[idx], pos_coherent[i]) < d_boidsParams.rule3Distance)
                    {
                        percieved_velocity += vel_coherent[i];
                        rule3neighbors++;
                    }

                }

            }

    if (rule1neighbors > 0) percieved_center /= (float)rule1neighbors;
    if (rule3neighbors > 0) percieved_velocity /= (float)rule3neighbors;

    vel_change += rule1neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos_coherent[idx]) * d_boidsParams.rule1Scale;
    vel_change += c * d_boidsParams.rule2Scale;
    vel_change += rule3neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * d_boidsParams.rule3Scale;
    vel_change += boundaryPush * d_boidsParams.boundaryScale;

    newVel += vel_change;
    float speed = glm::length(newVel);
    if (speed > d_boidsParams.maxSpeed)
        newVel = glm::normalize(newVel) * d_boidsParams.maxSpeed;

    vel2[pIdx] = newVel;


}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Simulator::stepBoidsSimulationNaive(float dt) {
    dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce <<<blocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos <<<blocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    std::swap(dev_vel1, dev_vel2);
    cudaDeviceSynchronize();
}

void Simulator::stepSPHSimulationCoherentGrid(float dt)
{
    dim3 blocksPerGrid((numParticles + blockSize - 1) / blockSize);
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);
    static int frame = 0;
    frame++;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    // Reset start indices
    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (
        gridCellCount,
        dev_gridCellStartIndices,
        -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset start indices: %.3f ms\n", ms);

    // Reset end indices
    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset end indices: %.3f ms\n", ms);

    // Compute indices
    cudaEventRecord(start);
    kernComputeIndices << <blocksPerGrid, blockSize >> > (
        numParticles, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("compute indices: %.3f ms\n", ms);

    // Sort by key
    cudaEventRecord(start);
    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numParticles,
        dev_thrust_particleArrayIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("sort by key: %.3f ms\n", ms);

    // Shuffle positions and velocities
    cudaEventRecord(start);
    kernShufflePosAndVel << <blocksPerGrid, blockSize >> > (
        numParticles, dev_particleArrayIndices,
        dev_pos_coherent, dev_vel_coherent, dev_pos, dev_vel1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("shuffle pos and vel: %.3f ms\n", ms);

    // Identify cell start/end
    cudaEventRecord(start);
    kernIdentifyCellStartEnd << <blocksPerGrid, blockSize >> > (
        numParticles, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("identify cell start/end: %.3f ms\n", ms);

    // Update densities and pressure
    cudaEventRecord(start);
    kernUpdateSPHDensitiesAndPressure << <blocksPerGrid, blockSize >> > (
        numParticles, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pressures, dev_densities, dev_pos_coherent);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("update densities and pressure: %.3f ms\n", ms);

    // Update forces
    cudaEventRecord(start);
    kernUpdateSPHForces << <blocksPerGrid, blockSize >> > (
        numParticles,numFluid, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pressures, dev_densities, dev_forces,
        dev_pos_coherent, dev_vel_coherent);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("update forces: %.3f ms\n", ms);

    // Update SPH position
    cudaEventRecord(start);
    kernUpdateSPHPosition << <blocksPerGrid, blockSize >> > (
        numParticles, numFluid, dt, dev_particleArrayIndices,
        dev_pos_coherent, dev_vel_coherent,
        dev_forces, dev_densities,
        dev_pos, dev_vel1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("update SPH position: %.3f ms\n", ms);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();
}

void Simulator::stepBoidsSimulationScatteredGrid(float dt)
{   
    // Uniform Grid Neighbor search using Thrust sort.
    // In Parallel:
//  // - label each particle with its array index as well as its grid index.
//  //   Use 2x width grids.
//  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
//  //   are welcome to do a performance comparison.
//  // - Naively unroll the loop for finding the start and end indices of each
//  //   cell's data pointers in the array of boid indices
//  // - Perform velocity updates using neighbor search
//  // - Update positions
//  // - Ping-pong buffers as needed
    dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);

    static int frame = 0;
    frame++;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (
        gridCellCount,
        dev_gridCellStartIndices,
        -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset start indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (
        gridCellCount,
        dev_gridCellEndIndices,
        -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset end indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernComputeIndices << <blocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("compute indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("sort: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernIdentifyCellStartEnd << <blocksPerGrid, blockSize >> > (
        numObjects,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("identify cells: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernUpdateVelNeighborSearchScattered << <blocksPerGrid, blockSize >> > (
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos,
        dev_vel1,
        dev_vel2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("neighbor search: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernUpdatePos << <blocksPerGrid, blockSize >> > (
        numObjects,
        dt,
        dev_pos,
        dev_vel2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
    {
        printf("update pos: %.3f ms\n", ms);
        printf("--------------------------------\n");
    }

    std::swap(dev_vel1, dev_vel2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
void Simulator::stepBoidsSimulationCoherentGrid(float dt) {
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. 
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  // - Perform velocity updates using neighbor search
  // - Update positions

    dim3 blocksPerGrid((numParticles + blockSize - 1) / blockSize);
    dim3 blocksPerGridFluid((numFluid + blockSize - 1) / blockSize);
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);

    static int frame = 0;
    frame++;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float ms;

    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (
        gridCellCount,
        dev_gridCellStartIndices,
        -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset start indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (
        gridCellCount,
        dev_gridCellEndIndices,
        -1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("reset end indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernComputeIndices << <blocksPerGrid, blockSize >> > (
        numParticles,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("compute indices: %.3f ms\n", ms);

    cudaEventRecord(start);
    thrust::sort_by_key(
        dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numParticles,
        dev_thrust_particleArrayIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("sort: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernShufflePosAndVel << <blocksPerGrid, blockSize >> > (numParticles,dev_particleArrayIndices, dev_pos_coherent, dev_vel_coherent, dev_pos, dev_vel1); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("coherency shuffle: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernIdentifyCellStartEnd << <blocksPerGrid, blockSize >> > (
        numParticles,
        dev_particleGridIndices,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("identify cells: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernUpdateBoidsVelNeighborSearchCoherent << <blocksPerGrid, blockSize >> > (
        numParticles, numFluid,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        gridCellWidth,
        dev_gridCellStartIndices,
        dev_gridCellEndIndices,
        dev_particleArrayIndices,
        dev_pos_coherent,
        dev_vel_coherent,
        dev_vel2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("neighbor search: %.3f ms\n", ms);

    cudaEventRecord(start);
    kernUpdatePos << <blocksPerGridFluid, blockSize >> > (
        numFluid,
        dt,
        dev_pos,
        dev_vel2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
    {
        printf("update pos: %.3f ms\n", ms);
        printf("--------------------------------\n");
    }

    std::swap(dev_vel1, dev_vel2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
void Simulator::endSimulation() {
    auto release = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
    release(dev_vel1); release(dev_vel2); release(dev_pos);
    release(dev_vel_coherent); release(dev_pos_coherent);
    release(dev_particleArrayIndices); release(dev_particleGridIndices);
    release(dev_gridCellStartIndices); release(dev_gridCellEndIndices);
    release(dev_pressures); release(dev_densities); release(dev_forces);
}

void Simulator::unitTest() {

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
