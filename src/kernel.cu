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

// Parameters for the boids algorithm.

#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

// settings for SPH 
#define mass  1.0f
#define restDensity  0.1f
#define gasConst  250.0f
#define viscosity  0.25f
#define PIf   3.1415f
#define tension 0.25f
#define g -5.0f

#define h    4.0f
#define h2   ((h)*(h))
#define h6   ((h2)*(h2)*(h2))
#define h9   ((h6)*(h2)*(h))

#define poly6     (315.0f / (64.0f * PIf * h9))
#define spikyGrad (-45.0f / (PIf * h6))
#define spikyLap (45.0f / (PIf * h6))
#define selfDens (mass * poly6 * h6)
#define massPoly6Product (mass * poly6)

//static constexpr float mass = 1.0f;
//static constexpr float restDensity = 1.0f;
//static constexpr float gasConst = 1.0f;
//static constexpr float viscosity = 1.0f;
//static constexpr float h = 2.0f;
//static constexpr float g = -9.8f;
//static constexpr float tension = 0.0f;
//static constexpr float Pi = 3.1415f;
//
//static constexpr float h2 = h * h;
//static constexpr float h6 = h2 * h2 * h2;
//static constexpr float h9 = h6 * h2 * h;
//
//static constexpr float poly6 = 315.0f / (64.0f * Pi * h9);
//static constexpr float spikyGrad = -45.0f / (Pi * h6);
//static constexpr float spikyLap = 45.0f / (Pi * h6);
//static constexpr float selfDens = mass * poly6 * h6;
//static constexpr float massPoly6Product = mass * poly6;

// Kernel state (pointers are device pointers) 


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

// Function for generating a random vec3.

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

    pos[index] = origin + glm::vec3(i, j, k) * spacing + jitter;
    vel[index] = glm::vec3(0.0f);   // start at rest
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);


  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel_coherent failed!");

  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc particle array indices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc particle grid indices failed!");

  cudaMalloc((void**)&dev_pressures, N * sizeof(float));
  checkCUDAErrorWithLine("cudaMalloc particle pressures failed!");

  cudaMalloc((void**)&dev_densities, N * sizeof(float));
  checkCUDAErrorWithLine("cudaMalloc particle densities failed!");

  cudaMalloc((void**)&dev_forces, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc particle densities failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);


  /*kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");*/

  float spacing = 0.5f * h;                       // ~h/2 -> ~30+ neighbors per particle
  int   perSide = (int)ceilf(cbrtf((float)N));    // smallest cube that holds N
  float blockWidth = (perSide - 1) * spacing;
  glm::vec3 latticeOrigin(
      -0.5f * blockWidth,        // centered in x
      -scene_scale + 10.0f,      // sits ~10 units above the floor (gentle drop)
      -0.5f * blockWidth);       // centered in z

  kernInitLattice << <fullBlocksPerGrid, blockSize >> > (
      numObjects, dev_pos, dev_vel1, perSide, spacing, latticeOrigin);
  checkCUDAErrorWithLine("kernInitLattice failed!");


  //gridCellWidth = 1.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  gridCellWidth = h;
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;
  gridCellCount = gridSideCount * gridSideCount * gridSideCount;

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc cell start indices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc cell end indices failed!");
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  cudaDeviceSynchronize();
}



/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
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
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

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
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule1Distance)
        {
            percieved_center += pos[i];
            neighbors++;
        }
    percieved_center /= float(neighbors);
    return neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos[iSelf]) * rule1Scale;
}
// try to keep a small distance away from other boids
__device__ glm::vec3 rule2(int iSelf, const glm::vec3* pos, const glm::vec3* vel, int start, int end)
{
    glm::vec3 c = { 0,0,0 };
    for (int i = start; i < end; i++)
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule2Distance)
            c -= (pos[i] - pos[iSelf]);
    return c * rule2Scale;
}
// try to match velocity with that of other boids
__device__ glm::vec3 rule3(int iSelf, const glm::vec3* pos, const glm::vec3* vel, int start, int end)
{
    glm::vec3 percieved_velocity = { 0,0,0 };
    int neighbors = 0;
    for (int i = start; i < end; i++)
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule3Distance)
        {
            percieved_velocity += vel[i];
            neighbors++;
        }
    percieved_velocity /= float(neighbors);
    return neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * rule3Scale;
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
        return;
    glm::vec3 newVel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
    float speed = glm::length(newVel);
    if (speed > maxSpeed)
        newVel = glm::normalize(newVel) * maxSpeed;
    vel2[index] = newVel;
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
    int pIdx = particleArrayIndices[idx];
    glm::vec3 newVel = vel1[pIdx];// +computeVelocityChange(N, idx, pos, vel1);
    glm::vec3 xyz = ((pos[pIdx] - gridMin) * inverseCellWidth);
    int cellX = (int)xyz.x;
    int cellY = (int)xyz.y;
    int cellZ = (int)xyz.z;
    glm::vec3 vel_change = { 0,0,0 };
    int rule1neighbors = 0;
    int rule3neighbors = 0;
    glm::vec3 percieved_velocity = { 0,0,0 };
    glm::vec3 percieved_center = { 0,0,0 };
    glm::vec3 c = { 0,0,0 };
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {   
               //  bounds check
                if (cellX + dx < 0 || cellX + dx >= gridResolution) continue;
                if (cellY + dy < 0 || cellY + dy >= gridResolution) continue;
                if (cellZ + dz < 0 || cellZ + dz >= gridResolution) continue;

                int gridIdx = gridIndex3Dto1D(cellX+dx, cellY+dy, cellZ + dz, gridResolution);
                int start = gridCellStartIndices[gridIdx];
                int end = gridCellEndIndices[gridIdx];

                if (start == -1) // empty cell
                    continue;
            

                for (int i = start; i <= end; i++)
                {   
                    int neighbor = particleArrayIndices[i];
                    if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule1Distance)
                    {
                        percieved_center += pos[neighbor];
                        rule1neighbors++;
                    }
                    if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule2Distance)
                        c -= (pos[neighbor] - pos[pIdx]);
                    if (neighbor != pIdx && glm::distance(pos[pIdx], pos[neighbor]) < rule3Distance)
                    {
                        percieved_velocity += vel1[neighbor];
                        rule3neighbors++;
                    }
            
                }
         
            }

    if (rule1neighbors > 0)
        percieved_center /= (float)rule1neighbors;

    if (rule3neighbors > 0)
        percieved_velocity /= (float)rule3neighbors;
    vel_change += rule1neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos[pIdx]) * rule1Scale;
    vel_change += c * rule2Scale;
    vel_change += rule3neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * rule3Scale;
    newVel += vel_change;
    float speed = glm::length(newVel);
    if (speed > maxSpeed)
        newVel = glm::normalize(newVel) * maxSpeed;
    vel2[pIdx] = newVel;
    
}


__global__ void kernUpdateDensitiesAndPressure(int N, int gridResolution, glm::vec3 gridMin,
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
    float density = selfDens; // initialize particle density with its self density

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
                    if (i != idx && dist2 < h2)
                    {
                        density += massPoly6Product * __powf(h2 - dist2, 3.0f);
                    }

                }


            }
    densities[idx] = density;
    pressures[idx] = gasConst * (density - restDensity);
    
}

__global__ void kernUpdateForces(int N, int gridResolution, glm::vec3 gridMin,
    float inverseCellWidth, float cellWidth,
    int* gridCellStartIndices, int* gridCellEndIndices, int* particleArrayIndices,
    const float* pressures, const float* densities,
    glm::vec3* forces, glm::vec3* pos_coherent, glm::vec3* vel_coherent)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) return;

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
                    if (dist2 >= h2 || dist2 < 1e-8f) continue;            

                    float dist = sqrtf(dist2);
                    glm::vec3 dir = rij / dist;
                    float falloff = h - dist;

                    // Pressure force (repulsive when pressures > 0).
                    // dir is neighbor and spikyGrad < 0, so +dir pushes self away.
                    float pcoeff = mass * (pressures[idx] + pressures[i])
                        / (2.0f * densities[i])
                        * spikyGrad * (falloff * falloff);
                    force += dir * pcoeff;

                    // Viscosity force: damps relative velocity (spikyLap > 0).
                    glm::vec3 dv = vel_coherent[i] - vel_coherent[idx];
                    force += viscosity * mass * (dv / densities[i]) * spikyLap * falloff;
                }
            }

    forces[idx] = force;
}

__global__ void kernUpdateSPHPosition(int N, float dt,
    int* particleArrayIndices,
    glm::vec3* pos_coherent, glm::vec3* vel_coherent,
    glm::vec3* forces, float* densities,
    glm::vec3* pos_out, glm::vec3* vel_out)
{
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= N) return;

    glm::vec3 accel = forces[idx] / densities[idx] + glm::vec3(0.0f,0.0f, - g);
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
    int orig = particleArrayIndices[idx];
    pos_out[orig] = pos;
    vel_out[orig] = vel;
}
__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices, int* particleArrayIndices,
  glm::vec3 *pos_coherent, glm::vec3 *vel_coherent, glm::vec3 *vel2) {
  // this basically copies the scattered one except that we can directly get vel1 and pos from coherent_vel1 and coherent_pos so we can skip the indirection of getting index from particleArrayIndex first

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (idx >= N)
        return;
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
                    if (i != idx && glm::distance(pos_coherent[idx], pos_coherent[i]) < rule1Distance)
                    {
                        percieved_center += pos_coherent[i];
                        rule1neighbors++;
                    }
                    if (i != idx && glm::distance(pos_coherent[idx], pos_coherent[i]) < rule2Distance)
                        c -= (pos_coherent[i] - pos_coherent[idx]);
                    if (i != idx && glm::distance(pos_coherent[idx], pos_coherent[i]) < rule3Distance)
                    {
                        percieved_velocity += vel_coherent[i];
                        rule3neighbors++;
                    }

                }

            }

    if (rule1neighbors > 0)
        percieved_center /= (float)rule1neighbors;

    if (rule3neighbors > 0)
        percieved_velocity /= (float)rule3neighbors;
    vel_change += rule1neighbors == 0 ? glm::vec3(0.0f) : (percieved_center - pos_coherent[idx]) * rule1Scale;
    vel_change += c * rule2Scale;
    vel_change += rule3neighbors == 0 ? glm::vec3(0.0f) : percieved_velocity * rule3Scale;
    newVel += vel_change;
    float speed = glm::length(newVel);
    if (speed > maxSpeed)
        newVel = glm::normalize(newVel) * maxSpeed;
    int pIdx = particleArrayIndices[idx];
    vel2[pIdx] = newVel;


}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce <<<blocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos <<<blocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);
    std::swap(dev_vel1, dev_vel2);
    cudaDeviceSynchronize();
}

void Boids::stepSPHSimulationCoherentGrid(float dt)
{
    dim3 blocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 cellBlocks((gridCellCount + blockSize - 1) / blockSize);

    kernResetIntBuffer << <cellBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <cellBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernComputeIndices << <blocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    thrust::sort_by_key(dev_thrust_particleGridIndices,
        dev_thrust_particleGridIndices + numObjects,
        dev_thrust_particleArrayIndices);

    kernShufflePosAndVel << <blocksPerGrid, blockSize >> > (
        numObjects, dev_particleArrayIndices,
        dev_pos_coherent, dev_vel_coherent, dev_pos, dev_vel1);

    kernIdentifyCellStartEnd << <blocksPerGrid, blockSize >> > (
        numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    kernUpdateDensitiesAndPressure << <blocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pressures, dev_densities, dev_pos_coherent);

    kernUpdateForces << <blocksPerGrid, blockSize >> > (
        numObjects, gridSideCount, gridMinimum,
        gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pressures, dev_densities, dev_forces,
        dev_pos_coherent, dev_vel_coherent);

    kernUpdateSPHPosition << <blocksPerGrid, blockSize >> > (
        numObjects, dt, dev_particleArrayIndices,
        dev_pos_coherent, dev_vel_coherent,
        dev_forces, dev_densities,
        dev_pos, dev_vel1);   

    cudaDeviceSynchronize();
}

void Boids::stepSimulationScatteredGrid(float dt)
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
void Boids::stepSimulationCoherentGrid(float dt) {
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
    kernShufflePosAndVel << <blocksPerGrid, blockSize >> > (numObjects,dev_particleArrayIndices, dev_pos_coherent, dev_vel_coherent, dev_pos, dev_vel1); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    if (frame % 10 == 0)
        printf("coherency shuffle: %.3f ms\n", ms);

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
    kernUpdateVelNeighborSearchCoherent << <blocksPerGrid, blockSize >> > (
        numObjects,
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

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel_coherent);
  cudaFree(dev_forces);
  cudaFree(dev_densities);
  cudaFree(dev_pressures);

}

void Boids::unitTest() {

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
