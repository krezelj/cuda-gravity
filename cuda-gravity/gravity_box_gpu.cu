#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gravity_box.cuh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cooperative_groups.h>

#define CHUNK_SIZE 8
#define THREADS_PER_BLOCK 1024

__global__ void UpdateBodiesGPU(float* position_x, float* position_y, float* velocity_x, float* velocity_y, float* acceleration_data, float delta_t, int N);
__global__ void UpdateAccelerationGPU(float* g_masses, float* position_x, float* position_y, float* acceleration_data, int N);
__global__ void CheckCollisions(float* position_x, float* position_y, float* g_mass, int* collision, int K);

__host__ void GravityBox::UpdateSimulationGPU(int n_steps)
{
	dim3 gridDim((unsigned int)ceilf((float)*N / CHUNK_SIZE), (unsigned int)ceilf((float)*N / CHUNK_SIZE));
	dim3 blockDim(CHUNK_SIZE, CHUNK_SIZE);
	cudaError_t cudaStatus;

	// push gmass positions and velocities
	// TODO: Only do that if there were collisions (because only then it needs updating)
	// or do collision handling (apart from index moving) on gpu
	cudaStatus = cudaMemcpy(d_g_mass, bodies->g_mass, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_position_x, bodies->x, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_position_y, bodies->y, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_velocity_x, bodies->vx, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(d_velocity_y, bodies->vy, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	for (int step = 0; step < n_steps; step++)
	{
		cudaMemset(d_acceleration, 0, *N * 2 * sizeof(float));
		UpdateAccelerationGPU<<<gridDim, blockDim>>>(d_g_mass, d_position_x, d_position_y, d_acceleration, *N);
		UpdateBodiesGPU<<<(unsigned int)ceilf((float)*N / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
			(d_position_x, d_position_y, d_velocity_x, d_velocity_y, d_acceleration, delta_t, *N);
	}

	int K = *N * (*N - 1) * 0.5;
	cudaMemset(d_collision, -1, *N * sizeof(int));
	CheckCollisions<<<(unsigned int)ceilf((float)K / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
		(d_position_x, d_position_y, d_g_mass, d_collision, K);

	cudaDeviceSynchronize();

	// get positions back
	cudaStatus = cudaMemcpy(bodies->x, d_position_x, *N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(bodies->y, d_position_y, *N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// TODO Only get velocities back if there is at least one collision
	// get velocities back
	cudaStatus = cudaMemcpy(bodies->vx, d_velocity_x, *N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(bodies->vy, d_velocity_y, *N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	// get and handle collisions
	cudaStatus = cudaMemcpy(collision, d_collision, *N * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	bodies->HandleCollisions(collision);
}

__global__ void CheckCollisions(float* position_x, float* position_y, float* g_mass, int* collision, int K)
{
	int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (globalIdx >= K)
	{
		return;
	}

	int thisBodyIdx = floorf((sqrtf(1 + 8 * globalIdx) + 1) * 0.5f);
	int otherBodyIdx = globalIdx - (thisBodyIdx) * (thisBodyIdx - 1) * 0.5f;

	float dx = position_x[otherBodyIdx] - position_x[thisBodyIdx];
	float dy = position_y[otherBodyIdx] - position_y[thisBodyIdx];

	float r2 = dx * dx + dy * dy;

	float m1 = g_mass[thisBodyIdx];
	float m2 = g_mass[otherBodyIdx];
	float collision_distance = fmaxf(sqrtf(m1 / GRAVITATIONAL_CONSTANT), sqrtf(m2 / GRAVITATIONAL_CONSTANT));

	if (r2 < collision_distance)
	{
		collision[thisBodyIdx] = otherBodyIdx;
		collision[otherBodyIdx] = thisBodyIdx;
	}
}

__global__ void UpdateBodiesGPU(float* position_x, float* position_y, float* velocity_x, float* velocity_y, float* acceleration_data, float delta_t, int N)
{
	int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (globalIdx >= N)
	{
		return;
	}

	velocity_x[globalIdx] += acceleration_data[2 * globalIdx] * delta_t;
	velocity_y[globalIdx] += acceleration_data[2 * globalIdx + 1] * delta_t;

	position_x[globalIdx] += velocity_x[globalIdx] * delta_t;
	position_y[globalIdx] += velocity_y[globalIdx] * delta_t;
}

__global__ void UpdateAccelerationGPU(float* g_masses, float* position_x, float* position_y, float* acceleration_data, int N)
{
	int globalIdxX = threadIdx.x + blockIdx.x * blockDim.x;
	int globalIdxY = threadIdx.y + blockIdx.y * blockDim.y;

	if (globalIdxX >= N && globalIdxY >= N)
	{
		return;
	}

	__shared__ float s_other_body_gmass[CHUNK_SIZE];
	__shared__ float s_other_body_x[CHUNK_SIZE];
	__shared__ float s_other_body_y[CHUNK_SIZE];

	__shared__ float s_this_body_x[CHUNK_SIZE];
	__shared__ float s_this_body_y[CHUNK_SIZE];

	__shared__ float s_acceleration[CHUNK_SIZE][CHUNK_SIZE][2];

	// fetch data about body
	if (threadIdx.x == threadIdx.y)
	{
		s_other_body_gmass[threadIdx.x] = g_masses[globalIdxX];
		s_other_body_x[threadIdx.x] = position_x[globalIdxX];
		s_other_body_y[threadIdx.x] = position_y[globalIdxX];

		s_this_body_x[threadIdx.y] = position_x[globalIdxY];
		s_this_body_y[threadIdx.y] = position_y[globalIdxY];
	}

	__syncthreads();

	if (globalIdxX >= N || globalIdxY >= N)
	{
		return;
	}

	// threads on global diagonal correspond to calculations on one body
	if (globalIdxX == globalIdxY)
	{
		s_acceleration[threadIdx.x][threadIdx.y][0] = 0;
		s_acceleration[threadIdx.x][threadIdx.y][1] = 0;
	}
	else 
	{
		float dx = s_other_body_x[threadIdx.x] - s_this_body_x[threadIdx.y];
		float dy = s_other_body_y[threadIdx.x] - s_this_body_y[threadIdx.y];
		float r2 = dx * dx + dy * dy;
		float r = sqrtf(r2);

		float a = s_other_body_gmass[threadIdx.x] / r2;
		s_acceleration[threadIdx.x][threadIdx.y][0] = a * dx / r;
		s_acceleration[threadIdx.x][threadIdx.y][1] = a * dy / r;
	}

	__syncthreads();

	//// in-block parallel sum
	if (threadIdx.x < 4 && globalIdxX + 4 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][0] += s_acceleration[threadIdx.x + 4][threadIdx.y][0];
		s_acceleration[threadIdx.x][threadIdx.y][1] += s_acceleration[threadIdx.x + 4][threadIdx.y][1];
	}
	if (threadIdx.x < 2 && globalIdxX + 2 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][0] += s_acceleration[threadIdx.x + 2][threadIdx.y][0];
		s_acceleration[threadIdx.x][threadIdx.y][1] += s_acceleration[threadIdx.x + 2][threadIdx.y][1];
	}
	if (threadIdx.x < 1 && globalIdxX + 1 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][0] += s_acceleration[threadIdx.x + 1][threadIdx.y][0];
		s_acceleration[threadIdx.x][threadIdx.y][1] += s_acceleration[threadIdx.x + 1][threadIdx.y][1];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		// int data_idx = globalIdxY * gridDim.x * blockDim.z + blockIdx.x * blockDim.z + threadIdx.z;
		// acceleration_data[data_idx] = s_acceleration[0][threadIdx.y][threadIdx.z];

		// atomicAdd(&acceleration_data[2 * globalIdxY], s_acceleration[0][threadIdx.y][0]);
		// atomicAdd(&acceleration_data[2 * globalIdxY + 1], s_acceleration[0][threadIdx.y][1]);
	}
}