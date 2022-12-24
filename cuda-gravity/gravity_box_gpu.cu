#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gravity_box.cuh"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cooperative_groups.h>

#define CHUNK_SIZE 16
#define THREADS_PER_BLOCK 1024

__global__ void UpdateBodiesGPU(float* position_x, float* position_y, float* velocity_x, float* velocity_y, float* acceleration_data, float delta_t, int N);
__global__ void UpdateAccelerationGPU(float* g_masses, float* position_x, float* position_y, float* acceleration_data, int N);

__host__ void GravityBox::UpdateSimulationGPU(int n_steps)
{
	dim3 gridDim((unsigned int)ceilf((float)N / CHUNK_SIZE), (unsigned int)ceilf((float)N / CHUNK_SIZE));
	dim3 blockDim(CHUNK_SIZE, CHUNK_SIZE, 2);
	cudaError_t cudaStatus;

	for (int step = 0; step < n_steps; step++)
	{
		cudaMemset(d_acceleration, 0, N * 2 * sizeof(float));
		UpdateAccelerationGPU << <gridDim, blockDim >> > (d_g_mass, d_position_x, d_position_y, d_acceleration, N);
		UpdateBodiesGPU << <(unsigned int)ceilf((float)N / THREADS_PER_BLOCK), THREADS_PER_BLOCK >> >
			(d_position_x, d_position_y, d_velocity_x, d_velocity_y, d_acceleration, delta_t, N);
	}

	// cudaDeviceSynchronize();

	// get position back
	cudaStatus = cudaMemcpy(bodies->x, d_position_x, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	cudaStatus = cudaMemcpy(bodies->y, d_position_y, N * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	//// system("cls");
	//for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	//{
	//	printf("%f, %f\n", bodies->x[bodyIdx], bodies->y[bodyIdx]);
	//}

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
	int globalIdxZ = threadIdx.z;

	if (globalIdxX >= N && globalIdxY >= N)
	{
		return;
	}

	__shared__ float s_other_body_gmass[CHUNK_SIZE];
	__shared__ float s_other_body_x[CHUNK_SIZE];
	__shared__ float s_other_body_y[CHUNK_SIZE];

	// __shared__ float this_body_gmass[CHUNK_SIZE];
	__shared__ float s_this_body_x[CHUNK_SIZE];
	__shared__ float s_this_body_y[CHUNK_SIZE];

	__shared__ float s_acceleration[CHUNK_SIZE][CHUNK_SIZE][2];

	// fetch data about body
	if (threadIdx.x == threadIdx.y)
	{
		if (threadIdx.z == 0)
		{
			s_other_body_gmass[threadIdx.x] = g_masses[globalIdxX];
			s_other_body_x[threadIdx.x] = position_x[globalIdxX];
			s_other_body_y[threadIdx.x] = position_y[globalIdxX];
		}
		else // 8 * 32 threads per layer so warps do not overlap different layers
		{
			// this_body_gmass[threadIdx.y] = g_masses[globalIdxY];
			s_this_body_x[threadIdx.y] = position_x[globalIdxY];
			s_this_body_y[threadIdx.y] = position_y[globalIdxY];
		}
	}

	__syncthreads();

	if (globalIdxX >= N || globalIdxY >= N)
	{
		return;
	}

	// threads on global diagonal correspond to calculations on one body
	if (globalIdxX == globalIdxY)
	{
		s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] = 0;
	}
	else 
	{
		// calculate distance
		
		float dx = s_other_body_x[threadIdx.x] - s_this_body_x[threadIdx.y];
		float dy = s_other_body_y[threadIdx.x] - s_this_body_y[threadIdx.y];
		float r2 = dx * dx + dy * dy;
		float r = sqrtf(r2);

		float a = s_other_body_gmass[threadIdx.x] / r2;
		// s_acceleration[threadIdx.x][threadIdx.y][0] = a * dx / r;
		// s_acceleration[threadIdx.x][threadIdx.y][1] = a * dy / r;

		if (threadIdx.z == 0)
		{
			// ax
			s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] = a * dx / r;
		}
		else
		{
			// ay
			s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] = a * dy / r;
		}
	}

	__syncthreads();

	// in-block parallel sum
	if (threadIdx.x < 8 && globalIdxX + 8 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] += s_acceleration[threadIdx.x + 8][threadIdx.y][threadIdx.z];
	}
	if (threadIdx.x < 4 && globalIdxX + 4 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] += s_acceleration[threadIdx.x + 4][threadIdx.y][threadIdx.z];
	}
	if (threadIdx.x < 2 && globalIdxX + 2 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] += s_acceleration[threadIdx.x + 2][threadIdx.y][threadIdx.z];
	}
	if (threadIdx.x < 1 && globalIdxX + 1 < N)
	{
		s_acceleration[threadIdx.x][threadIdx.y][threadIdx.z] += s_acceleration[threadIdx.x + 1][threadIdx.y][threadIdx.z];
	}

	__syncthreads();

	if (threadIdx.x == 0)
	{
		int data_idx = globalIdxY * gridDim.x * blockDim.z + blockIdx.x * blockDim.z + threadIdx.z;

		// acceleration_data[data_idx] = s_acceleration[0][threadIdx.y][threadIdx.z];
		atomicAdd(&acceleration_data[2 * globalIdxY + threadIdx.z], s_acceleration[0][threadIdx.y][threadIdx.z]);
	}
}