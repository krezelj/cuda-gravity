#include "gravity_box.cuh"
#include <cstdio>

GravityBox::GravityBox(BodyArray* bodies, float delta_t) : bodies(bodies)
{
	this->delta_t = delta_t;
	this->N = &(bodies->N);

	ax = new float[*N];
	ay = new float[*N];
	collision = new int[*N];

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	
	d_acceleration = 0;
	cudaStatus = cudaMalloc((void**)&d_acceleration, *N * 2 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_g_mass = 0;
	cudaStatus = cudaMalloc((void**)&d_g_mass, *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_position_x = 0;
	cudaStatus = cudaMalloc((void**)&d_position_x, *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_position_y = 0;
	cudaStatus = cudaMalloc((void**)&d_position_y, *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_velocity_x = 0;
	cudaStatus = cudaMalloc((void**)&d_velocity_x, *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_velocity_y = 0;
	cudaStatus = cudaMalloc((void**)&d_velocity_y, *N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	d_collision = 0;
	cudaStatus = cudaMalloc((void**)&d_collision, *N * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	// copy data to gpu
	// NOTE: While using GPU mode only in-gpu properties are updates (except x and y)
	// g_mass
	cudaStatus = cudaMemcpy(d_g_mass, bodies->g_mass, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_g_mass);
	}

	// x
	cudaStatus = cudaMemcpy(d_position_x, bodies->x, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_position_x);
	}

	// y
	cudaStatus = cudaMemcpy(d_position_y, bodies->y, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_position_y);
	}

	// vx
	cudaStatus = cudaMemcpy(d_velocity_x, bodies->vx, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_velocity_x);
	}


	// vy
	cudaStatus = cudaMemcpy(d_velocity_y, bodies->vy, *N * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(d_velocity_y);
	}
}

GravityBox::~GravityBox()
{
	cudaFree(d_acceleration);
	cudaFree(d_g_mass);
	cudaFree(d_position_x);
	cudaFree(d_position_y);
	cudaFree(d_velocity_x);
	cudaFree(d_velocity_y);
	cudaFree(d_collision);

	delete[] ax;
	delete[] ay;
	delete[] collision;

	cudaDeviceReset();
}

void GravityBox::UpdateSimulation(GB_MODE mode, int n_steps)
{
	if (mode == GB_USE_CPU)
	{
		for (int step = 0; step < n_steps; step++)
		{
			UpdateAccelerationsCPU();
			UpdateBodiesCPU();
		}
	}
	else 
	{
		UpdateSimulationGPU(n_steps);
	}	
}