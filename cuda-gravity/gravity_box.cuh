#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "body_array.h"

typedef int GB_MODE;
const GB_MODE GB_USE_CPU = 0;
const GB_MODE GB_USE_GPU = 1;

class GravityBox
{
public:

	BodyArray* bodies;
	float delta_t;
	int N;

	GravityBox(BodyArray* bodies, float delta_t);
	~GravityBox();

	void UpdateSimulation(GB_MODE mode, int n_steps);

private:
	
	float* ax;
	float* ay;

	void UpdateAccelerationsCPU();
	void UpdateVelocitiesCPU();
	void UpdatePositionsCPU();

	float* d_acceleration;
	float* d_g_mass;
	float* d_position_x;
	float* d_position_y;
	float* d_velocity_x;
	float* d_velocity_y;

	__host__ void UpdateSimulationGPU(int n_steps);
};