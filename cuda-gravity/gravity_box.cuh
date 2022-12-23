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

	void UpdateSimulation(GB_MODE mode);

private:

	float* ax;
	float* ay;

	void UpdateAccelerationsCPU();
	void UpdateVelocitiesCPU();
	void UpdatePositionsCPU();

	__host__ void UpdateAccelerationsGPU();
	__host__ void UpdateVelocitiesGPU();
	__host__ void UpdatePositionsGPU();

};