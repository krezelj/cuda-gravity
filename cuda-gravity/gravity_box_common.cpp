#include "gravity_box.cuh"

GravityBox::GravityBox(BodyArray* bodies, float delta_t) : bodies(bodies)
{
	this->delta_t = delta_t;
	this->N = bodies->N;

	ax = new float[N];
	ay = new float[N];
}

void GravityBox::UpdateSimulation(GB_MODE mode = GB_USE_CPU)
{
	if (mode == GB_USE_CPU)
	{
		UpdateAccelerationsCPU();
		UpdateVelocitiesCPU();
		UpdatePositionsCPU();
	}
	else 
	{
		UpdateAccelerationsGPU();
		UpdateVelocitiesGPU();
		UpdatePositionsGPU();
	}	
}