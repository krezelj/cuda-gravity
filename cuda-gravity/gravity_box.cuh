#pragma once

#include "body_array.cuh";

const int GB_USE_CPU = 0;
const int GB_USE_GPU = 1;

class GravityBox
{
public:

	BodyArray* bodies;
	float delta_t;

	GravityBox(BodyArray* bodies);

	void UpdateSimulation();

private:

	float* ax;
	float* ay;

	void UpdateAccelerations();
	void UpdateVelocities();
	void UpdatePositions();

};