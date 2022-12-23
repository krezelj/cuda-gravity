#include "gravity_box.cuh";


GravityBox::GravityBox(BodyArray* bodies) : bodies(bodies)
{
	ax = new float[bodies->N];
	ay = new float[bodies->N];

	// TODO See if there is a better way to do this

}

void GravityBox::UpdateSimulation()
{

}

void GravityBox::UpdateAccelerations()
{

}

void GravityBox::UpdateVelocities()
{

}

void GravityBox::UpdatePositions()
{

}