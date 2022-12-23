#include "gravity_box.cuh";
#include <math.h>

GravityBox::GravityBox(BodyArray* bodies, float delta_t) : bodies(bodies)
{
	this->delta_t = delta_t;
	this->N = bodies->N;

	ax = new float[N];
	ay = new float[N];
}

void GravityBox::UpdateSimulation()
{
	UpdateAccelerations();
	UpdateVelocities();
	UpdatePositions();
}

void GravityBox::UpdateAccelerations()
{
	// TODO See if there is a better way to do this
	for (int i = 0; i < N; i++)
	{
		ax[i] = 0;
		ay[i] = 0;
	}

	for (int firstBodyIdx = 0; firstBodyIdx < N; firstBodyIdx++)
	{
		// since calculations are symmetric and calculating 
		// square of distance, cos and sin are all expensive operations we will only calculate them once per pair
		for (int secondBodyIdx = firstBodyIdx + 1; secondBodyIdx < N; secondBodyIdx++)
		{
			float r2 = bodies->GetDistanceSquared(firstBodyIdx, secondBodyIdx);
			float r = sqrtf(r2);

			float sin = sinf(r / (bodies->y[secondBodyIdx] - bodies->y[firstBodyIdx]));
			float cos = sinf(r / (bodies->x[secondBodyIdx] - bodies->x[firstBodyIdx]));
			
			// calculate accelerations for the first body
			float a = bodies->Gm[secondBodyIdx] / r2;
			ax[firstBodyIdx] += a * cos;
			ay[firstBodyIdx] += a * sin;

			// calculate accelerations for the second body
			a = bodies->Gm[firstBodyIdx] / r2;
			ax[firstBodyIdx] -= a * cos; // minus instead of plus because trig functions are reversed
			ay[firstBodyIdx] -= a * sin;
		}
	}
}

void GravityBox::UpdateVelocities()
{
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		bodies->vx[bodyIdx] += ax[bodyIdx] * delta_t;
		bodies->vy[bodyIdx] += ay[bodyIdx] * delta_t;
	}
}

void GravityBox::UpdatePositions()
{
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		bodies->x[bodyIdx] += bodies->vx[bodyIdx] * delta_t;
		bodies->y[bodyIdx] += bodies->vy[bodyIdx] * delta_t;
	}
}