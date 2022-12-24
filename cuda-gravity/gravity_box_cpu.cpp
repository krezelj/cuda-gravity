#include "gravity_box.cuh"
#include <math.h>


void GravityBox::UpdateAccelerationsCPU()
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

			float dx = bodies->x[secondBodyIdx] - bodies->x[firstBodyIdx];
			float dy = bodies->y[secondBodyIdx] - bodies->y[firstBodyIdx];

			float sin = dy / r;
			float cos = dx / r;
			
			// calculate accelerations for the first body
			float a = bodies->g_mass[secondBodyIdx] / r2;
			ax[firstBodyIdx] += a * cos;
			ay[firstBodyIdx] += a * sin;

			// calculate accelerations for the second body
			a = bodies->g_mass[firstBodyIdx] / r2;
			ax[secondBodyIdx] -= a * cos; // minus instead of plus because trig functions are reversed
			ay[secondBodyIdx] -= a * sin;
		}
	}
}

void GravityBox::UpdateVelocitiesCPU()
{
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		bodies->vx[bodyIdx] += ax[bodyIdx] * delta_t;
		bodies->vy[bodyIdx] += ay[bodyIdx] * delta_t;
	}
}

void GravityBox::UpdatePositionsCPU()
{
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		bodies->x[bodyIdx] += bodies->vx[bodyIdx] * delta_t;
		bodies->y[bodyIdx] += bodies->vy[bodyIdx] * delta_t;
	}
}