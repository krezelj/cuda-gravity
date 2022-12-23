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

			float sin = sinf((bodies->y[secondBodyIdx] - bodies->y[firstBodyIdx]) / r);
			float cos = sinf((bodies->x[secondBodyIdx] - bodies->x[firstBodyIdx]) / r);
			
			// calculate accelerations for the first body
			float a = bodies->Gm[secondBodyIdx] / r2;
			ax[firstBodyIdx] += a * cos;
			ay[firstBodyIdx] += a * sin;

			if (isnan(ax[firstBodyIdx]))
			{
				float x1 = bodies->x[firstBodyIdx], y1 = bodies->y[firstBodyIdx];
				float x2 = bodies->x[secondBodyIdx], y2 = bodies->y[secondBodyIdx];
				int a = 0;
			}

			// calculate accelerations for the second body
			a = bodies->Gm[firstBodyIdx] / r2;
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