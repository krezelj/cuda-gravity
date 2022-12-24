#include "body_array.h"
#include <cstdio>

float BodyArray::GetDistanceSquared(int i, int j)
{
    float x1 = x[i], y1 = y[i];
    float x2 = x[j], y2 = y[j];

    return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

void BodyArray::HandleCollisions(int* collision)
{
    bool* to_discard = new bool[N];
    std::memset(to_discard, false, N * sizeof(bool));

    for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
    {
        int otherBodyIdx = collision[bodyIdx];
        if (otherBodyIdx == -1 || to_discard[bodyIdx] == true || to_discard[otherBodyIdx] == true)
        {
            continue;
        }

        float m1 = g_mass[bodyIdx];
        float m2 = g_mass[otherBodyIdx];

        float x1 = x[bodyIdx];
        float y1 = y[bodyIdx];

        float x2 = x[otherBodyIdx];
        float y2 = y[otherBodyIdx];

        float vx1 = vx[bodyIdx];
        float vy1 = vy[bodyIdx];

        float vx2 = vx[otherBodyIdx];
        float vy2 = vy[otherBodyIdx];

        float inverse_sum_of_masses = 1 / (m1 + m2);

        // make this body the new body

        x[bodyIdx] = (x1 * m1 + x2 * m2) * inverse_sum_of_masses;
        y[bodyIdx] = (y1 * m1 + y2 * m2) * inverse_sum_of_masses;

        vx[bodyIdx] = (m1 * vx1 + m2 * vx2) * inverse_sum_of_masses;
        vy[bodyIdx] = (m1 * vy1 + m2 * vy2) * inverse_sum_of_masses;

        g_mass[bodyIdx] = m1 + m2;

        // discard the other body
        // discarding means copying last index to its index and reducing N by 1 essentialy forgetting about it
        to_discard[otherBodyIdx] = true;
    }

    // discard bodies by placing them at the end
    int lastIdx = N - 1;
    for (int bodyIdx = 0; bodyIdx <= lastIdx; bodyIdx++)
    {
        if (to_discard[bodyIdx] == false)
        {
            continue;
        }

        while (to_discard[lastIdx] && lastIdx > bodyIdx)
        {
            lastIdx--;
        }

        x[bodyIdx] = x[lastIdx];
        y[bodyIdx] = y[lastIdx];

        vx[bodyIdx] = vx[lastIdx];
        vy[bodyIdx] = vy[lastIdx];

        g_mass[bodyIdx] = g_mass[lastIdx];

        lastIdx--;
    }
    N = lastIdx + 1;
    delete[] to_discard;
}