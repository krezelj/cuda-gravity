#pragma once

#include "utils.cuh";

struct BodyArray
{
    int N;
    float* Gm;
    float* x;
    float* y;
    float* vx;
    float* vy;

    BodyArray(int N, float* masses, float* x, float* y, float* vx, float* vy) : N(N)
    {
        this->Gm = new float[N];

        this->x = new float[N];
        this->y = new float[N];
        
        this->vx = new float[N];
        this->vy = new float[N];

        for (int i = 0; i < N; i++)
        {
            this->Gm[i] = G * masses[i];

            this->x[i] = x[i];
            this->y[i] = y[i];
            
            this->vx[i] = vx[i];
            this->vy[i] = vy[i];
        }
    }

    ~BodyArray()
    {
        delete[] Gm;

        delete[] x;
        delete[] y;
        
        delete[] vx;
        delete[] vy;
    }

    float GetDistanceSquared(int i, int j)
    {
        float x1 = x[i], y1 = y[i];
        float x2 = x[j], y2 = y[j];

        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y2);
    }
};