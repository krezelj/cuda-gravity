#pragma once

#include "utils.cuh"
#include <cstring>

class BodyArray
{
public:

    int N;
    float* g_mass;
    float* x;
    float* y;
    float* vx;
    float* vy;
    float sum_g_mass;

    BodyArray(int N, float* masses, float* x, float* y, float* vx, float* vy) : N(N)
    {
        g_mass = new float[N];

        this->x = new float[N];
        this->y = new float[N];
        
        this->vx = new float[N];
        this->vy = new float[N];

        sum_g_mass = 0.0f;
        for (int i = 0; i < N; i++)
        {
            g_mass[i] = GRAVITATIONAL_CONSTANT * masses[i];
            sum_g_mass += g_mass[i];

            this->x[i] = x[i];
            this->y[i] = y[i];
            
            this->vx[i] = vx[i];
            this->vy[i] = vy[i];
        }
    }

    ~BodyArray()
    {
        delete[] g_mass;

        delete[] x;
        delete[] y;
        
        delete[] vx;
        delete[] vy;
    }

    float GetDistanceSquared(int i, int j);

    void HandleCollisions(int* collision);
};