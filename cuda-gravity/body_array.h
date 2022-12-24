#pragma once

#include "utils.cuh"

struct BodyArray
{
    int N;
    float* g_mass;
    float* x;
    float* y;
    float* vx;
    float* vy;
    float sum_g_mass;

    BodyArray(int N, float* masses, float* x, float* y, float* vx, float* vy) : N(N)
    {
        this->g_mass = new float[N];

        this->x = new float[N];
        this->y = new float[N];
        
        this->vx = new float[N];
        this->vy = new float[N];

        sum_g_mass = 0.0f;
        for (int i = 0; i < N; i++)
        {
            this->g_mass[i] = GRAVITATIONAL_CONSTANT * masses[i];
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

    float GetDistanceSquared(int i, int j)
    {
        float x1 = x[i], y1 = y[i];
        float x2 = x[j], y2 = y[j];

        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    }
};