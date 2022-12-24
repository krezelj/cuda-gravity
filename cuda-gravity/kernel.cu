
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SFML/Graphics.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#include <windows.h>  
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>

#include "body_array.h"
#include "gravity_box.cuh"
#include "visualiser.h"


int main()
{
    int N = 1000;
    float* masses = new float[N + 2];
    float* x = new float[N + 2];
    float* y = new float[N + 2];
    float* vx = new float[N + 2];
    float* vy = new float[N + 2];

    masses[N] = 6e24f;
    x[N] = y[N] = vx[N] = vy[N] = 0;

    masses[N + 1] = 1e22;
    x[N + 1] = 20e6;
    y[N + 1] = 0;
    vx[N + 1] = 0;
    vy[N + 1] = -4.47e3f;

    /*std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<float> d{ 1e5,1e3 };*/

    /*std::default_random_engine generator;
    std::normal_distribution<float> distribution(1, 1e3);*/
    for (int i = 0; i < N; i++)
    {
        float t = ((float)rand() / RAND_MAX);
        if (t < 0.9)
        {
            masses[i] = 1e4f;
        }
        else if (t < 0.99)
        {
            masses[i] = 1e8f;
        }
        else
        {
            masses[i] = 1e12f;
        }
        // masses[i] = std::max(1.0f, d(gen));

        float R = ((float)rand() / RAND_MAX) * 25e6 + 5e6;
        float theta = ((float)rand() / RAND_MAX) * 2 * M_PI;

        x[i] = cosf(theta) * R;
        y[i] = sinf(theta) * R;

        float v = sqrtf(GRAVITATIONAL_CONSTANT * 6e24 / R);
        // v *= ((float)rand() / RAND_MAX) * 0.1f + 0.95f;
        vx[i] = cosf(3 * M_PI / 2 + theta) * v;
        vy[i] = sinf(3 * M_PI / 2 + theta) * v;
    }

    BodyArray bodies(N + 2, masses, x, y, vx, vy);

    float delta_t = 1.0f;
    GravityBox gb(&bodies, delta_t);
    Visualiser visualiser(&bodies, 864, 80e6);

    sf::RenderWindow window(sf::VideoMode(864, 864), "Gravity Box");
    window.setFramerateLimit(60);


    std::chrono::steady_clock::time_point start, stop;
    std::chrono::milliseconds duration_update;
    std::chrono::milliseconds duration_display;

    int iterations = 0;
    int steps_per_update = 50;
    while (window.isOpen() && iterations < 500)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        // simulate
        start = std::chrono::high_resolution_clock::now();

        gb.UpdateSimulation(GB_USE_GPU, steps_per_update);
        iterations++;

        stop = std::chrono::high_resolution_clock::now();
        duration_update = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        
        // draw
        start = std::chrono::high_resolution_clock::now();
        window.clear(sf::Color::Black);
        visualiser.DisplayHeatmap(&window, 288); // 2 * 2 * 2 * 2 * 2 * 3 * 3 * 3
        // visualiser.DisplayParticles(&window);
        window.display();

        stop = std::chrono::high_resolution_clock::now();
        duration_display = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);


        int total_miliseconds = duration_update.count() + duration_display.count();
        system("cls");
        std::cout << "Time to update: " << duration_update.count() << "ms" << std::endl;
        std::cout << "Time to display: " << duration_display.count() << "ms" << std::endl;
        std::cout << "FPS: " << 1000.0f / total_miliseconds << std::endl;
        std::cout << "UPS: " << steps_per_update * 1000.0f / duration_update.count() << std::endl;
        std::cout << "Simulated Bodies: " << bodies.N << std::endl;
        std::cout << "Total Iterations: " << iterations << std::endl;
        std::cout << "Total Time Simulated: " << iterations * delta_t * steps_per_update << "s" << std::endl;
    }

    return 0;
}

