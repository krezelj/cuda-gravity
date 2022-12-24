
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
    int N = 10000;
    float* masses = new float[N + 2];
    float* x = new float[N + 2];
    float* y = new float[N + 2];
    float* vx = new float[N + 2];
    float* vy = new float[N + 2];

    masses[N] = 1.0f;
    x[N] = y[N] = vx[N] = vy[N] = 0;

    masses[N + 1] = 1.0f;
    x[N + 1] = 5e6;
    y[N + 1] = 10e6;
    vx[N + 1] = 0;
    vy[N + 1] = -1e5;

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::normal_distribution<float> d{ 1e2,1e1 };

    /*std::default_random_engine generator;
    std::normal_distribution<float> distribution(1, 1e3);*/
    for (int i = 0; i < N; i++)
    {
        masses[i] = std::max(1.0f, d(gen));

        float R = ((float)rand() / RAND_MAX) * 10e6 + 5e6;
        float theta = ((float)rand() / RAND_MAX) * 2 * M_PI;

        x[i] = cosf(theta)* R;
        y[i] = sinf(theta)* R;

        float v = ((float)rand() / RAND_MAX + 1.0f) * 1e1;
        vx[i] = cosf(theta + M_PI / 4)* v;
        vy[i] = sinf(theta + M_PI / 4)* v;
    }

    BodyArray bodies(N + 2, masses, x, y, vx, vy);

    GravityBox gb(&bodies, 5.0f);
    Visualiser visualiser(&bodies, 864, 7e6 / 150);

    sf::RenderWindow window(sf::VideoMode(864, 864), "Gravity Box");
    window.setFramerateLimit(60);


    std::chrono::steady_clock::time_point start, stop;
    std::chrono::milliseconds duration_update;
    std::chrono::milliseconds duration_display;

    int iters = 0;
    while (window.isOpen() && iters++ < 1e3)
    {

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        
        // simulate
        start = std::chrono::high_resolution_clock::now();

        gb.UpdateSimulation(GB_USE_GPU, 3);

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

        system("cls");
        std::cout << "Iterations: " << iters << std::endl;
        std::cout << "Time to update: " << duration_update.count() << "ms" << std::endl;
        std::cout << "Time to display: " << duration_display.count() << "ms" << std::endl;

        int total_miliseconds = duration_update.count() + duration_display.count();
        std::cout << "FPS: " << 1000.0f / total_miliseconds << std::endl;
    }

    window.close();

    return 0;
}

