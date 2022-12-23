
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "body_array.h"
#include "gravity_box.cuh"
#include "visualiser.h"
#include <windows.h>  

#include "SFML/Graphics.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

int main()
{
    int N = 500;
    float* masses = new float[N + 1];
    float* x = new float[N + 1];
    float* y = new float[N + 1];
    float* vx = new float[N + 1];
    float* vy = new float[N + 1];

    masses[N] = 6e24f;
    x[N] = y[N] = vx[N] = vy[N] = 0;

    for (int i = 0; i < N; i++)
    {
        masses[i] = ((float)rand() / RAND_MAX + 1) * 1e20;

        float R = ((float)rand() / RAND_MAX + 1) * 4e6;
        float theta = ((float)rand() / RAND_MAX) * 2 * M_PI;

        x[i] = cosf(theta) * R;
        y[i] = sinf(theta) * R;

        float v = ((float)rand() / RAND_MAX + 1) * 6e3;
        vx[i] = cosf(theta + M_PI / 4) * v;
        vy[i] = sinf(theta + M_PI / 4) * v;
    }

    BodyArray bodies(N + 1, masses, x, y, vx, vy);

    GravityBox gb(&bodies, 10.0f);
    Visualiser visualiser(&bodies, 945, 7e6 / 150);

    sf::RenderWindow window(sf::VideoMode(945, 945), "Gravity Box");
    window.setFramerateLimit(240);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        // simulate
        gb.UpdateSimulation();

        // draw
        window.clear(sf::Color::Black);

        visualiser.DisplayHeatmap(&window, 63);

        window.display();
    }

    return 0;
}

