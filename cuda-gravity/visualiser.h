#pragma once

#include "body_array.h"
#include "SFML/Graphics.hpp"

class Visualiser
{
public:

	BodyArray* bodies;
	int* N;

	int size;
	float resolution;

	Visualiser(BodyArray* bodies, int size, float real_size) : 
		bodies(bodies), N(&(bodies->N)), size(size), resolution(real_size / size) {}

	void DisplayParticles(sf::RenderWindow* window);
	void DisplayHeatmap(sf::RenderWindow* window, int grid_size);

private:

	std::pair<float, float> GetWindowPosition(float x, float y);

};