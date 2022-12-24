#include "visualiser.h"
#include <vector>
#include <set>

sf::Color GetColour(float t);


void Visualiser::DisplayParticles(sf::RenderWindow* window)
{
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		sf::CircleShape body;
		float radius = 1.0f;

		body.setRadius(radius);

		std::pair<float, float> position = GetWindowPosition(bodies->x[bodyIdx], bodies->y[bodyIdx]);
		body.setPosition(position.first - radius/2, position.second - radius/2);

		window->draw(body);
	}
}

void Visualiser::DisplayHeatmap(sf::RenderWindow* window, int grid_size)
{
	float bucket_length = size / grid_size;
	float** mass_count = new float*[grid_size];
	std::set<std::pair<int, int>> non_empty_blocks;

	for (int i = 0; i < grid_size; i++)
	{
		mass_count[i] = new float[grid_size];
		for (int j = 0; j < grid_size; j++)
		{
			mass_count[i][j] = 0;
		}
	}
		
	for (int bodyIdx = 0; bodyIdx < N; bodyIdx++)
	{
		std::pair<float, float> position = GetWindowPosition(bodies->x[bodyIdx], bodies->y[bodyIdx]);
		int display_x = (int)position.first;
		int display_y = (int)position.second;

		if (display_x >= 0 && display_x < size && display_y >= 0 && display_y < size)
		{
			int i = display_y / bucket_length;
			int j = display_x / bucket_length;
			mass_count[j][i] += bodies->g_mass[bodyIdx];
			non_empty_blocks.insert(std::make_pair(j, i));
		}
	}

	// draw
	for (auto iterator : non_empty_blocks)
	{
		int j = iterator.first;
		int i = iterator.second;
		sf::RectangleShape rect;
		rect.setSize(sf::Vector2f(bucket_length, bucket_length));
		rect.setPosition(j * bucket_length, i * bucket_length);

		// colour
		float t = logf(mass_count[j][i] / GRAVITATIONAL_CONSTANT) * 2 / logf(bodies->sum_g_mass / GRAVITATIONAL_CONSTANT);
		t = std::min(1.0f, t);
		// float t = std::min(1.0f, (float)body_count[j][i] * 10 / N);
		sf::Color colour = GetColour(t);
		rect.setFillColor(colour);

		window->draw(rect);
	}

	//for (int i = 0; i < grid_size; i++)
	//{
	//	for (int j = 0; j < grid_size; j++)
	//	{
	//		if (mass_count[j][i] == 0) 
	//		{
	//			// TODO Add threshold e.g. if less than 1% of all bodies are in the bucket don't draw it
	//			continue;
	//		}

	//		sf::RectangleShape rect;
	//		rect.setSize(sf::Vector2f(bucket_length, bucket_length));
	//		rect.setPosition(i * bucket_length, j * bucket_length);

	//		// colour
	//		float t = logf(mass_count[j][i] / GRAVITATIONAL_CONSTANT) * 2 / logf(bodies->sum_g_mass / GRAVITATIONAL_CONSTANT);
	//		t = std::min(1.0f, t);
	//		// float t = std::min(1.0f, (float)body_count[j][i] * 10 / N);
	//		sf::Color colour = GetColour(t);
	//		rect.setFillColor(colour);

	//		window->draw(rect);
	//	}
	//}

	for (size_t i = 0; i < grid_size; ++i)
	{
		delete[] mass_count[i];
	}
	delete[] mass_count;
}

std::pair<float, float> Visualiser::GetWindowPosition(float x, float y)
{
	static float offset = size / 2.0f;
	float display_x = x / resolution + offset;
	float display_y = y / resolution + offset;

	return std::make_pair(display_x, display_y);
}

sf::Color GetColour(float t)
{
	float R1, G1, B1, R2, G2, B2;
	if (t < 0.25f)
	{
		t *= 4;
		R1 = G1 = B1 = 0;
		R2 = 50;
		G2 = 80;
		B2 = 160;
	}
	else if (t < 0.5f)
	{
		t = (t - 0.25f) * 4;
		R1 = 50;
		G1 = 80;
		B1 = 160;

		R2 = 100;
		G2 = 215;
		B2 = 50;
	}
	else if (t < 0.75f)
	{
		t = (t - 0.5f) * 4;
		R1 = 100;
		G1 = 215;
		B1 = 50;

		R2 = 250;
		G2 = 175;
		B2 = 75;
	}
	else 
	{
		t = (t - 0.75f) * 4;
		R1 = 250;
		G1 = 175;
		B1 = 75;

		R2 = 255;
		G2 = 0;
		B2 = 0;
	}

	sf::Uint8 R = (sf::Uint8)(R1 + (R2 - R1) * t);
	sf::Uint8 G = (sf::Uint8)(G1 + (G2 - G1) * t);
	sf::Uint8 B = (sf::Uint8)(B1 + (B2 - B1) * t);

	return sf::Color(R, G, B);
}