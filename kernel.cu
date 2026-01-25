#include <cuda_runtime_api.h>
#include <iostream>
#include <cmath>

#include <SFML/Graphics.hpp>

__global__ void add(int n, float* x, float* y) {
	for (int i = 0; i < n; ++i) {
		y[i] = x[i] + y[i];
	}
}

int main(void) {
	int N = 1 << 20; // 1M elements

	float* x, * y;
	auto err1 = cudaMallocManaged(reinterpret_cast<void**>(&x), N * sizeof(float));
	auto err2 = cudaMallocManaged(reinterpret_cast<void**>(&y), N * sizeof(float));

	if (err1 != cudaSuccess || err2 != cudaSuccess)
		goto Failure;

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the CPU
	add<<<1, 1>>>(N, x, y);

	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "Max error: " << maxError << std::endl;

Failure:
	// Free memory
	cudaFree(x);
	cudaFree(y);

	sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
	sf::CircleShape shape(100.f);
	shape.setFillColor(sf::Color::Green);

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		window.clear();
		window.draw(shape);
		window.display();
	}

	return 0;
}