#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#include <SFML/Graphics.hpp>

constexpr std::size_t DIM = 32; /* cell size */
constexpr std::size_t WIDTH = 1920;
constexpr std::size_t HEIGHT = 1080;
constexpr std::size_t CELLS = WIDTH / DIM + 1; /* number of cells in a grid */

__global__ void init(int n, sf::Vertex* top, sf::Vertex* left) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i += stride) {
		top[i].position.x = n * i;
		top[i].position.y = 0;
		left[i].position.x = 0;
		left[i].position.y = n * i;
	}
}

__global__ void nextGen(int n, bool* current, bool* successor) {
	const int index = threadIdx.x;
	const int stride = blockDim.x;
	for (int i = index; i < n; i += stride) {
		if (i > 0 && i < n - 1) {
			const int neighbours =
				static_cast<int>(current[(i - 1) * n + (i - 1)])
				+ static_cast<int>(current[i * n + (i - 1)])
				+ static_cast<int>(current[(i + 1) * n + (i - 1)])
				+ static_cast<int>(current[(i - 1) * n + i])
				+ static_cast<int>(current[(i + 1) * n + i])
				+ static_cast<int>(current[(i - 1) * n + (i + 1)])
				+ static_cast<int>(current[i * n + (i + 1)])
				+ static_cast<int>(current[(i + 1) * n + (i + 1)]);

			const bool alive = current[i * n + i];
			successor[i * n + i] = (neighbours == 3 || (neighbours == 2 && alive));
		}
	}
}

enum class State {
	Editing,
	Simulating
};

int main(void) {
	sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
	sf::CircleShape shape(100.f);
	shape.setFillColor(sf::Color::Green);

	sf::Vertex top[32], left[32];

	cudaMallocManaged(reinterpret_cast<void**>(&top), sizeof(top));
	cudaMallocManaged(reinterpret_cast<void**>(&left), sizeof(left));

	init<<<1, 256>>>(32, top, left);

	cudaDeviceSynchronize();

	bool* current, * successor;

	cudaMallocManaged(reinterpret_cast<void**>(&current), sizeof(current) * 32 * 32);
	cudaMallocManaged(reinterpret_cast<void**>(&successor), sizeof(successor) * 32 * 32);

	// initialize on host
	current = {}, successor = {};

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
		}

		nextGen<<<1, 256>>>(32, current, successor);
		cudaDeviceSynchronize();
		current = successor;

		window.clear();
		window.draw(shape);
		window.display();
	}

	cudaFree(top);
	cudaFree(left);

	return 0;
}
