#include <Windows.h>
#include <shobjidl.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#include <chrono>

#include <SFML/Graphics.hpp>

class Timer
{
private:
	// Type aliases to make accessing nested type easier
	using Clock = std::chrono::steady_clock;
	using Second = std::chrono::duration<double, std::ratio<1> >;

	std::chrono::time_point<Clock> m_beg{ Clock::now() };

public:
	void reset()
	{
		m_beg = Clock::now();
	}

	double elapsed() const
	{
		return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
	}
};

#define SIZE 128

__global__ void nextGen(bool* current, bool* successor) {
	// calculate global thread index
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int stride = blockDim.x * gridDim.x;

	// Grid Stride Loop: covers all cells even if threads < cells
	for (int k = idx; k < SIZE * SIZE; k += stride) {

		// Map 1D index 'k' to 2D coordinates
		const int i = k % SIZE; // x column
		const int j = k / SIZE; // y row

		if (i > 0 && j > 0 && i < SIZE - 1 && j < SIZE - 1) {
			int neighbors = 0;
			neighbors += current[(j - 1) * SIZE + (i - 1)];
			neighbors += current[(j - 1) * SIZE + i];
			neighbors += current[(j - 1) * SIZE + (i + 1)];
			neighbors += current[j * SIZE + (i - 1)];
			neighbors += current[j * SIZE + (i + 1)];
			neighbors += current[(j + 1) * SIZE + (i - 1)];
			neighbors += current[(j + 1) * SIZE + i];
			neighbors += current[(j + 1) * SIZE + (i + 1)];

			bool isAlive = current[j * SIZE + i];
			successor[k] = (neighbors == 3 || (neighbors == 2 && isAlive));
		}
		else {
			successor[k] = false;
		}
	}
}

enum class State {
	Editing,
	Simulating
};

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow){
	sf::RenderWindow window(sf::VideoMode(1024, 1024), "SFML works!");
	sf::CircleShape shape;
	shape.setFillColor(sf::Color::Green);
	const float scale = 1024.0f / SIZE;
	shape.setRadius(scale / 2.0f);
	State state = State::Editing;

	constexpr int N = SIZE * SIZE;
	bool *current, *successor;

	cudaMallocManaged(reinterpret_cast<void**>(&current), sizeof(current) * N);
	cudaMallocManaged(reinterpret_cast<void**>(&successor), sizeof(successor) * N);

	cudaMemset(current, 0, sizeof(current) * N);
	cudaMemset(successor, 0, sizeof(successor) * N);

	while (window.isOpen()) {
		sf::Event event;
		while (window.pollEvent(event)) {
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::MouseButtonPressed) {
				if (event.mouseButton.button == sf::Mouse::Button::Left) {
					const auto pixel = window.mapPixelToCoords(sf::Mouse::getPosition(window));
					const unsigned x = pixel.x / (1024 / SIZE);
					const unsigned y = pixel.y / (1024 / SIZE);
					current[y * SIZE + x] = !current[y * SIZE + x];
				}
			}
			else if (event.type == sf::Event::KeyPressed) {
				switch (event.key.scancode) {
				case sf::Keyboard::Scancode::R:
					state = State::Simulating;
					break;
				case sf::Keyboard::Scancode::S:
					state = State::Editing;
					break;
				case sf::Keyboard::Scancode::O:
				{
					HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
					if (SUCCEEDED(hr))
					{
						IFileOpenDialog* pFileOpen;

						hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
							IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

						if (SUCCEEDED(hr))
						{
							hr = pFileOpen->Show(NULL);

							// TODO: Retrieve file, open it and import pattern

							pFileOpen->Release();
						}
						CoUninitialize();
					}
				}
					break;
				default:
					break;
				}
			}
		}

		if (state == State::Simulating) {
			Timer t;
			const int numBlocks = (N + 255) / 256;
			nextGen << <numBlocks, 256 >> > (current, successor);
			
			cudaDeviceSynchronize();
			
			std::swap(current, successor);
			std::cout << t.elapsed() << '\n';
		}

		window.clear();

		for (int i = 0; i < SIZE; ++i) {
			for (int j = 0; j < SIZE; ++j) {
				if (current[j * SIZE +i]) {
					shape.setPosition(sf::Vector2f(i * scale, j * scale));
					window.draw(shape);
				}
			}
		}
		
		window.display();
	}

	cudaFree(current);
	cudaFree(successor);

	return 0;
}
