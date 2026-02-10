#include "life_rle.h"
#include "timer.h"

#include <Windows.h>
#include <shobjidl.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

#include <SFML/Graphics.hpp>

constexpr int GRID_SIZE = 128;
constexpr float CELL_SCALE = 32.f;

__global__ void nextGen(bool* current, bool* successor)
{
	// calculate global thread index
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int stride = blockDim.x * gridDim.x;

	// Grid Stride Loop: covers all cells even if threads < cells
	for (int k = idx; k < GRID_SIZE * GRID_SIZE; k += stride)
	{
		// Map 1D index 'k' to 2D coordinates
		const int i = k % GRID_SIZE; // x column
		const int j = k / GRID_SIZE; // y row

		if (i > 0 && j > 0 && i < GRID_SIZE - 1 && j < GRID_SIZE - 1)
		{
			int neighbors = 0;
			neighbors += current[(j - 1) * GRID_SIZE + (i - 1)];
			neighbors += current[(j - 1) * GRID_SIZE + i];
			neighbors += current[(j - 1) * GRID_SIZE + (i + 1)];
			neighbors += current[j * GRID_SIZE + (i - 1)];
			neighbors += current[j * GRID_SIZE + (i + 1)];
			neighbors += current[(j + 1) * GRID_SIZE + (i - 1)];
			neighbors += current[(j + 1) * GRID_SIZE + i];
			neighbors += current[(j + 1) * GRID_SIZE + (i + 1)];

			bool isAlive = current[j * GRID_SIZE + i];
			successor[k] = (neighbors == 3 || (neighbors == 2 && isAlive));
		}
		else
		{
			successor[k] = false;
		}
	}
}

void moveView(sf::View& view, sf::Keyboard::Scancode scan)
{
	sf::Vector2f offset;
	switch (scan)
	{
	case sf::Keyboard::Scancode::A:
		offset.x = -32.f;
		break;
	case sf::Keyboard::Scancode::D:
		offset.x = 32.f;
		break;
	case sf::Keyboard::Scancode::W:
		offset.y = -32.f;
		break;
	case sf::Keyboard::Scancode::S:
		offset.y = 32.f;
		break;
	default:
		break;
	}
	view.move(offset);

	const sf::Vector2f viewSize = view.getSize();
	sf::Vector2f center = view.getCenter();
	const float worldSize = GRID_SIZE * CELL_SCALE;

	const float minX = viewSize.x / 2.f;
	const float maxX = worldSize - viewSize.x / 2.f;
	const float minY = viewSize.y / 2.f;
	const float maxY = worldSize - viewSize.y / 2.f;

	// Clamp X
	if (center.x < minX)
		center.x = minX;
	else if (center.x > maxX)
		center.x = maxX;

	// Clamp Y
	if (center.y < minY)
		center.y = minY;
	else if (center.y > maxY)
		center.y = maxY;

	view.setCenter(center);
}

enum class State
{
	Editing,
	Simulating
};

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow)
{
	sf::RenderWindow window(sf::VideoMode(1024, 1024), "GoL CUDA");
	sf::View view(sf::FloatRect(0.f, 0.f, 1024.f, 1024.f));
	window.setView(view);

	sf::CircleShape shape;
	shape.setFillColor(sf::Color::Green);
	shape.setRadius(CELL_SCALE / 2.0f);
	State state = State::Editing;

	constexpr int N = GRID_SIZE * GRID_SIZE;
	bool* current, * successor;

	cudaMallocManaged(reinterpret_cast<void**>(&current), sizeof(bool) * N);
	cudaMallocManaged(reinterpret_cast<void**>(&successor), sizeof(bool) * N);

	cudaMemset(current, 0, sizeof(bool) * N);
	cudaMemset(successor, 0, sizeof(bool) * N);

	const WCHAR* szRLE = L"";
	COMDLG_FILTERSPEC rgSpec[] =
	{
			{ szRLE, L"*.rle" }
	};
	const auto rgSpecSize = sizeof(rgSpec) / sizeof(COMDLG_FILTERSPEC);
	PWSTR fPattern;

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}
			else if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Button::Left)
				{
					const auto worldPos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
					const unsigned x = worldPos.x / CELL_SCALE;
					const unsigned y = worldPos.y / CELL_SCALE;
					if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE)
					{
						current[y * GRID_SIZE + x] = !current[y * GRID_SIZE + x];
					}
				}
			}
			else if (event.type == sf::Event::KeyPressed)
			{
				switch (event.key.scancode) {
				case sf::Keyboard::Scancode::R:
					state = State::Simulating;
					break;
				case sf::Keyboard::Scancode::E:
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
							hr = pFileOpen->SetFileTypes(rgSpecSize, rgSpec);
							if (SUCCEEDED(hr))
							{
								hr = pFileOpen->Show(NULL);
								if (SUCCEEDED(hr))
								{
									IShellItem* pItem;
									hr = pFileOpen->GetResult(&pItem);
									if (SUCCEEDED(hr))
									{
										hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &fPattern);
										if (SUCCEEDED(hr))
										{
											auto pattern = importPattern(fPattern);
											if (pattern.isValid())
											{
												const int maxX = (pattern.dimension.x < GRID_SIZE) ? pattern.dimension.x : GRID_SIZE;
												const int maxY = (pattern.dimension.y < GRID_SIZE) ? pattern.dimension.y : GRID_SIZE;
												for (int i = 0; i < maxX; ++i)
													for (int j = 0; j < maxY; ++j)
														current[j * GRID_SIZE + i] = pattern.cells[j * pattern.dimension.x + i];
											}
										}
										pItem->Release();
									}
								}
							}
							pFileOpen->Release();
						}
						CoUninitialize();
					}
				}
				break;
				case sf::Keyboard::Scancode::A:
				case sf::Keyboard::Scancode::D:
				case sf::Keyboard::Scancode::W:
				case sf::Keyboard::Scancode::S:
					moveView(view, event.key.scancode);
					window.setView(view);
					break;
				default:
					break;
				}
			}
		}

		if (state == State::Simulating)
		{
			Timer t;
			const int numBlocks = (N + 255) / 256;
			nextGen << <numBlocks, 256 >> > (current, successor);

			cudaDeviceSynchronize();

			std::swap(current, successor);
			std::cout << t.elapsed() << '\n';
		}

		window.clear();

		for (int i = 0; i < GRID_SIZE; ++i)
		{
			for (int j = 0; j < GRID_SIZE; ++j)
			{
				if (current[j * GRID_SIZE + i])
				{
					shape.setPosition(sf::Vector2f(i * CELL_SCALE, j * CELL_SCALE));
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
