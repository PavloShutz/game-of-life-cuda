#include <Windows.h>
#include <shobjidl.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <regex>
#include <string>

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

__global__ void nextGen(bool* current, bool* successor)
{
	// calculate global thread index
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int stride = blockDim.x * gridDim.x;

	// Grid Stride Loop: covers all cells even if threads < cells
	for (int k = idx; k < SIZE * SIZE; k += stride)
	{
		// Map 1D index 'k' to 2D coordinates
		const int i = k % SIZE; // x column
		const int j = k / SIZE; // y row

		if (i > 0 && j > 0 && i < SIZE - 1 && j < SIZE - 1)
		{
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
		else
		{
			successor[k] = false;
		}
	}
}

enum class State
{
	Editing,
	Simulating
};

auto importPattern(PWSTR pattern) -> bool (*)[] {
	std::ifstream ifs(pattern);
	std::string line, contents;
	std::regex ptrn(R"(x\s*=\s*(\d+),\s*y\s*=\s*(\d+))");
	std::smatch matches;

	while (std::getline(ifs, line))
	{
		if (std::regex_search(line, matches, ptrn))
		{
			int x = std::stoi(matches[1]);
			int y = std::stoi(matches[2]);

			std::cout << "x=" << x << ", y=" << y << std::endl;
		}
		else if (!line.empty() && line[0] != '#')  // ignore comments
		{
			contents += line;
		}
	}

	std::size_t prevPos = 0, nextPos = 0;
	std::string repeatings;
	while (contents[nextPos] != '!')
	{
		if (contents[nextPos] == 'o' || contents[nextPos] == 'b')
		{
			// Get the amount of cell's repetition
			repeatings = contents.substr(prevPos, nextPos - prevPos);

			int cnt = 1;
			if (!repeatings.empty())
				cnt = std::stoi(repeatings);

			std::cout << std::string(cnt, (contents[nextPos] == 'b' ? '.' : '0'));

			prevPos = nextPos + 1;
		}
		else if (contents[nextPos] == '$')
		{
			repeatings = contents.substr(prevPos, nextPos - prevPos);
			int cnt = 1;
			if (!repeatings.empty())
				cnt = std::stoi(repeatings);
			std::cout << std::string(cnt, '\n');
			++prevPos;
		}
		++nextPos;
	}

	return nullptr;
}

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow)
{
	sf::RenderWindow window(sf::VideoMode(1024, 1024), "SFML works!");
	sf::CircleShape shape;
	shape.setFillColor(sf::Color::Green);
	const float scale = 1024.0f / SIZE;
	shape.setRadius(scale / 2.0f);
	State state = State::Editing;

	constexpr int N = SIZE * SIZE;
	bool* current, * successor;

	cudaMallocManaged(reinterpret_cast<void**>(&current), sizeof(current) * N);
	cudaMallocManaged(reinterpret_cast<void**>(&successor), sizeof(successor) * N);

	cudaMemset(current, 0, sizeof(current) * N);
	cudaMemset(successor, 0, sizeof(successor) * N);

	const WCHAR* szRLE = L"";

	COMDLG_FILTERSPEC rgSpec[] =
	{
			{ szRLE, L"*.rle" }
	};

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
					const auto pixel = window.mapPixelToCoords(sf::Mouse::getPosition(window));
					const unsigned x = pixel.x / (1024 / SIZE);
					const unsigned y = pixel.y / (1024 / SIZE);
					current[y * SIZE + x] = !current[y * SIZE + x];
				}
			}
			else if (event.type == sf::Event::KeyPressed)
			{
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
							hr = pFileOpen->SetFileTypes(1, rgSpec);
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
											importPattern(fPattern);
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

		for (int i = 0; i < SIZE; ++i)
		{
			for (int j = 0; j < SIZE; ++j)
			{
				if (current[j * SIZE + i])
				{
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
