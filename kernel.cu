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

constexpr int GRID_SIZE = 128;

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

enum class State
{
	Editing,
	Simulating
};

struct Dimension
{
	int x;
	int y;
};

struct Pattern {
	Dimension dimension;
	bool* cells;

	Pattern() : dimension{ 0, 0 }, cells(nullptr) {}

	Pattern(Dimension dim, bool* data) : dimension(dim), cells(data) {}

	~Pattern()
	{
		delete[] cells;
	}

	// Prevent copying (raw pointer ownership)
	Pattern(const Pattern&) = delete;
	Pattern& operator=(const Pattern&) = delete;

	// Allow moving
	Pattern(Pattern&& other) noexcept
		: dimension(other.dimension), cells(other.cells)
	{
		other.cells = nullptr;
	}

	inline bool isValid() const
	{
		return cells != nullptr && dimension.x > 0 && dimension.y > 0;
	}
};

int getCount(const std::string& repeatings)
{
	if (!repeatings.empty())
		return std::stoi(repeatings);
	return 1;
}

Dimension getPatternDimension(PWSTR pattern) {
	std::ifstream ifs(pattern);
	std::string line;
	std::regex ptrn(R"(x\s*=\s*(\d+),\s*y\s*=\s*(\d+))");
	std::smatch matches;
	while (std::getline(ifs, line))
	{
		if (std::regex_search(line, matches, ptrn))
		{
			Dimension dimension = Dimension{ std::stoi(matches[1]), std::stoi(matches[2]) };
			return dimension;
		}
	}
	return Dimension{};
}

std::string getPatternContent(PWSTR pattern)
{
	std::ifstream ifs(pattern);
	std::string line, contents;
	while (std::getline(ifs, line))
	{
		if (!line.empty() && line[0] != '#')  // ignore comments
		{
			contents += line;
		}
	}
	return contents;
}

void fillGridFromPattern(bool* grid, const Dimension& dimension, const std::string& contents)
{
	std::size_t currentX = 0, currentY = 0; // coordinates in a new pattern grid
	std::string buffer;
	for (char ch : contents)
	{
		if (ch == '!') // end of pattern
			break;

		if (std::isdigit(ch))
		{
			buffer += ch;
		}
		else
		{
			const int count = getCount(buffer);
			buffer.clear();

			if (ch == 'b')
			{
				currentX += count;
			}
			else if (ch == 'o')
			{
				for (int i = 0; i < count; ++i)
				{
					if (currentX < dimension.x && currentY < dimension.y)
					{
						grid[currentY * dimension.x + currentX] = true;
					}
					currentX++;
				}
			}
			else if (ch == '$') // new line
			{
				currentY += count;
				currentX = 0;
			}
		}
	}
}

Pattern importPattern(PWSTR pattern)
{
	Dimension dimension = getPatternDimension(pattern);
	bool* grid = new bool[dimension.x * dimension.y] {};
	std::string contents = getPatternContent(pattern);

	fillGridFromPattern(grid, dimension, contents);

	Pattern result(dimension, grid);
	return result;
}

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow)
{
	sf::RenderWindow window(sf::VideoMode(1024, 1024), "GoL CUDA");
	sf::CircleShape shape;
	shape.setFillColor(sf::Color::Green);
	const float scale = 1024.0f / GRID_SIZE;
	shape.setRadius(scale / 2.0f);
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
					const auto pixel = window.mapPixelToCoords(sf::Mouse::getPosition(window));
					const unsigned x = pixel.x / (1024 / GRID_SIZE);
					const unsigned y = pixel.y / (1024 / GRID_SIZE);
					current[y * GRID_SIZE + x] = !current[y * GRID_SIZE + x];
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
				default:
					break;
				}
			}
		}

		if (state == State::Simulating)
		{
			Timer t;
			const int numBlocks = (N + 255) / 256;
			nextGen<<<numBlocks, 256>>>(current, successor);

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
