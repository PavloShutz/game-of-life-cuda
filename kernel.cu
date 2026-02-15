#include "life_rle.h"
#include "timer.h"

#include <Windows.h>
#include <shobjidl.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <string>

#include <SFML/Graphics.hpp>

constexpr int GRID_SIZE = 1 << 12;
constexpr int CELL_COUNT = GRID_SIZE * GRID_SIZE;
constexpr float CELL_SCALE = 32.f;
constexpr float WORLD_SIZE = GRID_SIZE * CELL_SCALE;
constexpr int WINDOW_SIZE = 1024;
constexpr int BLOCK_SIZE = 256;

#define ENABLE_CUDA
//#undef ENABLE_CUDA

enum class State
{
	Editing,
	Simulating
};

#ifdef ENABLE_CUDA
__global__
#endif
void nextGen(bool* current, bool* successor)
{
#ifdef ENABLE_CUDA
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int stride = blockDim.x * gridDim.x;
#else
	const int idx = 0;
	const int stride = 1;
#endif
	for (int k = idx; k < CELL_COUNT; k += stride)
	{
		const int i = k % GRID_SIZE;
		const int j = k / GRID_SIZE;

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

			successor[k] = (neighbors == 3 || (neighbors == 2 && current[k]));
		}
		else
		{
			successor[k] = false;
		}
	}
}

bool allocateGrids(bool*& current, bool*& successor)
{
#ifdef ENABLE_CUDA
	cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&current), sizeof(bool) * CELL_COUNT);
	if (err != cudaSuccess)
	{
		MessageBoxA(NULL, cudaGetErrorString(err), "CUDA Error", MB_OK | MB_ICONERROR);
		return false;
	}
	err = cudaMallocManaged(reinterpret_cast<void**>(&successor), sizeof(bool) * CELL_COUNT);
	if (err != cudaSuccess)
	{
		MessageBoxA(NULL, cudaGetErrorString(err), "CUDA Error", MB_OK | MB_ICONERROR);
		cudaFree(current);
		return false;
	}
	cudaMemset(current, 0, sizeof(bool) * CELL_COUNT);
	cudaMemset(successor, 0, sizeof(bool) * CELL_COUNT);
#else
	current = new bool[CELL_COUNT]();
	successor = new bool[CELL_COUNT]();
#endif
	return true;
}

void freeGrids(bool* current, bool* successor)
{
#ifdef ENABLE_CUDA
	cudaFree(current);
	cudaFree(successor);
#else
	delete[] current;
	delete[] successor;
#endif
}

void stepSimulation(bool*& current, bool*& successor)
{
#ifdef ENABLE_CUDA
	constexpr int numBlocks = (CELL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
	nextGen<<<numBlocks, BLOCK_SIZE>>>(current, successor);
	cudaDeviceSynchronize();
#else
	nextGen(current, successor);
#endif
	std::swap(current, successor);
}

void clampView(sf::View& view)
{
	sf::Vector2f center = view.getCenter();
	const sf::Vector2f halfSize = view.getSize() / 2.f;

	if (center.x < halfSize.x)
		center.x = halfSize.x;
	else if (center.x > WORLD_SIZE - halfSize.x)
		center.x = WORLD_SIZE - halfSize.x;

	if (center.y < halfSize.y)
		center.y = halfSize.y;
	else if (center.y > WORLD_SIZE - halfSize.y)
		center.y = WORLD_SIZE - halfSize.y;

	view.setCenter(center);
}

void moveView(sf::View& view, sf::Keyboard::Scancode scan)
{
	sf::Vector2f offset;
	switch (scan)
	{
	case sf::Keyboard::Scancode::A: offset.x = -CELL_SCALE; break;
	case sf::Keyboard::Scancode::D: offset.x = CELL_SCALE; break;
	case sf::Keyboard::Scancode::W: offset.y = -CELL_SCALE; break;
	case sf::Keyboard::Scancode::S: offset.y = CELL_SCALE; break;
	default: break;
	}
	view.move(offset);
	clampView(view);
}

void loadPatternFromFile(bool* grid)
{
	static const COMDLG_FILTERSPEC rgSpec[] = { { L"", L"*.rle" } };

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
	if (FAILED(hr))
		return;

	IFileOpenDialog* pFileOpen = nullptr;
	hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
		IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

	if (SUCCEEDED(hr))
	{
		pFileOpen->SetFileTypes(ARRAYSIZE(rgSpec), rgSpec);
		if (SUCCEEDED(pFileOpen->Show(NULL)))
		{
			IShellItem* pItem = nullptr;
			if (SUCCEEDED(pFileOpen->GetResult(&pItem)))
			{
				PWSTR filePath = nullptr;
				if (SUCCEEDED(pItem->GetDisplayName(SIGDN_FILESYSPATH, &filePath)))
				{
					auto pattern = importPattern(filePath);
					if (pattern.isValid())
					{
						const int maxX = (pattern.dimension.x < GRID_SIZE) ? pattern.dimension.x : GRID_SIZE;
						const int maxY = (pattern.dimension.y < GRID_SIZE) ? pattern.dimension.y : GRID_SIZE;
						for (int j = 0; j < maxY; ++j)
							for (int i = 0; i < maxX; ++i)
								grid[j * GRID_SIZE + i] = pattern.cells[j * pattern.dimension.x + i];
					}
					CoTaskMemFree(filePath);
				}
				pItem->Release();
			}
		}
		pFileOpen->Release();
	}
	CoUninitialize();
}

void toggleCell(bool* grid, const sf::RenderWindow& window)
{
	const auto worldPos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
	const unsigned x = static_cast<unsigned>(worldPos.x / CELL_SCALE);
	const unsigned y = static_cast<unsigned>(worldPos.y / CELL_SCALE);
	if (x < static_cast<unsigned>(GRID_SIZE) && y < static_cast<unsigned>(GRID_SIZE))
		grid[y * GRID_SIZE + x] = !grid[y * GRID_SIZE + x];
}

void handleEvents(sf::RenderWindow& window, sf::View& view, State& state, bool* grid)
{
	sf::Event event;
	while (window.pollEvent(event))
	{
		switch (event.type)
		{
		case sf::Event::Closed:
			window.close();
			break;
		case sf::Event::MouseButtonPressed:
			if (event.mouseButton.button == sf::Mouse::Button::Left)
				toggleCell(grid, window);
			break;
		case sf::Event::KeyPressed:
			switch (event.key.scancode)
			{
			case sf::Keyboard::Scancode::R: state = State::Simulating; break;
			case sf::Keyboard::Scancode::E: state = State::Editing; break;
			case sf::Keyboard::Scancode::O: loadPatternFromFile(grid); break;
			case sf::Keyboard::Scancode::A:
			case sf::Keyboard::Scancode::D:
			case sf::Keyboard::Scancode::W:
			case sf::Keyboard::Scancode::S:
				moveView(view, event.key.scancode);
				window.setView(view);
				break;
			default: break;
			}
			break;
		default:
			break;
		}
	}
}

void renderGrid(sf::RenderWindow& window, const bool* grid, sf::RectangleShape& cell)
{
	const sf::View& view = window.getView();
	const sf::Vector2f center = view.getCenter();
	const sf::Vector2f halfSize = view.getSize() / 2.f;

	int minCol = static_cast<int>((center.x - halfSize.x) / CELL_SCALE);
	int maxCol = static_cast<int>((center.x + halfSize.x) / CELL_SCALE) + 1;
	int minRow = static_cast<int>((center.y - halfSize.y) / CELL_SCALE);
	int maxRow = static_cast<int>((center.y + halfSize.y) / CELL_SCALE) + 1;

	if (minCol < 0) minCol = 0;
	if (minRow < 0) minRow = 0;
	if (maxCol > GRID_SIZE) maxCol = GRID_SIZE;
	if (maxRow > GRID_SIZE) maxRow = GRID_SIZE;

	for (int j = minRow; j < maxRow; ++j)
	{
		const int rowOffset = j * GRID_SIZE;
		for (int i = minCol; i < maxCol; ++i)
		{
			if (grid[rowOffset + i])
			{
				cell.setPosition(static_cast<float>(i) * CELL_SCALE, static_cast<float>(j) * CELL_SCALE);
				window.draw(cell);
			}
		}
	}
}

int WINAPI WinMain(HINSTANCE hThisInstance, HINSTANCE hPrevInstance, LPSTR lpszArgument, int nCmdShow)
{
	bool* current = nullptr;
	bool* successor = nullptr;
	if (!allocateGrids(current, successor))
		return -1;

	sf::RenderWindow window(sf::VideoMode(WINDOW_SIZE, WINDOW_SIZE), "GoL CUDA");
	sf::View view(sf::FloatRect(0.f, 0.f,
		static_cast<float>(WINDOW_SIZE), static_cast<float>(WINDOW_SIZE)));
	window.setView(view);

	sf::RectangleShape cell(sf::Vector2f(CELL_SCALE, CELL_SCALE));
	cell.setFillColor(sf::Color::Green);

	State state = State::Editing;
	std::fstream log("log.txt", std::ios::app);

	while (window.isOpen())
	{
		handleEvents(window, view, state, current);

		if (state == State::Simulating)
		{
			Timer t;
			stepSimulation(current, successor);
			log << t.elapsed() << '\n';
		}

		window.clear();
		renderGrid(window, current, cell);
		window.display();
	}

	freeGrids(current, successor);
	return 0;
}
