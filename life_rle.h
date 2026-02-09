#pragma once


#include <Windows.h>

#include <fstream>

struct Dimension
{
	int x;
	int y;
};

struct Pattern
{
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

int getCount(const std::string& repeatings);

Dimension getPatternDimension(PWSTR pattern);
std::string getPatternContent(PWSTR pattern);

void fillGridFromPattern(bool* grid, const Dimension& dimension, const std::string& contents);

Pattern importPattern(PWSTR pattern);