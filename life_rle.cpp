#pragma once

#include "life_rle.h"

#include <regex>
#include <string>

int getCount(const std::string& repeatings)
{
	if (!repeatings.empty())
		return std::stoi(repeatings);
	return 1;
}

Dimension getPatternDimension(PWSTR pattern)
{
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