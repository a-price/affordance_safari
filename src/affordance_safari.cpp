/**
 * \file affordance_safari.cpp
 * \brief
 *
 * \author Andrew Price
 * \date November 27, 2013
 *
 * \copyright
 *
 * Copyright (c) 2013, Georgia Tech Research Corporation
 * All rights reserved.
 *
 * This file is provided under the following "BSD-style" License:
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define _USE_MATH_DEFINES // for MS C++
#include <cmath>
#include <random>
#include <time.h>

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

template <class T>
inline T square(const T &x) { return x*x; }

inline float randbetween(float min, float max)
{
	return (max - min) * ( (float)rand() / (float)RAND_MAX ) + min;
}

class Distribution
{
public:
	Distribution(double mu, double s, double w)
	{
		mean = mu;
		covariance = s;
		setWeight(w);
	}

	double mean;
	double covariance;

	inline void setWeight(const double w)
	{
		weight = w;
		l_weight = log(w);
	}

	double probability(const double x) const
	{
		double delta = (x-mean);
		return exp(-(delta*delta)/(2.0*covariance) + l_weight);
	}

	friend std::ostream& operator <<(std::ostream& s, const Distribution& d)
	{
		s << "{ Mean: " << d.mean << " Covar: " << d.covariance << " Weight: " << d.weight << " }";
		return s;
	}

protected:
	double weight;
	double l_weight;
};


const int MAP_HEIGHT = 300;
const int MAP_WIDTH = 300;
const float COLOR_STDDEV = 10.0f;
const int NUM_SEARCHERS = MAP_HEIGHT * MAP_WIDTH / 5000;
const int MAX_ITERS = 1000;

class World
{
public:
	World()
	{
		srand((unsigned int)time(NULL));

		goal = cv::Point2i(rand()%MAP_WIDTH, rand()%MAP_HEIGHT);
		distribution = std::normal_distribution<float>(0.0, COLOR_STDDEV);

		createMap();
		recolorMap();
	}

	unsigned char map[MAP_HEIGHT][MAP_WIDTH];
	float colorMap[MAP_HEIGHT][MAP_WIDTH];

	cv::Point2i goal;

	std::default_random_engine generator;
	std::normal_distribution<float> distribution;

	void createMap()
	{
		for (int y = 0; y < MAP_HEIGHT; ++y)
		{
			for (int x = 0; x < MAP_WIDTH; ++x)
			{
				map[y][x] = (uchar)rand();
			}
		}

		cv::Mat temp(MAP_HEIGHT, MAP_WIDTH, CV_8UC1, map, sizeof(uchar) * MAP_WIDTH);

		int neighborhoodSize = (MAP_HEIGHT > MAP_WIDTH) ? MAP_WIDTH : MAP_HEIGHT;
		neighborhoodSize /= 40;
		if (neighborhoodSize % 2 == 0) { neighborhoodSize += 1; }
		cv::GaussianBlur(temp, temp, cv::Size(neighborhoodSize,neighborhoodSize), 3, 3);

		map[goal.y][goal.x] = 0;
	}

	void recolorMap()
	{
		for (int y = 0; y < MAP_HEIGHT; ++y)
		{
			for (int x = 0; x < MAP_WIDTH; ++x)
			{
				colorMap[y][x] = (float)map[y][x] + distribution(generator);
			}
		}
	}
};

class Searcher
{
public:
	Searcher(float l, cv::Point2i start, World* w)
	{
		affordanceLevel = l;
		location = start;
		world = w;
	}

	float affordanceLevel;
	cv::Point2i location;
	World* world;

	inline float distanceFunction(cv::Point2i& a, cv::Point2i& b)
	{
		return square(square(a.x-b.x)+square(a.y-b.y));
	}

	std::vector<cv::Point2i> getNeighbors()
	{
		std::vector<cv::Point2i> neighbors;

		if (location.y > 0) { neighbors.push_back(cv::Point2i(location.x, location.y - 1)); }
		if (location.y < MAP_HEIGHT - 1) { neighbors.push_back(cv::Point2i(location.x, location.y + 1)); }
		if (location.x > 0) { neighbors.push_back(cv::Point2i(location.x - 1, location.y)); }
		if (location.x < MAP_WIDTH - 1) { neighbors.push_back(cv::Point2i(location.x + 1, location.y)); }

		return neighbors;
	}

	bool goLocation(const cv::Point2i& destination)
	{
		if ((float)world->map[destination.y][destination.x] < affordanceLevel)
		{
			location = destination;
			return true;
		}
		else
		{
			std::cout << "Move Failed:" << (float)world->map[destination.y][destination.x] << ">" << affordanceLevel << std::endl;
			return false;
		}
	}

	cv::Point2i proposeMove()
	{
		float neighborScores[4];
		float totalScore = 0;
		std::vector<cv::Point2i> neighbors = getNeighbors();
		for (int n = 0; n < neighbors.size(); ++n)
		{
			if (neighbors[n] == world->goal) { return neighbors[n]; } // Victory!
			float score = 1/distanceFunction(world->goal, neighbors[n]);
			neighborScores[n] = score + (n > 0 ? neighborScores[n-1] : 0);
			totalScore += score;
		}

		for (int n = 0; n < neighbors.size(); ++n)
		{
			neighborScores[n] /= totalScore;
		}

		float r = randbetween(0, 1);

		for (int n = 0; n < neighbors.size(); ++n)
		{
			if (r < neighborScores[n])
			{
				return neighbors[n];
			}
		}
		return neighbors.back();
	}
};

World world;

cv::Vec3b displayColor(uchar val)
{
	cv::Vec3b vec;
	//vec[1] = 255-val;
	//vec[2] = val;
	float a = (float)val / 255.0f * M_PI / 2.0;
	vec[1] = 255.0f * cos(a);
	vec[2] = 255.0f * sin(a);
	return vec;
}


cv::Mat drawMap()
{
	cv::Mat mapPic(MAP_HEIGHT, MAP_WIDTH, CV_8UC1);
	for (int y = 0; y < MAP_HEIGHT; ++y)
	{
		for (int x = 0; x < MAP_WIDTH; ++x)
		{
			mapPic.data[y * MAP_WIDTH + x] = 255-world.map[y][x];
		}
	}
	return mapPic;
}

cv::Mat drawColorMap(std::vector<Searcher> searchers = std::vector<Searcher>())
{
	cv::Mat mapPic(MAP_HEIGHT, MAP_WIDTH, CV_8UC3);
	for (int y = 0; y < MAP_HEIGHT; ++y)
	{
		for (int x = 0; x < MAP_WIDTH; ++x)
		{
			mapPic.at<cv::Vec3b>(cv::Point2i(x,y)) = displayColor(world.colorMap[y][x]);
		}
	}
	mapPic.at<cv::Vec3b>(world.goal) = cv::Vec3b(255, 255, 255);
	for (Searcher s : searchers)
	{
		mapPic.at<cv::Vec3b>(s.location) = cv::Vec3b(255, 0, 0);
	}
	return mapPic;
}

int main()
{
	cv::Mat mapPic = drawMap();
	cv::Mat mapPicColor = drawColorMap();

	cv::namedWindow("True Map", cv::WINDOW_NORMAL);
	cv::imshow("True Map", mapPic);

	cv::namedWindow("Color Map", cv::WINDOW_NORMAL);
	cv::imshow("Color Map", mapPicColor);
	cv::waitKey();

	std::vector<Searcher> searchers;
	for (int i = 0; i < NUM_SEARCHERS; ++i)
	{
		searchers.push_back(Searcher(randbetween(150, 230), cv::Point2i(rand()%MAP_WIDTH, rand()%MAP_HEIGHT), &world));
	}

	int iters = 0;
	while (iters < MAX_ITERS)
	{
		for (Searcher& s : searchers)
		{
			s.goLocation(s.proposeMove());
			if (s.location == world.goal) { return 0; }
		}

		mapPicColor = drawColorMap(searchers);
		cv::imshow("Color Map", mapPicColor);
		cv::waitKey(10);
		iters++;
	}


}
