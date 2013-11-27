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
 * Humanoid Robotics Lab Georgia Institute of Technology
 * Director: Mike Stilman http://www.golems.org
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <random>

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


const int MAP_HEIGHT = 10;
const int MAP_WIDTH = 10;
const float COLOR_STDDEV = 10.0f;

class World
{
public:
	World()
	{
		srand(time(NULL));

		goal = cv::Point2i(random()%MAP_WIDTH, random()%MAP_HEIGHT);
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
				map[y][x] = (uchar)random();
			}
		}
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

cv::Mat drawColorMap()
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
}
