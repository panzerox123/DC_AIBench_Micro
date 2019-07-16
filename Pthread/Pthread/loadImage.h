//loadImage.h

/**
 * Attension: link opencv_imgcodecs library as you compile
 * eg: g++ -std=c++11 loadImage.cpp -o loadImage -lopencv_imgcodecs
 */
#ifndef _LOADIMAGE_H
#define _LOADIMAGE_H

#define LOADIMAGE_TEST

#include <opencv2/core/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <uchar.h>
#include <dirent.h>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

typedef vector<vector<vector<float>>> singleImage;
typedef vector<vector<vector<vector<float>>>> batchImage;
typedef vector<vector<vector<float>>> singleMatrix;
typedef vector<vector<vector<vector<float>>>> batchMatrix;


void loadImage_single(string imagePath, string format, singleImage& outputImage)
{
	Mat img = imread(imagePath.c_str());
	int rows = img.rows;
	int cols = img.cols;

	if(format == "NHWC")
	{
		vector<float> oneDimension(3, 0.0);
		vector<vector<float>> twoDimension(cols, oneDimension);
		outputImage.resize(rows, twoDimension);
	}
	else
	{
		vector<float> oneDimension(cols, 0.0);
		vector<vector<float>> twoDimension(rows, oneDimension);
		outputImage.resize(3, twoDimension);
	}

	for(int i = 0; i < rows; ++i)
	{
		for(int j = 0; j < cols; ++j)
		{
			Vec3b pix = img.at<Vec3b>(i, j);
			uchar B = pix[0];
			uchar G = pix[1];
			uchar R = pix[2];
			if(format == "NHWC")
			{
				outputImage[i][j][0] = static_cast<float>(B);
				outputImage[i][j][1] = static_cast<float>(G);
				outputImage[i][j][2] = static_cast<float>(R);
			}
			else
			{
				outputImage[0][i][j] = static_cast<float>(B);
				outputImage[1][i][j] = static_cast<float>(G);
				outputImage[2][i][j] = static_cast<float>(R);
			}
		}
	}
	//cout<<"Image shape: <"<<rows<<", "<<cols<<">"<<endl;
}


void loadImage_single(string imagePath, string format, int _rows, int _cols, singleImage& outputImage)
{
	Mat img = imread(imagePath.c_str());
	int rows = _rows;
	int cols = _cols;

	if(format == "NHWC")
	{
		vector<float> oneDimension(3, 0.0);
		vector<vector<float>> twoDimension(cols, oneDimension);
		outputImage.resize(rows, twoDimension);
	}
	else
	{
		vector<float> oneDimension(cols, 0.0);
		vector<vector<float>> twoDimension(rows, oneDimension);
		outputImage.resize(3, twoDimension);
	}

	for(int i = 0; i < rows; ++i)
	{
		for(int j = 0; j < cols; ++j)
		{
			Vec3b pix = img.at<Vec3b>(i, j);
			uchar B = pix[0];
			uchar G = pix[1];
			uchar R = pix[2];
			if(format == "NHWC")
			{
				outputImage[i][j][0] = static_cast<float>(B);
				outputImage[i][j][1] = static_cast<float>(G);
				outputImage[i][j][2] = static_cast<float>(R);
			}
			else
			{
				outputImage[0][i][j] = static_cast<float>(B);
				outputImage[1][i][j] = static_cast<float>(G);
				outputImage[2][i][j] = static_cast<float>(R);
			}
		}
	}
	//cout<<"Image shape: <"<<rows<<", "<<cols<<">"<<endl;
}



void loadImage_batch(string dirPath, string format, int rows, int cols, batchImage& outputBatch)
{
	struct dirent *ptr;
	DIR *dir;
	dir = opendir(dirPath.c_str());

	while((ptr = readdir(dir)) != NULL)
	{
		if(ptr->d_name[0] == '.')
			continue;

		string filename = dirPath + ptr->d_name;
		singleImage newImage;
		loadImage_single(filename, format, rows, cols, newImage);
		outputBatch.push_back(newImage);
	}
	closedir(dir);
}


void fourDim2oneDim(batchImage& imgBatch, vector<float>& data)
{
	if(imgBatch.size() == 0 || imgBatch[0].size() == 0 || imgBatch[0][0].size() == 0 || imgBatch[0][0][0].size() == 0)
		return ;
	int batchSize = imgBatch.size();
	int heightSize = imgBatch[0].size();
	int wightSize = imgBatch[0][0].size();
	int channelSize = imgBatch[0][0][0].size();

	data.clear();

	for(int n = 0; n < batchSize; ++n)
	for(int h = 0; h < heightSize; ++h)
	for(int w = 0; w < wightSize; ++w)
	for(int c = 0; c < channelSize; ++c)
		data.push_back(imgBatch[n][h][w][c]);
	
	return ;
}



void loadMatrix_single(string filePath, string format, singleMatrix& outputMatrix)
{
	outputMatrix.clear();

	/* read data from input file */
	ifstream fin(filePath.c_str());
	if(!fin)
	{
		cout<<"Openning input file :"<<filePath<<" failed!"<<endl;
		return ;
	}

	/* read one line each time(data of one patient occupies one line) */
	string line;
	int rows, cols, channels;

	/* read rows, cols, channels */
	getline(fin, line);
	istringstream iss(line);
	iss>>rows;
	iss>>cols;
	iss>>channels;

	/* resize outputMatrix */
	if(format == "NHWC")
	{
		vector<float> oneDimension(channels, 0.0);
		vector<vector<float>> twoDimension(cols, oneDimension);
		outputMatrix.resize(rows, twoDimension);
	}
	else
	{
		vector<float> oneDimension(cols, 0.0);
		vector<vector<float>> twoDimension(rows, oneDimension);
		outputMatrix.resize(channels, twoDimension);
	}

	/* read matrix data */
	for(int c = 0; c < channels; ++c)
	for(int h = 0; h < rows; ++h)
	for(int w = 0; w < cols; ++w)
	{
		if(format == "NHWC")
			fin>>outputMatrix[h][w][c];
		else
			fin>>outputMatrix[c][h][w];
	}

	fin.close();

}


void loadMatrix_batch(string dirPath, string format, batchMatrix& outputBatch)
{
	struct dirent *ptr;
	DIR *dir;
	dir = opendir(dirPath.c_str());

	outputBatch.clear();
	while((ptr = readdir(dir)) != NULL)
	{
		if(ptr->d_name[0] == '.')
			continue;

		string filename = dirPath + ptr->d_name;
		singleMatrix newImage;
		loadMatrix_single(filename, format, newImage);
		outputBatch.push_back(newImage);
	}
	closedir(dir);
}


#endif


