//conv2d.h

#ifndef _CONV2D_H
#define _CONV2D_H

#define CONV_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include "loadImage.h"

using namespace std;


typedef vector<vector<vector<float>>> singleImage;
typedef vector<vector<vector<vector<float>>>> imageBatchType;
typedef vector<vector<vector<vector<float>>>> filterBatchType;
typedef vector<int> strideType;

class conv2d
{
public:

	conv2d(imageBatchType& _input, filterBatchType& _filter, strideType& _strides, string _padding, string _data_format);

	imageBatchType* run(int pthreadNumber);


private:

	bool validInput;			//weather input data is valid

	imageBatchType* input;		//pointer to input data
	filterBatchType* filter;	//pointer to filter data
	strideType* strides;		//pointer to strides data
	string padding;				//the type of padding algorithm to use
	string data_format;			//the data format of the input and output data

	imageBatchType* output;		//pointer to output data

	int input_batchSize;		//batch size of input data
	int input_heightSize;		//height size of input data
	int input_widthSize;		//width size of input data
	int input_channelsSize;		//channels size of input data

	int filter_heightSize;		//height size of filter
	int filter_widthSize;		//width size of filter
	int filter_inChannelsSize;	//in_channels size of filter
	int filter_outChannelsSize;	//out_channels size of filter

	int stride_batch;			//the stride of the sliding window for batch dimension of input
	int stride_height;			//the stride of the sliding window for height dimension of input
	int stride_width;			//the stride of the sliding window for width dimension of input
	int stride_channels;		//the stride of the sliding window for channels dimension of input

	int output_batchSize;		//batch size of output data
	int output_heightSize;		//height size of output data
	int output_widthSize;		//width size of output data
	int output_channelsSize;	//channels size of output data

	int pad_top;				//padding for top
	int pad_down;				//padding for down
	int pad_left;				//padding for left
	int pad_right;				//padding for right
	int pad_needed_height;		//needed padding for height dimension
	int pad_needed_width;		//needed padding for width dimension



	/* process convolution of single filter */
	void conv2dSingleFilter(singleImage& input, filterBatchType& filter, int filterId, singleImage& output, int outputId);

	/* main part of each pthread */
	void pthreadFunction(int jobId_start, int jobId_end)
	{
		cout<<"current thread's jobId_start:"<<jobId_start<<", jobId_end:"<<jobId_end<<"!"<<endl;
		for(int jId = jobId_start; jId < jobId_end; ++jId)
		{
			int batchId = jId / filter_outChannelsSize;
			int filterId = jId % filter_outChannelsSize;
			conv2dSingleFilter((*input)[batchId], (*filter), filterId, (*output)[batchId], filterId);
		}
	}

	/* check weather input data is empty */
	bool inputCheck_input(imageBatchType& input)
	{
		if(input.size() == 0 || input[0].size() == 0 || input[0][0].size() == 0 || input[0][0][0].size() == 0)
		{
			cout<<"Error: input data is empty!"<<endl;
			return false;
		}
		return true;
	}

	/* check weather filter is empty */
	bool inputCheck_filter(filterBatchType& filter)
	{
		if(filter.size() == 0 || filter[0].size() == 0 || filter[0][0].size() == 0 || filter[0][0][0].size() == 0)
		{
			cout<<"Error: filter is empty!"<<endl;
			return false;
		}
		return true;
	}

	/* check weather strides is a 1-D tensor of length 4 */
	bool inputCheck_strides(strideType& strides)
	{
		if(strides.size() != 4)
		{
			cout<<"Error: strides is not a 1-D tensor of length 4!"<<endl;
			return false;
		}
		return true;
	}

	/* check weather padding is 'SAME' or 'VALID' */
	bool inputCheck_padding(string padding)
	{
		if(padding != "SAME" && padding != "VALID")
		{
			cout<<"Error: invalid padding input(should be 'SAME' or 'VALID')!"<<endl;
			return false;
		}
		return true;
	}

	/* check weather data_format is 'NHWC' or 'NCHW' */
	bool inputCheck_dataFormat(string data_format)
	{
		if(data_format != "NHWC" && data_format != "NCHW")
		{
			cout<<"Error: invalid data_format input(should be 'NHWC' or 'NCHW')!"<<endl;
			return false;
		}
		return true;
	}

};


#endif
