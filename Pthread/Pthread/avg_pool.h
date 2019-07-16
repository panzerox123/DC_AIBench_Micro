//avg_pool.h

#ifndef _AVG_POOL_H
#define _AVG_POOL_H

#define AVG_POOL_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include "loadImage.h"

using namespace std;

const float FLOAT_MIN = - FLT_MAX;

typedef vector<vector<vector<float>>> singleImage;
typedef vector<vector<vector<vector<float>>>> imageBatchType;
typedef vector<int> ksizeType;
typedef vector<int> strideType;


class avg_pool
{
public:

	avg_pool(imageBatchType& _input, ksizeType& _ksize, strideType& _strides, string _padding, string _data_format);

	imageBatchType* run(int pthreadNumber);


private:

	bool validInput;			//weather input data is valid

	imageBatchType* input;		//pointer to input data
	ksizeType* ksize;			//pointer to ksize data
	strideType* strides;		//pointer to strides data
	string padding;				//the type of padding algorithm to use
	string data_format;			//the data format of the input and output data

	imageBatchType* output;		//pointer to output data

	int input_batchSize;		//batch size of input data
	int input_heightSize;		//height size of input data
	int input_widthSize;		//width size of input data
	int input_channelsSize;		//channels size of input data

	int ksize_batch;			//height size of filter
	int ksize_height;			//width size of filter
	int ksize_width;			//in_channels size of filter
	int ksize_channels;			//out_channels size of filter

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



	/* process avg_pool of single image */
	void avg_poolSingleImage(singleImage& input, singleImage& output);

	/* main part of each pthread */
	void pthreadFunction(int jobId_start, int jobId_end);

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

	/* check weather ksize is empty */
	bool inputCheck_ksize(ksizeType& ksize)
	{
		if(ksize.size() != 4)
		{
			cout<<"Error: ksize is not a 1-D tensor of length 4!"<<endl;
			return false;
		}

		if(ksize[0] != 1)
		{
			cout<<"Error: batch dimension of ksize should be 1!"<<endl;
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

		if(strides[0] != 1)
		{
			cout<<"Error: batch dimension of strides should be 1!"<<endl;
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
