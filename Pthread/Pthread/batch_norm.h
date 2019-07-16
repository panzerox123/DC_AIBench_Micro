//batch_norm.h

#ifndef _BATCH_NORM_H
#define _BATCH_NORM_H

#define BATCH_NORM_TEST

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
typedef vector<vector<vector<float>>> meanType;
typedef vector<vector<vector<float>>> varianceType;


class batch_norm
{
public:

	batch_norm(imageBatchType& _input, meanType& _mean, varianceType& _variance, float _offset, float _scale, float _variance_epsilon);

	imageBatchType* run(int pthreadNumber);


private:

	bool validInput;			//weather input data is valid

	imageBatchType* input;
	meanType* mean;	
	varianceType* variance;	
	float offset;		
	float scale;		
	float variance_epsilon;

	imageBatchType* output;		//pointer to output data

	int input_batchSize;		//batch size of input data
	int input_heightSize;		//height size of input data
	int input_widthSize;		//width size of input data
	int input_channelsSize;		//channels size of input data


	/* process normalization of single image */
	void norm_SingleImage(singleImage& input, meanType& mean, varianceType& variance, int id);

	/* main part of each pthread */
	void pthreadFunction(int jobId_start, int jobId_end)
	{
		//cout<<"current thread's jobId_start:"<<jobId_start<<", jobId_end:"<<jobId_end<<"!"<<endl;
		for(int jId = jobId_start; jId < jobId_end; ++jId)
			norm_SingleImage((*input)[jId], (*mean), (*variance), jId);
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
		

};


#endif
