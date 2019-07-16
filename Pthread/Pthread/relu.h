//relu.h

#ifndef _RELU_H
#define _RELU_H

#define RELU_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include "loadImage.h"
#include "tensor.h"

using namespace std;

template <typename T>
class relu
{
public:
	relu(tensor<T>* _features)
	{
		features = new tensor<T>(_features);
	}
	~relu()
	{
		if(features != NULL)
			delete features;
	}

	void pthreadFunction(int pos_start, int pos_end)
	{
		vector<T>* data = features->get_data();
		if(data == NULL)
			return ;
		for(int pos = pos_start; pos < pos_end; ++pos)
			if((*data)[pos] < 0)
				(*data)[pos] = 0;
	}

	tensor<T>* run(int pthreadNumber)
	{
		if(features == NULL || features->get_data() == NULL)
			return NULL;

		if(pthreadNumber <= 0)
		{
			cout<<"Error: pthread number should be a positive integer!"<<endl;
			return NULL;
		}

		int jobsNumber = (features->get_data())->size();
		int jobsForEachPthread = jobsNumber / pthreadNumber;

		/* create threads and allot jobs */
		vector<thread> pthreads(pthreadNumber);
		//thread pthreads[pthreadNumber];
		for(int i = 0; i < pthreadNumber-1; ++i)
		{
			pthreads[i] = thread(&relu<T>::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads[pthreadNumber-1] = thread(&relu<T>::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		cout<<pthreadNumber<<" threads are created!"<<endl;
		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads[i].join();

		return features;
	}

	/* data */
	tensor<T>* features;
};

#endif
