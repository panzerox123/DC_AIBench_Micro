//dropout.h

#ifndef _DROPOUT_H
#define _DROPOUT_H

#define DROPOUT_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include "loadImage.h"
#include "tensor.h"

using namespace std;

float randFloat()
{
	srand(time(NULL));

	return (float)(rand() / (float)RAND_MAX);
}

template <typename T>
class dropout
{
public:
	dropout(tensor<T>* _features, float _keep_prob)
	{
		features = NULL;

		if(_keep_prob > 0 && _keep_prob <= 1)
		{
			features = new tensor<T>(_features);
			keep_prob = _keep_prob;
		}
		else
		{
			cout<<"keep_prob should be in (0, 1] !"<<endl;
		}
	}

	~dropout()
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
		{
			float keep = randFloat();
			if(keep < keep_prob)
				(*data)[pos] *= 1 / keep_prob;
			else
				(*data)[pos] = 0;
		}
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
			pthreads[i] = thread(&dropout<T>::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads[pthreadNumber-1] = thread(&dropout<T>::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		cout<<pthreadNumber<<" threads are created!"<<endl;
		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads[i].join();

		return features;
	}

	/* data */
	tensor<T>* features;
	float keep_prob;
};

#endif
