//softmax.h

#ifndef _SOFTMAX_H
#define _SOFTMAX_H

#define SOFTMAX_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <mutex>
#include "loadImage.h"
#include "tensor.h"

using namespace std;

template <typename T>
class softmax
{
public:
	softmax(tensor<T>* _logits)
	{
		logits = new tensor<T>(_logits);
		sum = 0;
	}

	~softmax()
	{
		if(logits != NULL)
			delete logits;
	}

	void pthreadFunction_pre(int pos_start, int pos_end)
	{
		vector<T>* data = logits->get_data();
		if(data == NULL)
			return ;
		T temp_sum = 0;
		for(int pos = pos_start; pos < pos_end; ++pos)
		{
			(*data)[pos] = exp((*data)[pos]);
			temp_sum += (*data)[pos];
		}
		lock_guard<mutex> guard(sum_mutex);
		sum += temp_sum;
	}

	void pthreadFunction_next(int pos_start, int pos_end)
	{
		vector<T>* data = logits->get_data();
		if(data == NULL)
			return ;
		for(int pos = pos_start; pos < pos_end; ++pos)
		{
			if(sum != 0)
			(*data)[pos] = (*data)[pos] / sum;
		}
	}

	tensor<T>* run(int pthreadNumber)
	{
		if(logits == NULL || logits->get_data() == NULL)
			return NULL;

		if(pthreadNumber <= 0)
		{
			cout<<"Error: pthread number should be a positive integer!"<<endl;
			return NULL;
		}

		int jobsNumber = (logits->get_data())->size();
		int jobsForEachPthread = jobsNumber / pthreadNumber;

		/* create threads and allot jobs */
		vector<thread> pthreads_pre(pthreadNumber);
		for(int i = 0; i < pthreadNumber-1; ++i)
		{
			pthreads_pre[i] = thread(&softmax<T>::pthreadFunction_pre, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads_pre[pthreadNumber-1] = thread(&softmax<T>::pthreadFunction_pre, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads_pre[i].join();


		/* create threads and allot jobs */
		vector<thread> pthreads_next(pthreadNumber);
		for(int i = 0; i < pthreadNumber-1; ++i)
		{
			pthreads_next[i] = thread(&softmax<T>::pthreadFunction_next, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads_next[pthreadNumber-1] = thread(&softmax<T>::pthreadFunction_next, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads_next[i].join();

		return logits;
	}

	/* data */
	tensor<T>* logits;
	T sum;
	mutex sum_mutex;
};

#endif
