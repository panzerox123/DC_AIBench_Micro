//add.h

#ifndef _ADD_H
#define _ADD_H

#define ADD_TEST

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include "tensor.h"
#include "loadImage.h"

using namespace std;

template <typename T>
class addOperator
{
public:
	addOperator(tensor<T>* _operand1, tensor<T>* _operand2)
	{
		operand1 = _operand1;
		operand2 = _operand2;
		validInput = true;
		if(operand1 == NULL || operand2 == NULL)
		{
			validInput = false;
			return ;
		}

		int dim1 = operand1->get_dimension();
		int dim2 = operand2->get_dimension();

		if(dim1 != dim2)
		{
			validInput = false;
			return ;
		}

		vector<int> shape1(operand1->get_shape());
		vector<int> shape2(operand2->get_shape());
		for(int i = 0; i < dim1; ++i)
		{
			if(shape1[i] != shape2[i])
			{
				validInput = false;
				return ;
			}
		}

		output = new tensor<T>(operand1);
	}


	void pthreadFunction(int pos_start, int pos_end)
	{
		vector<T>* data1 = operand1->get_data();
		vector<T>* data2 = operand2->get_data();
		vector<T>* data_output = output->get_data();
		
		for(int pos = pos_start; pos < pos_end; ++pos)
			(*data_output)[pos] = (*data1)[pos] + (*data2)[pos];
	}

	tensor<T>* run(int pthreadNumber)
	{
		if(validInput == false)
			return NULL;

		if(pthreadNumber <= 0)
		{
			cout<<"Error: pthread number should be a positive integer!"<<endl;
			return NULL;
		}

		int jobsNumber = (output->get_data())->size();
		int jobsForEachPthread = jobsNumber / pthreadNumber;

		/* create threads and allot jobs */
		vector<thread> pthreads(pthreadNumber);
		//thread pthreads[pthreadNumber];
		for(int i = 0; i < pthreadNumber-1; ++i)
		{
			pthreads[i] = thread(&addOperator<T>::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads[pthreadNumber-1] = thread(&addOperator<T>::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		cout<<pthreadNumber<<" threads are created!"<<endl;
		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads[i].join();

		return output;
	}

	/* data */
	tensor<T>* operand1;
	tensor<T>* operand2;
	tensor<T>* output;

	bool validInput;

};

#endif
