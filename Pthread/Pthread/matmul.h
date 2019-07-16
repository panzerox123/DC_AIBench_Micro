//matmul.h

#ifndef _MATMUL_H
#define _MATMUL_H

#define MATMUL_TEST

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
class matmul
{
public:
	matmul(tensor<T>* _operand1, tensor<T>* _operand2)
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

		if(dim1 != 2 || dim2 != 2)
		{
			cout<<"Error(matmul): dimension of two operands should be 2!"<<endl;
			validInput = false;
			return ;
		}


		vector<int> shape1(operand1->get_shape());
		vector<int> shape2(operand2->get_shape());
		if(shape1[1] != shape2[0])
		{
			cout<<"Error(matmul): second dimension of first operand should be equal to first dimension of second operand!"<<endl;
			validInput = false;
			return ;
		}
		
		dimension = 2;
		shape.clear();
		shape.push_back(shape1[0]);
		shape.push_back(shape2[1]);
		data.clear();
		data.resize(shape[0] * shape[1]);
		
	}


	void pthreadFunction(int pos_start, int pos_end)
	{
		vector<T>* data1 = operand1->get_data();
		vector<T>* data2 = operand2->get_data();

		vector<int> shape1(operand1->get_shape());
		vector<int> shape2(operand2->get_shape());
		
		for(int row1 = pos_start; row1 < pos_end; ++row1)
		{
			int head1 = row1 * shape1[1];
			for(int col2 = 0; col2 < shape2[1]; ++col2)
			{
				T sum = 0;
				for(int pos = 0; pos < shape1[1]; ++pos)
				{
					sum += (*data1)[head1 + pos] * (*data2)[col2 + pos * shape2[1]];
				}
				data[row1 * shape[1] + col2] = sum;
			}
		}
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

		pthreadNumber = pthreadNumber > shape[0] ? shape[0] : pthreadNumber;
		int jobsNumber = shape[0];
		int jobsForEachPthread = jobsNumber / pthreadNumber;

		/* create threads and allot jobs */
		vector<thread> pthreads(pthreadNumber);
		//thread pthreads[pthreadNumber];
		for(int i = 0; i < pthreadNumber-1; ++i)
		{
			pthreads[i] = thread(&matmul<T>::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
		}
		pthreads[pthreadNumber-1] = thread(&matmul<T>::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

		cout<<pthreadNumber<<" threads are created!"<<endl;
		/* wait for all threads completing its jobs */
		for(int i = 0; i < pthreadNumber; ++i)
			pthreads[i].join();

		///remember to delete after use
		return (new tensor<T>(dimension, shape, data));
	}

	/* data */
	tensor<T>* operand1;
	tensor<T>* operand2;

	int dimension;
	vector<int> shape;
	vector<T> data;

	bool validInput;

};

#endif
