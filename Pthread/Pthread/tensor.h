//tensor.h

#ifndef _TENSOR_H
#define _TENSOR_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cfloat>

using namespace std;


template <typename T>
class tensor
{
public:

	tensor(tensor<T>* newTensor)
	{
		if(newTensor == this)
			return ;

		dimension = newTensor->dimension;
		shape = new vector<int>(*(newTensor->shape));
		data = new vector<T>(*(newTensor->data));
		weight = new vector<long>(*(newTensor->weight));
		validInput = newTensor->validInput;
	}

	/* constructor of tensor */
	tensor(int _dimension, vector<int>& _shape, vector<T>& _data)
	{
		validInput = true;
		weight = NULL;

		/* check weather dimension is a positive integer */
		if(_dimension < 0)
		{
			cout<<"Error: dimension should be a positive integer!"<<endl;
			validInput = false;
			return ;
		}
		
		/* check weather dimension matches to shape */
		if(_shape.size() != _dimension)
		{
			cout<<"Error: dimension dose not match to shape!"<<endl;
			validInput = false;
			return ;
		}
	
		/* check weather size of each dimension is a positive integer, and get element number according to shape */
		long elementNumber = 1;
		for(auto s : _shape)
		{
			if(s <= 0)
			{
				cout<<"Error: element of shape should be a positive integer!"<<endl;
				validInput = false;
				return ;
			}
			else
				elementNumber *= s;
		}

		/* check weather element number of data is the same as that destribed by shape */
		if(elementNumber != _data.size())
		{
			cout<<"Error: shape dose not match to length of elements!"<<endl;
			validInput = false;
			return ;
		}
			
		/* input is valid */	
		dimension = _dimension;
		shape = new vector<int>(_shape);
		data = new vector<T>(_data);

		weight = new vector<long>(dimension, 1);
		for(int di = dimension-2; di >= 0; --di)
			(*weight)[di] = (*weight)[di+1] * (*shape)[di+1];

	}

	
	~tensor()
	{
		if(shape != NULL)
			delete shape;

		if(weight != NULL)
			delete weight;

		if(data != NULL)
			delete data;
	}

	T* get_element(const vector<int>& position)
	{
		if(position.size() != dimension)
		{
			cout<<"Error: dimension of position is not corrent!"<<endl;
			return NULL;
		}

		long pos = 0;
		for(int di = 0; di < dimension; ++di)
		{
			if(position[di] <= 0)
			{
				cout<<"Error: value of position of each dimension should be a positive integer!"<<endl;
				return NULL;
			}
			if(position[di] >= (*shape)[di])
			{
				cout<<"Error: out of range!"<<endl;
				return NULL;
			}
			pos += position[di] * (*weight)[di];
		}
		return &((*data)[pos]);
	}

	int get_dimension()
	{
		return dimension;
	}

	vector<int> get_shape()
	{
		return *shape;
	}


	void change_shape(const vector<int>& newShape)
	{
		int len = newShape.size();
		if(len == 0)
		{
			cout<<"Error: new shape is empty!"<<endl;
			return ;
		}

		long elementNumber = 1;
		for(auto s : newShape)
		{
			if(s <= 0)
			{
				cout<<"Error: element of shape should be a positive integer!"<<endl;
				return ;
			}
			else
				elementNumber *= s;
		}	
		if(elementNumber != data->size())
		{
			cout<<"Error: element number of new shape dose not match the original!"<<endl;
			return ;
		}
		if(shape != NULL)
			delete shape;
		shape = new vector<int>(newShape);

	}

	/* only friend class or function can get data */
	vector<T>* get_data()
	{
		return data;
	}


private:

	int dimension;
	vector<int>* shape;
	vector<long>* weight;
	vector<T>* data; 
	bool validInput;

};

#endif
