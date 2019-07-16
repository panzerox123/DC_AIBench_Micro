//multiply.cpp

#include "multiply.h"

void threadBatch(batchImage* input, int pthreadCount)
{
	vector<float> data;
	fourDim2oneDim((*input), data);

	vector<int> shape;
	shape.push_back((*input).size() * (*input)[0].size());
	shape.push_back((*input)[0][0].size() * (*input)[0][0][0].size());

	tensor<float> tensor1(2, shape, data);
	tensor<float> tensor2(2, shape, data);

	multiplyOperator<float> newMultiply(&tensor1, &tensor2);
	tensor<float>* output = newMultiply.run(pthreadCount);

	delete input;
}


int main(int argc, char *argv[])
{
	string dirPath = argv[1];				//dir path
	string data_format = "NHWC";			//data format
	int pthreadCount = atoi(argv[2]);		//pthread number
	int rows = atoi(argv[3]);				//row size
	int cols = atoi(argv[4]);				//col size
	int batch_size = atoi(argv[5]);			//batch size


	struct dirent *ptr;
	DIR *dir;
	dir = opendir(dirPath.c_str());

	int counter = 0;
	batchImage input;
	vector<thread> threads;

	while((ptr = readdir(dir)) != NULL)
	{
		if(ptr->d_name[0] == '.')
			continue;
		counter ++;
		string filename = dirPath + ptr->d_name;
		singleImage newImage;
		loadImage_single(filename, data_format, rows, cols, newImage);
		input.push_back(newImage);

		if(counter == batch_size)
		{
			batchImage* newImageBatch = new batchImage(input);
			threads.push_back(thread(threadBatch, newImageBatch, pthreadCount));
			counter = 0;
			input.clear();
		}
	}
	if(counter != 0)
	{
		batchImage* newImageBatch = new batchImage(input);
		threads.push_back(thread(threadBatch, newImageBatch, pthreadCount));
		counter = 0;
		input.clear();
	}
	
	closedir(dir);

	for(auto& th : threads)
		th.join();
	return 0;
}

