//filename: batch_norm.cpp

/**
 * function: batch normalization
 * compile eg: g++ -std=c++11 batch_norm.cpp -o batch_norm -lopencv_imgcodecs -g
 * running eg: ./batch_norm $dirPath $data_format $pthreadNumber $rows $cols $batchSize
 */


#include "batch_norm.h"

using namespace std;

batch_norm::batch_norm(imageBatchType& _input, meanType& _mean, varianceType& _variance, float _offset, float _scale, float _variance_epsilon)
{
	validInput = inputCheck_input(_input);

	if(validInput == true)
	{
		input = &_input;
		mean = &_mean;
		variance = &_variance;
		offset = _offset;
		scale = _scale;
		variance_epsilon = _variance_epsilon;
			
		input_batchSize = _input.size();
		input_heightSize = _input[0].size();
		input_widthSize = _input[0][0].size();
		input_channelsSize = _input[0][0][0].size();	
		
		vector<float> oneDimension(input_channelsSize, 0.0);
		vector<vector<float>> twoDimensions(input_widthSize, oneDimension);
		vector<vector<vector<float>>> threeDimensions(input_heightSize, twoDimensions);
		output = new vector<vector<vector<vector<float>>>>(input_batchSize, threeDimensions);
		
	}
	else
	{
		output = NULL;
	}
}


/**
 * run convolution task
 *
 * pthreadNumber: number of threads for current task
 */
imageBatchType* batch_norm::run(int pthreadNumber)
{
	if(validInput == false)
		return NULL;
	
	if(pthreadNumber <= 0)
	{
		cout<<"Error: pthread number should be a positive integer!"<<endl;
		return NULL;
	}

	int jobsNumber = input_batchSize;
	int jobsForEachPthread = jobsNumber / pthreadNumber;

	/* create threads and allot jobs */
	vector<thread> pthreads;
	//thread pthreads[pthreadNumber];
	for(int i = 0; i < pthreadNumber-1; ++i)
	{
		pthreads.push_back(thread(&batch_norm::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread));
	}
	pthreads.push_back(thread(&batch_norm::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber));

	cout<<pthreadNumber<<" threads are created!"<<endl;
	/* wait for all threads completing its jobs */
	for(auto& th : pthreads)
		th.join();

	return output;
}


void batch_norm::norm_SingleImage(singleImage& input, meanType& mean, varianceType& variance, int id)
{
	for(int h = 0; h < input_heightSize; ++h)
	{
		for(int w = 0; w < input_widthSize; ++w)
		{
			for(int c = 0; c < input_channelsSize; ++c)
			{
				float v = variance[id][h][w];
				if(v == 0)
					v = variance_epsilon;
				(*output)[id][h][w][c] = scale * (input[h][w][c] - mean[id][h][w])/v + offset;
			}
		}
	}
}



void thread_Batch_norm(imageBatchType* input, int pthreadCount)
{
	int batchSize = (*input).size();
	int heightSize = (*input)[0].size();
	int widthSize = (*input)[0][0].size();

	vector<float> oneDimension(widthSize, 0.0);
	vector<vector<float>> twoDimensions(heightSize, oneDimension);
	vector<vector<vector<float>>> mean_tmp(batchSize, twoDimensions);

	vector<float> oneDimension2(widthSize, 1);
	vector<vector<float>> twoDimensions2(heightSize, oneDimension2);
	vector<vector<vector<float>>> variance_tmp(batchSize, twoDimensions2);

	batch_norm newBatch_norm((*input), mean_tmp, variance_tmp, 1, 1, 1);
	imageBatchType* output = newBatch_norm.run(pthreadCount);
	delete input;
}


#ifdef BATCH_NORM_TEST

int main(int argc, char *argv[])
{

	string dirPath = argv[1];				//dir path
	string data_format = "NHWC";			//data format
	int pthreadCount = atoi(argv[2]);		//pthread number
	int rows = atoi(argv[3]);				//row size
	int cols = atoi(argv[4]);				//col size
	int batch_size = atoi(argv[5]);			//batch size


	//loadImage_batch(dirPath, data_format, rows, cols, input);


	

	struct dirent *ptr;
	DIR *dir;
	dir = opendir(dirPath.c_str());
	int counter = 0;
	imageBatchType input;

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
			imageBatchType* newImageBatch = new imageBatchType(input);
			threads.push_back(thread(thread_Batch_norm, newImageBatch, pthreadCount));
			counter = 0;
			input.clear();
		}
	}
	if(counter != 0)
	{
		imageBatchType* newImageBatch = new imageBatchType(input);
		threads.push_back(thread(thread_Batch_norm, newImageBatch, pthreadCount));
		counter = 0;
		input.clear();
	}
	
	closedir(dir);

	for(auto& th : threads)
		th.join();

	return 0;
}
	
#endif
