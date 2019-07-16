//filename: avg_pool.cpp


#include "avg_pool.h"

using namespace std;

/**
 * constructor of avg_pool
 *
 * _input: reference of input data
 * _ksize: reference of ksize
 * _strides: reference of strides
 * _padding: padding algorithm
 * _data_format: data format of input and output data
 */
avg_pool::avg_pool(imageBatchType& _input, ksizeType& _ksize, strideType& _strides, string _padding, string _data_format)
{
	validInput = inputCheck_input(_input) && inputCheck_ksize(_ksize) && inputCheck_strides(_strides) && inputCheck_padding(_padding) && inputCheck_dataFormat(_data_format);

	if(validInput == true)
	{
		input = &_input;
		ksize = &_ksize;
		strides = &_strides;
		padding = _padding;
		data_format = _data_format;

		/* input and strides configuration */
		if(data_format == "NHWC")
		{
			input_batchSize = (*input).size();
			input_heightSize = (*input)[0].size();
			input_widthSize = (*input)[0][0].size();
			input_channelsSize = (*input)[0][0][0].size();

			stride_batch = (*strides)[0];
			stride_height = (*strides)[1];
			stride_width = (*strides)[2];
			stride_channels = (*strides)[3];

			ksize_batch = (*ksize)[0];
			ksize_height = (*ksize)[1];
			ksize_width = (*ksize)[2];
			ksize_channels = (*ksize)[3];
		}
		else
		{
			input_batchSize = (*input).size();
			input_channelsSize = (*input)[0].size();
			input_heightSize = (*input)[0][0].size();
			input_widthSize = (*input)[0][0][0].size();

			stride_batch = (*strides)[0];
			stride_channels = (*strides)[1];
			stride_height = (*strides)[2];
			stride_width = (*strides)[3];

			ksize_batch = (*ksize)[0];
			ksize_channels = (*ksize)[1];
			ksize_height = (*ksize)[2];
			ksize_width = (*ksize)[3];
		}


		/* output and padding configuration */
		if(padding == "VALID")
		{
			output_batchSize = input_batchSize;
			output_heightSize = static_cast<int>(ceil((static_cast<double>(input_heightSize - ksize_height) + 1.0) / stride_height));
			output_widthSize = static_cast<int>(ceil((static_cast<double>(input_widthSize - ksize_width) + 1.0) / stride_width));
			output_channelsSize = input_channelsSize;

			pad_needed_height = 0;
			pad_needed_width = 0;
			pad_top = 0;
			pad_down = 0;
			pad_left = 0;
			pad_right = 0;
		}
		else
		{
			output_batchSize = input_batchSize;
			output_heightSize = static_cast<int>(ceil((static_cast<double>(input_heightSize) / stride_height)));
			output_widthSize = static_cast<int>(ceil((static_cast<double>(input_widthSize) / stride_width)));
			output_channelsSize = input_channelsSize;

			pad_needed_height = (output_heightSize - 1) * stride_height + ksize_height - input_heightSize;
			pad_needed_width = (output_widthSize - 1) * stride_width + ksize_width - input_widthSize;
			pad_top = pad_needed_height / 2;
			pad_down = pad_needed_height - pad_top;
			pad_left = pad_needed_width / 2;
			pad_right = pad_needed_width - pad_left;
		}
		
		/* malloc output data memory */
		if(data_format == "NHWC")
		{
			vector<float> oneDimension(output_channelsSize, 0.0);
			vector<vector<float>> twoDimensions(output_widthSize, oneDimension);
			vector<vector<vector<float>>> threeDimensions(output_heightSize, twoDimensions);
			output = new vector<vector<vector<vector<float>>>>(output_batchSize, threeDimensions);
		}
		else
		{
			vector<float> oneDimension(output_widthSize, 0.0);
			vector<vector<float>> twoDimensions(output_heightSize, oneDimension);
			vector<vector<vector<float>>> threeDimensions(output_channelsSize, twoDimensions);
			output = new vector<vector<vector<vector<float>>>>(output_batchSize, threeDimensions);
		}

		if(output == NULL)
			cout<<"Error: mallocing output data memory failed!"<<endl;
	}
/*
	cout<<"constrution complete!"<<endl;
	cout<<"input:"<<endl;
	cout<<"N:"<<input_batchSize<<", C:"<<input_channelsSize<<", H:"<<input_heightSize<<", W:"<<input_widthSize<<endl;
	cout<<"ksize:"<<endl;
	cout<<"N:"<<ksize_batch<<", c:"<<ksize_channels<<", H:"<<ksize_height<<", W:"<<ksize_width<<endl;
	cout<<"strides:"<<endl;
	cout<<"N:"<<stride_batch<<", C:"<<stride_channels<<", H:"<<stride_height<<", W:"<<stride_width<<endl;
	cout<<"output:"<<endl;
	cout<<"N:"<<output_batchSize<<", C:"<<output_channelsSize<<", H:"<<output_heightSize<<", W:"<<output_widthSize<<endl;
*/
}


/**
 * main part of each pthread 
 *
 * batchId_start: start of the batch for current thread
 * batchId_end: end of the batch for current thread
 */
void avg_pool::pthreadFunction(int batchId_start, int batchId_end)
{
	cout<<"current thread's batchId_start:"<<batchId_start<<", batchId_end:"<<batchId_end<<"!"<<endl;
	for(int batchId = batchId_start; batchId < batchId_end; ++batchId)
	{
		avg_poolSingleImage((*input)[batchId], (*output)[batchId]);
	}
}


/**
 * run avg_pool task
 *
 * pthreadNumber: number of threads for current task
 */
imageBatchType* avg_pool::run(int pthreadNumber)
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
	vector<thread> pthreads(pthreadNumber);
	//thread pthreads[pthreadNumber];
	for(int i = 0; i < pthreadNumber-1; ++i)
	{
		pthreads[i] = thread(&avg_pool::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread);
	}
	pthreads[pthreadNumber-1] = thread(&avg_pool::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber);

	cout<<pthreadNumber<<" threads are created!"<<endl;
	/* wait for all threads completing its jobs */
	for(int i = 0; i < pthreadNumber; ++i)
		pthreads[i].join();

	return output;
}



/**
 * avg_pool for single image
 *
 * input: reference of a single input image data
 * output: reference of a single output image data
 */
void avg_pool::avg_poolSingleImage(singleImage& input, singleImage& output)
{
	//cout<<"avg_poolSingleImage!"<<endl;
	/* if padding algorithm is 'VALID' */
	if(padding == "VALID")
	{
		/* if data_format is 'NHWC' */
		if(data_format == "NHWC")
		{
			for(int pos_height = 0; pos_height < output_heightSize; ++pos_height)
			{
				for(int pos_width = 0; pos_width < output_widthSize; ++pos_width)
				{
					/* avg_pool for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* average number among current window */
						float subSum = 0;
						for(int h = 0; h < ksize_height; ++h)
						for(int w = 0; w < ksize_width; ++w)
							subSum += input[h + offset_height][w + offset_width][offset_channel];
						output[pos_height][pos_width][pos_channel] = subSum / (ksize_height * ksize_width);
					}
					
				}
			}
		}
		/* if data_format is 'NCHW' */
		else
		{
			for(int pos_height = 0; pos_height < output_heightSize; ++pos_height)
			{
				for(int pos_width = 0; pos_width < output_widthSize; ++pos_width)
				{
					/* avg_pool for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* average number among current window */
						float subSum = 0;
						for(int h = 0; h < ksize_height; ++h)
						for(int w = 0; w < ksize_width; ++w)
							subSum += input[offset_channel][h + offset_height][w + offset_width];
						output[pos_channel][pos_height][pos_width] = subSum / (ksize_height * ksize_width);
					}
					
				}
			}
		}//end of 'BHWC' data_format
	}//end of 'VALID' padding algorithm
	/* if padding algorithm is 'SAME' */
	else
	{
		/* if data_format is 'NHWC' */
		if(data_format == "NHWC")
		{
			/* bound of input in padded_input */
			int bound_top = pad_top;
			int bound_down = pad_top + input_heightSize - 1;
			int bound_left = pad_left;
			int bound_right = pad_left + input_widthSize - 1;

			for(int pos_height = 0; pos_height < output_heightSize; ++pos_height)
			{
				for(int pos_width = 0; pos_width < output_widthSize; ++pos_width)
				{
					/* avg_pool for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* average number among current window */
						float subSum = 0;
						for(int h = 0; h < ksize_height; ++h)
						for(int w = 0; w < ksize_width; ++w)
						{
							/* position in padded_input */
							int newPos_height = h + offset_height;
							int newPos_width = w + offset_width;
							/* check weather current point is in padded area */
							if(newPos_height >= bound_top && newPos_height <= bound_down && newPos_width >= bound_left && newPos_width <= bound_right)
							{
								subSum += input[newPos_height - pad_top][newPos_width - pad_left][offset_channel];
							}
						}
						output[pos_height][pos_width][pos_channel] = subSum / (ksize_height * ksize_width);
					}

				}
			}

		}
		/* if data_format is 'NCHW' */
		else
		{
			/* bound of input in padded_input */
			int bound_top = pad_top;
			int bound_down = pad_top + input_heightSize - 1;
			int bound_left = pad_left;
			int bound_right = pad_left + input_widthSize - 1;

			for(int pos_height = 0; pos_height < output_heightSize; ++pos_height)
			{
				for(int pos_width = 0; pos_width < output_widthSize; ++pos_width)
				{
					/* avg_pool for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* average number among current window */
						float subSum = 0;
						for(int h = 0; h < ksize_height; ++h)
						for(int w = 0; w < ksize_width; ++w)
						{
							/* position in padded_input */
							int newPos_height = h + offset_height;
							int newPos_width = w + offset_width;
							/* check weather current point is in padded area */
							if(newPos_height >= bound_top && newPos_height <= bound_down && newPos_width >= bound_left && newPos_width <= bound_right)
							{
								subSum += input[offset_channel][newPos_height - pad_top][newPos_width - pad_left];
							}
						}
						output[pos_channel][pos_height][pos_width] = subSum / (ksize_height * ksize_width);
					}
					
				}
			}
		}//end of 'NCHW' data_format
	}//end of 'SAME' padding algorithm

	return ;
}


void thread_avgPoolBatch(imageBatchType* input, ksizeType& ksize, strideType& strides, string padding, string data_format, int pthreadCount)
{
	avg_pool newPool((*input), ksize, strides, padding, data_format);
	imageBatchType* output = newPool.run(pthreadCount);

	delete input;
}



#ifdef AVG_POOL_TEST

int main(int argc, char *argv[])
{
/*
	float a1[] = {0, 1, 1, 2, 2};
	vector<float> v1(a1, a1+5);
	float a2[] = {0, 1, 1, 0, 0};
	vector<float> v2(a2, a2+5);
	float a3[] = {1, 1, 0, 1, 0};
	vector<float> v3(a3, a3+5);
	float a4[] = {1, 0, 1, 1, 1};
	vector<float> v4(a4, a4+5);
	float a5[] = {0, 2, 0, 1, 0};
	vector<float> v5(a5, a5+5);

	vector<vector<float>> vv1;
	vv1.push_back(v1);
	vv1.push_back(v2);
	vv1.push_back(v3);
	vv1.push_back(v4);
	vv1.push_back(v5);


	float b1[] = {1, 1, 1, 2, 0};
	vector<float> x1(b1, b1+5);
	float b2[] = {0, 2, 1, 1, 2};
	vector<float> x2(b2, b2+5);
	float b3[] = {1, 2, 0, 0, 2};
	vector<float> x3(b3, b3+5);
	float b4[] = {0, 2, 1, 2, 1};
	vector<float> x4(b4, b4+5);
	float b5[] = {2, 0, 1, 2, 0};
	vector<float> x5(b5, b5+5);

	vector<vector<float>> vv2;
	vv2.push_back(x1);
	vv2.push_back(x2);
	vv2.push_back(x3);
	vv2.push_back(x4);
	vv2.push_back(x5);


	float c1[] = {2, 0, 2, 0, 2};
	vector<float> y1(c1, c1+5);
	float c2[] = {0, 0, 1, 2, 1};
	vector<float> y2(c2, c2+5);
	float c3[] = {1, 0, 2, 2, 1};
	vector<float> y3(c3, c3+5);
	float c4[] = {2, 0, 2, 0, 0};
	vector<float> y4(c4, c4+5);
	float c5[] = {0, 0, 1, 1, 2};
	vector<float> y5(c5, c5+5);

	vector<vector<float>> vv3;
	vv3.push_back(y1);
	vv3.push_back(y2);
	vv3.push_back(y3);
	vv3.push_back(y4);
	vv3.push_back(y5);


	vector<vector<vector<float>>> vvv1;
	vvv1.push_back(vv1);
	vvv1.push_back(vv2);
	vvv1.push_back(vv3);

	vector<vector<vector<vector<float>>>> input;
	input.push_back(vvv1);
*/

	string dirPath = argv[1];				//dir path
	string data_format = argv[2];			//data format
	int pthreadCount = atoi(argv[3]);		//pthread number
	int rows = atoi(argv[4]);				//row size
	int cols = atoi(argv[5]);				//col size
	int batch_size = atoi(argv[6]);			//batch size
	//loadImage_batch(dirPath, data_format, rows, cols, input);

	/* ksize */
	int k[] = {1, 1, 3, 3};
	vector<int> ksize(k, k+4);
	


	/* strides */
	int s[] = {1, 1, 2, 2};
	vector<int> strides(s, s+4);

	/* padding */
	string padding = "SAME";

	

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
			threads.push_back(thread(thread_avgPoolBatch, newImageBatch, ref(ksize), ref(strides), padding, data_format, pthreadCount));
			counter = 0;
			input.clear();
		}
	}
	if(counter != 0)
	{
		imageBatchType* newImageBatch = new imageBatchType(input);
		threads.push_back(thread(thread_avgPoolBatch, newImageBatch, ref(ksize), ref(strides), padding, data_format, pthreadCount));
		counter = 0;
		input.clear();
	}

	closedir(dir);

	for(auto& th : threads)
		th.join();


	return 0;
}
	
#endif
