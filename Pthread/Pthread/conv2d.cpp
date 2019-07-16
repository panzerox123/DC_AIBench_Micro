//filename: conv2d.cpp

/**
 * function: Computes a 2-D convolution given 4-D input and filter tensors
 * compile eg: g++ -std=c++11 conv2d.cpp -o conv2d -lopencv_imgcodecs -g
 * running eg: ./conv2d $dirPath $data_format $pthreadNumber $rows $cols $batchSize
 */


#include "conv2d.h"

using namespace std;

/**
 * constructor of conv2d
 *
 * _input: reference of input data
 * _filter: reference of filter
 * _strides: reference of strides
 * _padding: padding algorithm
 * _data_format: data format of input and output data
 */
conv2d::conv2d(imageBatchType& _input, filterBatchType& _filter, strideType& _strides, string _padding, string _data_format)
{
	validInput = inputCheck_input(_input) && inputCheck_filter(_filter) && inputCheck_strides(_strides) && inputCheck_padding(_padding) && inputCheck_dataFormat(_data_format);

	if(validInput == true)
	{
		input = &_input;
		filter = &_filter;
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
		}

		/* filter configuration */
		filter_heightSize = (*filter).size();
		filter_widthSize = (*filter)[0].size();
		filter_inChannelsSize = (*filter)[0][0].size();
		filter_outChannelsSize = (*filter)[0][0][0].size();


		/* output and padding configuration */
		if(padding == "VALID")
		{
			output_batchSize = input_batchSize;
			output_heightSize = static_cast<int>(ceil((static_cast<double>(input_heightSize - filter_heightSize) + 1.0) / stride_height));
			output_widthSize = static_cast<int>(ceil((static_cast<double>(input_widthSize - filter_widthSize) + 1.0) / stride_width));
			output_channelsSize = filter_outChannelsSize;

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
			output_channelsSize = filter_outChannelsSize;

			pad_needed_height = (output_heightSize - 1) * stride_height + filter_heightSize - input_heightSize;
			pad_needed_width = (output_widthSize - 1) * stride_width + filter_widthSize - input_widthSize;
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
	cout<<"filter:"<<endl;
	cout<<"H:"<<filter_heightSize<<", W:"<<filter_widthSize<<", CI:"<<filter_inChannelsSize<<", CO:"<<filter_outChannelsSize<<endl;
	cout<<"strides:"<<endl;
	cout<<"N:"<<stride_batch<<", C:"<<stride_channels<<", H:"<<stride_height<<", W:"<<stride_width<<endl;
	cout<<"output:"<<endl;
	cout<<"N:"<<output_batchSize<<", C:"<<output_channelsSize<<", H:"<<output_heightSize<<", W:"<<output_widthSize<<endl;
*/
}


/**
 * run convolution task
 *
 * pthreadNumber: number of threads for current task
 */
imageBatchType* conv2d::run(int pthreadNumber)
{
	if(validInput == false)
		return NULL;
	
	if(pthreadNumber <= 0)
	{
		cout<<"Error: pthread number should be a positive integer!"<<endl;
		return NULL;
	}

	int jobsNumber = input_batchSize * filter_outChannelsSize;
	int jobsForEachPthread = jobsNumber / pthreadNumber;

	/* create threads and allot jobs */
	vector<thread> pthreads;
	//thread pthreads[pthreadNumber];
	for(int i = 0; i < pthreadNumber-1; ++i)
	{
		pthreads.push_back(thread(&conv2d::pthreadFunction, this, i * jobsForEachPthread, (i+1) * jobsForEachPthread));
	}
	pthreads.push_back(thread(&conv2d::pthreadFunction, this, (pthreadNumber-1) * jobsForEachPthread, jobsNumber));

	cout<<pthreadNumber<<" threads are created!"<<endl;
	/* wait for all threads completing its jobs */
	for(auto& th : pthreads)
		th.join();

	return output;
}



/**
 * convolution for single filter
 *
 * input: reference of a single input image data
 * filter: reference of filter
 * filterOutChannelId: out_channel id of filter
 * output: reference of a single output image data
 * outputChannelId: out_channel id of output 
 */
void conv2d::conv2dSingleFilter(singleImage& input, filterBatchType& filter, int filterOutChannelId, singleImage& output, int outputChannelId)
{
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
					float sum = 0.0;
					/* convolution for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* mul-sum between each channel of a single filter and the related part in input */
						float subSum = 0.0;
						for(int h = 0; h < filter_heightSize; ++h)
						for(int w = 0; w < filter_widthSize; ++w)
							subSum += input[h + offset_height][w + offset_width][offset_channel] * filter[h][w][offset_channel][filterOutChannelId];
						sum += subSum;
					}
					output[pos_height][pos_width][outputChannelId] = sum;
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
					float sum = 0.0;
					/* convolution for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* mul-sum between each channel of a single filter and the related part in input */
						float subSum = 0.0;
						for(int h = 0; h < filter_heightSize; ++h)
						for(int w = 0; w < filter_widthSize; ++w)
							subSum += input[offset_channel][h + offset_height][w + offset_width] * filter[h][w][offset_channel][filterOutChannelId];
						sum += subSum;
					}
					output[outputChannelId][pos_height][pos_width] = sum;
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
					float sum = 0.0;
					/* convolution for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* mul-sum between each channel of a single filter and the related part in input */
						float subSum = 0.0;
						for(int h = 0; h < filter_heightSize; ++h)
						for(int w = 0; w < filter_widthSize; ++w)
						{
							/* position in padded_input */
							int newPos_height = h + offset_height;
							int newPos_width = w + offset_width;
							/* check weather current point is in padded area */
							if(newPos_height >= bound_top && newPos_height <= bound_down && newPos_width >= bound_left && newPos_width <= bound_right)
							subSum += input[newPos_height - pad_top][newPos_width - pad_left][offset_channel] * filter[h][w][offset_channel][filterOutChannelId];
						}
						sum += subSum;
					}
					output[pos_height][pos_width][outputChannelId] = sum;
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
					float sum = 0.0;
					/* convolution for each channel of a single filter */
					for(int pos_channel = 0; pos_channel < input_channelsSize; ++pos_channel)
					{
						int offset_height = pos_height * stride_height;
						int offset_width = pos_width * stride_width;
						int offset_channel = pos_channel * stride_channels;
						/* mul-sum between each channel of a single filter and the related part in input */
						float subSum = 0.0;
						for(int h = 0; h < filter_heightSize; ++h)
						for(int w = 0; w < filter_widthSize; ++w)
						{
							/* position in padded_input */
							int newPos_height = h + offset_height;
							int newPos_width = w + offset_width;
							/* check weather current point is in padded area */
							if(newPos_height >= bound_top && newPos_height <= bound_down && newPos_width >= bound_left && newPos_width <= bound_right)
							{
								subSum += input[offset_channel][newPos_height - pad_top][newPos_width - pad_left] * filter[h][w][offset_channel][filterOutChannelId];
							}
						}
						sum += subSum;
					}
					output[outputChannelId][pos_height][pos_width] = sum;
				}
			}
		}//end of 'NCHW' data_format
	}//end of 'SAME' padding algorithm

	return ;
}


void thread_conv2dBatch(imageBatchType* input, filterBatchType& filter, strideType& strides, string padding, string data_format, int pthreadCount)
{
	conv2d newConv((*input), filter, strides, padding, data_format);
	imageBatchType* output = newConv.run(pthreadCount);
	delete input;
}


#ifdef CONV_TEST

int main(int argc, char *argv[])
{

	string dirPath = argv[1];				//dir path
	string data_format = argv[2];			//data format
	int pthreadCount = atoi(argv[3]);		//pthread number
	int rows = atoi(argv[4]);				//row size
	int cols = atoi(argv[5]);				//col size
	int batch_size = atoi(argv[6]);			//batch size


	//loadImage_batch(dirPath, data_format, rows, cols, input);


	/* filter */
	vector<float> oneDimension(2, 0);
	vector<vector<float>> twoDimensions(3, oneDimension);
	vector<vector<vector<float>>> threeDimensions(3, twoDimensions);
	vector<vector<vector<vector<float>>>> filter(3, threeDimensions);

	float a[] = {1, -1, -1, 1, 0, -1,
				 1, -1, 0, -1, 1, 0,
				 -1, 0, -1, 0, 0, 1,
				 -1, -1, 0, -1, 1, 1,
				 0, 1, 0, 0, 0, 0,
				 1, 0, -1, -1, 1, 1,
				 -1, -1, 1, -1, 0, 0,
				 -1, 1, -1, 0, -1, -1,
				 0, 0, 0, 0, 1, 0};
	vector<float> vec(a, a+54);
	int pos = 0;
	for(int h = 0; h < 3; ++h)
		for(int w = 0; w < 3; ++w)
			for(int i = 0; i < 3; ++i)
				for(int o = 0; o < 2; ++o)
					filter[h][w][i][o] = vec[pos++];


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
			threads.push_back(thread(thread_conv2dBatch, newImageBatch, ref(filter), ref(strides), padding, data_format, pthreadCount));
			counter = 0;
			input.clear();
		}
	}
	if(counter != 0)
	{
		imageBatchType* newImageBatch = new imageBatchType(input);
		threads.push_back(thread(thread_conv2dBatch, newImageBatch, ref(filter), ref(strides), padding, data_format, pthreadCount));
		counter = 0;
		input.clear();
	}
	
	closedir(dir);

	for(auto& th : threads)
		th.join();

	return 0;
}
	
#endif
