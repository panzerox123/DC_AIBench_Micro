//loadImage.cpp

#include "loadImage.h"


int main()
{
	string dirPath = "/home/zdy/imagenet/images/image_100/img100/";
	string format = "NHWC";
	batchImage batchImg;
	loadImage_batch(dirPath, format, batchImg);
	return 0;
}
