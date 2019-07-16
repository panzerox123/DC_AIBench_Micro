#!/bin/bash

##
# The input data set is under ImageData/image_1000
# You can use the other data as input
##

echo "start parsec workload"

WORK_DIR=`pwd`
logdir=$WORK_DIR/result/Pthread-AI-Motif
mkdir -p $logdir
host=`hostname`
batchsize=100

if [ $2 = "large" ];then
    imgsize=100000
elif [ $2 = "medium" ];then
    imgsize=10000
else
    imgsize=1000
fi

sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

start=`date +%s`

if [ $1 = "conv" ];then
echo "running motif conv"
./Pthread/conv2d ImageData/image_$imgsize/img$imgsize/ NCHW 12 227 227 $batchsize 

elif [ $1 = "maxpool" ];then
echo "running motif maxpool"
./Pthread/max_pool ImageData/image_$imgsize/img$imgsize/ NCHW 12 227 227 $batchsize

elif [ $1 = "avgpool" ];then
echo "running motif avgpool"
./Pthread/avg_pool ImageData/image_$imgsize/img$imgsize/ NCHW 12 227 227 $batchsize

elif [ $1 = "relu" ];then
echo "running motif relu"
./Pthread/relu2 ImageData/image_$imgsize/img$imgsize/ 12 227 227 $batchsize

elif [ $1 = "matmul" ];then
echo "running motif matmul"
./Pthread/matmul ImageData/image_$imgsize/img$imgsize/ 12 227 227 $batchsize

elif [ $1 = "multiply" ];then
echo "running motif multiply"
./Pthread/multiply ImageData/image_$imgsize/img$imgsize/ 12 227 227 $batchsize

elif [ $1 = "sigmoid" ];then
echo "running motif sigmoid"
./Pthread/sigmoid ImageData/image_$imgsize/img$imgsize/ 12 227 227 $batchsize

elif [ $1 = "tanh" ];then
echo "running motif tanh"
./Pthread/tanh ImageData/image_$imgsize/img$imgsize/ 12 227 227 $batchsize

else
   echo "wrong $1 parameter"
fi

end=`date +%s`

echo "$1 end"
