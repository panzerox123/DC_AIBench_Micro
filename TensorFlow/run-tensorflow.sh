#!/bin/bash

echo "start AI TensorFlow micro benchmark"

WORK_DIR=`pwd`
host=`hostname`
batchsize=128
filter=3

if [ $2 = "large" ];then
    imgsize=224
    channel=64
elif [ $2 = "medium" ];then
    imgsize=112
    channel=128
else
    imgsize=56
    channel=256
fi
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

start=`date +%s`

if [ $1 = "conv" ];then
echo "running dwarf conv"
python conv2d.py $batchsize $imgsize $channel $filter

elif [ $1 = "maxpool" ];then
echo "running dwarf maxpool"
python max_pool.py $batchsize $imgsize $channel

elif [ $1 = "avgpool" ];then
echo "running dwarf avgpool"
python avg_pool.py $batchsize $imgsize $channel

elif [ $1 = "relu" ];then
echo "running dwarf relu"
python relu.py $batchsize $imgsize $channel

elif [ $1 = "batchNorm" ];then
echo "running dwarf batchNorm"
python batch_normalization.py $batchsize $imgsize

elif [ $1 = "matmul" ];then
echo "running dwarf matmul"
python matmul.py $batchsize $imgsize $channel

elif [ $1 = "multiply" ];then
echo "running dwarf multiply"
python multiply.py $batchsize $imgsize $channel

elif [ $1 = "sigmoid" ];then
echo "running dwarf sigmoid"
python sigmoid.py $batchsize $imgsize $channel 

elif [ $1 = "tanh" ];then
echo "running dwarf tanh"
python tanh.py $batchsize $imgsize $channel

elif [ $1 = "dropout" ];then
echo "running dwarf dropout"
python dropout.py $batchsize $imgsize $channel

else
   echo "wrong $1 parameter"
fi

end=`date +%s`


echo "$1 end, kill monitor script"
