#!/usr/bin/env bash

for i in {20..1}
do
    echo $i
    ./test_resnet.sh $i > test_$i.log
done