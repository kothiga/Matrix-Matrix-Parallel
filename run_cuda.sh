#!/bin/bash

./MatMultCUDA -n 512  > ./data/CUDA_512.txt
./MatMultCUDA -n 1024 > ./data/CUDA_1024.txt
./MatMultCUDA -n 2048 > ./data/CUDA_2048.txt
./MatMultCUDA -n 4096 > ./data/CUDA_4096.txt
./MatMultCUDA -n 8192 > ./data/CUDA_8192.txt
./MatMultCUDA -n 16384 > ./data/CUDA_16384.txt
