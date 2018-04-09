#!/bin/bash

./MatMultOpenMP -n 512  -s > ./data/Serial_512.txt
./MatMultOpenMP -n 1024 -s > ./data/Serial_1024.txt
./MatMultOpenMP -n 2048 -s > ./data/Serial_2048.txt
./MatMultOpenMP -n 4096 -s > ./data/Serial_4096.txt

./MatMultOpenMP -n 512  -t 1 -p > ./data/Parallel_512_1.txt
./MatMultOpenMP -n 1024 -t 1 -p > ./data/Parallel_1024_1.txt
./MatMultOpenMP -n 2048 -t 1 -p > ./data/Parallel_2048_1.txt
./MatMultOpenMP -n 4096 -t 1 -p > ./data/Parallel_4096_1.txt

./MatMultOpenMP -n 512  -t 2 -p > ./data/Parallel_512_2.txt
./MatMultOpenMP -n 1024 -t 2 -p > ./data/Parallel_1024_2.txt
./MatMultOpenMP -n 2048 -t 2 -p > ./data/Parallel_2048_2.txt
./MatMultOpenMP -n 4096 -t 2 -p > ./data/Parallel_4096_2.txt

./MatMultOpenMP -n 512  -t 4 -p > ./data/Parallel_512_4.txt
./MatMultOpenMP -n 1024 -t 4 -p > ./data/Parallel_1024_4.txt
./MatMultOpenMP -n 2048 -t 4 -p > ./data/Parallel_2048_4.txt
./MatMultOpenMP -n 4096 -t 4 -p > ./data/Parallel_4096_4.txt

./MatMultOpenMP -n 512  -t 8 -p > ./data/Parallel_512_8.txt
./MatMultOpenMP -n 1024 -t 8 -p > ./data/Parallel_1024_8.txt
./MatMultOpenMP -n 2048 -t 8 -p > ./data/Parallel_2048_8.txt
./MatMultOpenMP -n 4096 -t 8 -p > ./data/Parallel_4096_8.txt

./MatMultOpenMP -n 512  -t 16 -p > ./data/Parallel_512_16.txt
./MatMultOpenMP -n 1024 -t 16 -p > ./data/Parallel_1024_16.txt
./MatMultOpenMP -n 2048 -t 16 -p > ./data/Parallel_2048_16.txt
./MatMultOpenMP -n 4096 -t 16 -p > ./data/Parallel_4096_16.txt

./MatMultOpenMP -n 512  -t 32 -p > ./data/Parallel_512_32.txt
./MatMultOpenMP -n 1024 -t 32 -p > ./data/Parallel_1024_32.txt
./MatMultOpenMP -n 2048 -t 32 -p > ./data/Parallel_2048_32.txt
./MatMultOpenMP -n 4096 -t 32 -p > ./data/Parallel_4096_32.txt
