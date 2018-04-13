#!/bin/bash

./MatMultOpenMP -n 4096 -t 4 -p > ./data/Parallel_4096_4.txt

./MatMultOpenMP -n 4096 -t 8 -p > ./data/Parallel_4096_8.txt

./MatMultOpenMP -n 4096 -t 16 -p > ./data/Parallel_4096_16.txt

./MatMultOpenMP -n 4096 -t 32 -p > ./data/Parallel_4096_32.txt
