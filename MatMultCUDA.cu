// get device (GPU) information and specifications

#include <iostream>

int main(void) {
    cudaDeviceProp prop;

    int count;

    cudaGetDeviceCount( &count );

    for(int i=0; i<count; i++)
    {
        std::cout << "---------------------------------------------------------------" << std::endl;

        cudaGetDeviceProperties(&prop, i);
        std::cout << "Name                             " << prop.name       << std::endl;
        std::cout << "GPU clock rate                   " << (double)prop.clockRate / 1024 << " MHz" << std::endl;
        std::cout << "Registers Per Block              " << prop.regsPerBlock  << std::endl;
        std::cout << "Compute capability               " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total global memory              " << (double)prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "Total constant memory            " << (double)prop.totalConstMem / (1024) << " KB" << std::endl;
        std::cout << "Shared memory per block          " << (double)prop.sharedMemPerBlock / (1024) << " KB" << std::endl;
        std::cout << "Maximum threads per block        " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum threads along X          " << prop.maxThreadsDim[0] << std::endl;
        std::cout << "                      Y          " << prop.maxThreadsDim[1] << std::endl;
        std::cout << "                      Z          " << prop.maxThreadsDim[2] << std::endl;
std::cout << "Maximum grid size along X        " << prop.maxGridSize[0] << std::endl;
        std::cout << "                        Y        " << prop.maxGridSize[1] << std::endl;
        std::cout << "                        Z        " << prop.maxGridSize[2] << std::endl;
        std::cout << "Warp size                        " << prop.warpSize << std::endl;
        std::cout << "Multiprocessor count             " << prop.multiProcessorCount << std::endl;
        std::cout << "Device overlap                   " << prop.deviceOverlap << std::endl;
        std::cout << "Maximum resident threads per multi-processor  " << prop.maxThreadsPerMultiProcessor << std::endl;

        std::cout << std::endl;
    }

    return 0;
}

