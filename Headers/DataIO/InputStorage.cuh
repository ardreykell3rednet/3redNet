#ifndef INPUT_STORAGE_CUH
#define INPUT_STORAGE_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

class InputStorage {
	public:
		__device__ __host__ InputStorage();

		__device__ __host__ void clear();
		__device__ __host__ void push_back(double* next, int size);

		__device__ __host__ double* get_next();

		__device__ __host__ void reset_location();
	
	private:

		std::vector<double*> output_stack;

		int location = 0;

};

#endif