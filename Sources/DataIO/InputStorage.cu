#include "InputStorage.cuh"

__device__ __host__ InputStorage::InputStorage() {
	output_stack.reserve(30);
}

__device__ __host__ void InputStorage::clear()
{
	for (int i = 0; i < output_stack.size(); i++) {
	
		if (cudaFree(output_stack.at(i)) != cudaSuccess) {
			printf("O-Stack Clear Err\n");
		}
	}

	output_stack.clear();
}

__device__ __host__ void InputStorage::push_back(double * next, int size)
{
	double* d_next;

	if (cudaMalloc(&d_next, size * sizeof(double)) != cudaSuccess) {
		printf("Input Arr Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_next, next, size * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Input Arr Cpy: %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	output_stack.push_back(d_next);
}

__device__ __host__ double * InputStorage::get_next()
{
	if (location >= output_stack.size() - 1) { reset_location(); }

	double* ret = output_stack.at(location++);

	return ret;
}

__device__ __host__ void InputStorage::reset_location() {
	location = 0;
}


