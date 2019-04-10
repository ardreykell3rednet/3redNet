#ifndef MEAN_SQUARED_ERROR_CUH
#define MEAN_SQUARED_ERROR_CUH

#include "Error.cuh"

class MeanSquaredError : public Error {
	public:
		__host__ MeanSquaredError();

		__device__ __host__ float compute(float output, float preferredValue);
		__device__ __host__ float derive(float output, float preferredValue);
};

#endif