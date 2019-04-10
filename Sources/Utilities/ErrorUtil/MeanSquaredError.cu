#include "MeanSquaredError.cuh"

__host__ MeanSquaredError::MeanSquaredError() {
	Error();
}

__device__ __host__ float MeanSquaredError::compute(float output, float preferredValue) {
	return 0.5 * (preferredValue - output) * (preferredValue - output);
}

__device__ __host__ float MeanSquaredError::derive(float output, float preferredValue) {
	return output - preferredValue;
}