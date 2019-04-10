#include <cmath>
#include "Linear.cuh"

__device__ __host__ Linear::Linear() {
	Activation();
}

__device__ __host__ float Linear::compute(float input) {
	return input;
}

__device__ __host__ float Linear::derive(float input) {
	return 1.0;
}
