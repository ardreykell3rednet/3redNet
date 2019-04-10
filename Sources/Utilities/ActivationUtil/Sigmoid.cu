#include <cmath>
#include "Sigmoid.cuh"

__device__ __host__ Sigmoid::Sigmoid() {
	Activation();
}

__device__ __host__ float Sigmoid::compute(float input) {
	return ((float)exp(input)/(1 + exp(input)));
}

__device__ __host__ float Sigmoid::derive(float input) {
	return compute(input) * (1.0 - compute(input));
}
