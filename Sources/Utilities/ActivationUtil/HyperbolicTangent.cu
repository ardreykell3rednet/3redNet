#include <cmath>
#include "HyperbolicTangent.cuh"

__device__ __host__ HyperbolicTangent::HyperbolicTangent() {
	Activation();
}

__device__ __host__ float HyperbolicTangent::compute(float input) {
	return tanh(input);
}

__device__ __host__ float HyperbolicTangent::derive(float input) {
	return 1.0 - (pow(compute(input), 2));
}
