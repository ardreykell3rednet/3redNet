#include <cmath>
#include "ExponentialLinearUnits.cuh"

__device__ __host__ ExponentialLinearUnits::ExponentialLinearUnits() {
	Activation();
}

__device__ __host__ float ExponentialLinearUnits::compute(float input) {
	return input > 0 ? input : (exp(input) - 1);
}

__device__ __host__ float ExponentialLinearUnits::derive(float input) {
	return input > 0 ? 1.0 : compute(input) + 1;
}
