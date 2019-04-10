#include <cmath>
#include "RectifiedLinearUnits.cuh"

__device__ __host__ RectifiedLinearUnits::RectifiedLinearUnits() {
	Activation();
}

__device__ __host__ float RectifiedLinearUnits::compute(float input) {
	return 0 > input ? 0 : input;
}

__device__ __host__ float RectifiedLinearUnits::derive(float input) {
	return input > 0 ? 0 : 1;
}