#include "Error.cuh"

__host__ Error::Error() {

}

__device__ __host__ double Error::compute(double output, double preferredValue, ErrorType et) {
	
	if (et == ErrorType::MSE) {
		return 0.5 * (preferredValue - output) * (preferredValue - output);
	}
	
	return preferredValue - output;
}

__device__ __host__ double Error::derive(double output, double preferredValue, ErrorType et) {
	
	if (et == ErrorType::MSE) {
		return output - preferredValue;
	}
	
	return 1.0;
}