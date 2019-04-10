#include "Activation.cuh"
#include <cmath>

__device__ __host__ Activation::Activation() {

}

__device__ __host__ double Activation::compute(double input, ActivationType at) {
	
	if (at == ActivationType::ELU) {
		//printf("%f\n", input > 0 ? input : 0.01 * (exp(input) - 1));
		return input > 0 ? input : 0.01 * (exp(input) - 1);

	}
	else if (at == ActivationType::HT) {
		return tanh(input);
	}
	else if (at == ActivationType::LIN) {
		return input;
	}
	else if (at == ActivationType::ReLU) {
		return 0 > input ? 0 : input;
	}
	else if (at == ActivationType::SIG) {
		return ((double)exp(input) / (1 + exp(input)));
	}

	
	
	return input;
}

__device__ __host__ double Activation::derive(double input, ActivationType at) {
	
	if (at == ActivationType::ELU) {
		return input > 0 ? 1.0 : compute(input, at) + 0.01;
	}
	else if (at == ActivationType::HT) {
		return 1.0 - (pow(compute(input, at), 2));
	}
	else if (at == ActivationType::LIN) {
		return 1.0;
	}
	else if (at == ActivationType::ReLU) {
		return input > 0 ? 0 : 1;
	}
	else if (at == ActivationType::SIG) {
		return compute(input, at) * (1.0 - compute(input, at));
	}
	
	
	return input;
}
