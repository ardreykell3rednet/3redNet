#include "NeuralGradientMatrix.cuh"


NeuralGradientMatrix::NeuralGradientMatrix() {
	to = { {0,0,0}, -1 };
}

NeuralGradientMatrix::NeuralGradientMatrix(Dimension d, LayerLocation to_relay)
{
	this->grad_dim = d;
	this->to = to_relay;

	gradient_matrix = new double[d.width * d.height * d.depth];
	beta_matrix = new double[d.width * d.height * d.depth];

	for (int i = 0; i < grad_dim.width * grad_dim.height * grad_dim.depth; i++) {
		gradient_matrix[i] = 0;
	}

	num_accesses = 0;
}

NeuralGradientMatrix::NeuralGradientMatrix(unsigned int width, unsigned int height, unsigned int depth, LayerLocation to_relay)
{
	Dimension d = { width, height, depth };

	this->grad_dim = d;
	this->to = to_relay;

	gradient_matrix = new double[d.width * d.height * d.depth];

	for (int i = 0; i < grad_dim.width * grad_dim.height * grad_dim.depth; i++) {
		gradient_matrix[i] = 0;
	}

	num_accesses = 0;

}

__device__ __host__ double * NeuralGradientMatrix::get_gradient_matrix()
{
	return gradient_matrix;
}

__device__ __host__ double NeuralGradientMatrix::get_gradient_value(Vector loc)
{
	return !oob_error(loc) ? 0.0 : gradient_matrix[loc.x + loc.y * grad_dim.width + loc.z * grad_dim.width * grad_dim.height];
}

__device__ __host__ double NeuralGradientMatrix::get_gradient_value(int x, int y, int z)
{
	Vector v = { x,y,z };
	return get_gradient_value(v);
}

__device__ __host__ void NeuralGradientMatrix::set_gradient_value(Vector loc, double value)
{
	num_accesses++;

	if (!oob_error(loc)) {
		gradient_matrix[loc.x + loc.y * grad_dim.width + loc.z * grad_dim.width * grad_dim.height] = value;
	}
}

__device__ __host__ void NeuralGradientMatrix::set_next_gradient_value(double value) {
	
	if (num_accesses < grad_dim.width * grad_dim.height * grad_dim.depth) {
		gradient_matrix[num_accesses++] = value;
	}
}

__device__ __host__ void NeuralGradientMatrix::set_gradient_value(int x, int y, int z, double value)
{
	Vector v = { x,y,z };
	set_gradient_value(v, value);
}

__device__ __host__ double * NeuralGradientMatrix::get_beta_matrix()
{
	return gradient_matrix;
}

__device__ __host__ double NeuralGradientMatrix::get_beta_value(Vector loc)
{
	return !oob_error(loc) ? 0.0 : beta_matrix[loc.x + loc.y * grad_dim.width + loc.z * grad_dim.width * grad_dim.height];
}

__device__ __host__ double NeuralGradientMatrix::get_beta_value(int x, int y, int z)
{
	Vector v = { x,y,z };
	return get_beta_value(v);
}

__device__ __host__ void NeuralGradientMatrix::set_beta_value(Vector loc, double value)
{
	num_accesses++;

	if (!oob_error(loc)) {
		beta_matrix[loc.x + loc.y * grad_dim.width + loc.z * grad_dim.width * grad_dim.height] = value;
	}
}

__device__ __host__ void NeuralGradientMatrix::set_next_beta_value(double value) {

	if (num_accesses < grad_dim.width * grad_dim.height * grad_dim.depth) {
		beta_matrix[num_accesses++] = value;
	}
}

__device__ __host__ void NeuralGradientMatrix::set_beta_value(int x, int y, int z, double value)
{
	Vector v = { x,y,z };
	set_beta_value(v, value);
}


__device__ __host__ LayerLocation NeuralGradientMatrix::get_application_location()
{
	return to;
}

__device__ __host__ Dimension NeuralGradientMatrix::get_dimension()
{
	return grad_dim;
}

__device__ __host__ void NeuralGradientMatrix::set_dimension(Dimension d) {
	this->grad_dim = d;
}

__device__ __host__ bool NeuralGradientMatrix::is_full()
{
	return num_accesses == grad_dim.width * grad_dim.height * grad_dim.depth;
}

__device__ __host__ void NeuralGradientMatrix::reset() {

	num_accesses = 0;
	for (int i = 0; i < grad_dim.width * grad_dim.height * grad_dim.depth; i++) {
		gradient_matrix[i] = 0;
		beta_matrix[i] = 0;
	}
}

__device__ __host__ bool NeuralGradientMatrix::oob_error(Vector loc)
{
	if (loc.x >= grad_dim.width || loc.y >= grad_dim.height || loc.z >= grad_dim.depth) {
		return true;
	}
	else if (loc.x < 0 || loc.y < 0 || loc.z < 0) {
		return true;
	}

	return false;
}
