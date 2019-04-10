#ifndef VECTOR_STRUCTS_CUH
#define VECTOR_STRUCTS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

struct Dimension {
	unsigned int width = 1;
	unsigned int height = 1;
	unsigned int depth = 1;

	__device__ __host__ Dimension& operator+(const Dimension& other) {
		this->width += other.width;
		this->height += other.height;
		this->depth += other.depth;

		return *this;
	}

	__device__ __host__ Dimension& operator-(const Dimension& other) {
		this->width -= other.width;
		this->height -= other.height;
		this->depth -= other.depth;

		return *this;
	}

	__device__ __host__ Dimension& operator/(const Dimension& other) {
		this->width /= other.width;
		this->height /= other.height;
		this->depth /= other.depth;

		return *this;
	}

	__device__ __host__ Dimension& operator*(const Dimension& other) {
		this->width *= other.width;
		this->height *= other.height;
		this->depth *= other.depth;

		return *this;
	}



};

struct Vector {
	int x;
	int y;
	int z;


	__device__ __host__ Vector& operator+(const Vector& other) {
		this->x += other.x;
		this->y += other.y;
		this->z += other.z;

		return *this;
	}

	__device__ __host__ Vector& operator-(const Vector& other) {
		this->x -= other.x;
		this->y -= other.y;
		this->z -= other.z;

		return *this;
	}

	__device__ __host__ Vector& operator/(const Vector& other) {
		this->x /= other.x;
		this->y /= other.y;
		this->z /= other.z;

		return *this;
	}

	__device__ __host__ Vector& operator*(const Vector& other) {
		this->x *= other.x;
		this->y *= other.y;
		this->z *= other.z;

		return *this;
	}


};

struct LayerLocation {
	Vector location;
	int layerId;
};

struct Neuron {

	LayerLocation location;

	double bias;
	double gradient;
	double prev_gradient = 0.0;

	double input;
	double output;

	bool activated;
	bool learned;
	bool grad_applied;

	int bias_updates = 0;

	int id;

};

struct Connection {
	LayerLocation from;
	LayerLocation to;
};

struct Weight {
	Connection conn;
	
	double weight;
	double prev_weight = 0.0;

	int weight_updates = 0;

	bool learned;
};



#endif
