#include "InputParser.cuh"
#include "InputGenerator.cuh"
#include "InputCreator.cuh"

#include "OutputWriter.cuh"

#include "NeuralNetwork.cuh"
#include "NeuralNetworkFileManager.h"

#include <iostream>

const int num_layers = 4;

int main() {

	NeuralNetwork nn_write = NeuralNetwork();
	NeuralNetworkFileManager manager = NeuralNetworkFileManager();
}
