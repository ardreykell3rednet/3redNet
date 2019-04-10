#ifndef CONNECTION_FACTORY_H
#define CONNECTION_FACTORY_H

#include "ConnectionFormat.cpp"
#include "NeuralLayer.cuh"

#define DEFAULT_KERNEL_SIZE 3

namespace ConnectionFactory {

	void connect(NeuralLayer from, NeuralLayer to, ConnectionFormat format = ConnectionFormat::REG, bool reverse_application = false, unsigned int kernel_size = DEFAULT_KERNEL_SIZE);

	void connect_conv(NeuralLayer from, NeuralLayer to, bool reverse_application, unsigned int kernel_size = DEFAULT_KERNEL_SIZE);

	void connect_reg(NeuralLayer from, NeuralLayer to,bool reverse_application);

	void connect_rand(NeuralLayer from, NeuralLayer to, bool reverse_application);

	void connect_pool(NeuralLayer from, NeuralLayer to, bool reverse_application);

};

#endif
