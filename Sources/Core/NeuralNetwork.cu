#include "NeuralNetwork.cuh"

NeuralNetwork::NeuralNetwork() {
	nem = NetworkExecutionKernel();
}

NeuralNetwork::NeuralNetwork(int layers, Dimension* layer_dims, LayerSettings* layer_settings, ConnectionFormat* connections, std::string name) {
	network = new NeuralLayer[layers];

	this->layer_settings = layer_settings;
	this->connections = connections;
	this->layer_dimensions = layer_dims;

	this->layers = layers;

	nem = NetworkExecutionKernel();

	this->name = name;
}

__device__ __host__ NeuralLayer NeuralNetwork::get_layer(int layer_index)
{
	return !oob_error(layer_index) ? network[layer_index] : NeuralLayer();
}

__device__ __host__ void NeuralNetwork::set_layer(int layer_index, NeuralLayer to_set)
{
	if (!oob_error(layer_index)) {
		network[layer_index] = to_set;
		update_required = true;
	}
}

__device__ __host__ void NeuralNetwork::set_network(NeuralLayer * network)
{
	this->network = network;
}

__device__ __host__ ConnectionFormat NeuralNetwork::get_connection_format(int layer_index)
{
	return !oob_error(layer_index) ? connections[layer_index] : ConnectionFormat::REG;
}

__device__ __host__ void NeuralNetwork::set_connection_format(int layer_index, ConnectionFormat cf)
{
	if (!oob_error(layer_index)) {
		connections[layer_index] = cf;
		update_required = true;
	}
}

__device__ __host__ void NeuralNetwork::set_network_connection_properties(ConnectionFormat * cf)
{
	this->connections = cf;
}

__device__ __host__ LayerSettings NeuralNetwork::get_layer_settings(int layer_index)
{
	return !oob_error(layer_index) ? layer_settings[layer_index] : LayerSettings({});
}

__device__ __host__ void NeuralNetwork::set_layer_settings(int layer_index, LayerSettings ls)
{
	if (!oob_error(layer_index)) {
		layer_settings[layer_index] = ls;
		update_required = true;
	}
}

__device__ __host__ void NeuralNetwork::set_network_layer_settings(LayerSettings * ls)
{
	layer_settings = ls;
}

__device__ __host__ Dimension NeuralNetwork::get_dim(int layer_index)
{
	return !oob_error(layer_index) ? layer_dimensions[layer_index] : Dimension({ 0,0,0 });
}

__device__ __host__ void NeuralNetwork::set_dim(int layer_index, Dimension size)
{
	if (!oob_error(layer_index)) {
		layer_dimensions[layer_index] = size;
	}
}

__device__ __host__ void NeuralNetwork::set_network_dim(Dimension * sizes)
{
	layer_dimensions = sizes;
}

__device__  __host__ void NeuralNetwork::set_stream(cudaStream_t &stream)
{
	this->stream = stream;
	nem.stream = stream;
}

__device__ __host__ void NeuralNetwork::set_execution_device(int device)
{
	this->execution_device = device;
	nem.execution_device = device;
}

__device__ __host__ bool NeuralNetwork::oob_error(int layer_index)
{
	return layer_index > layers;
}

__device__ __host__ void NeuralNetwork::push_input(double * input)
{
	Dimension in_size = layer_dimensions[0];
	nem.push(input, in_size.width * in_size.height * in_size.depth);
}

__device__ __host__ void NeuralNetwork::execute(double* preferred)
{
	output = nem.network_exec(network, ErrorType::MSE, layers, preferred);
}
__device__ __host__ void NeuralNetwork::execute()
{
	output = nem.network_exec(network, ErrorType::MSE, layers);
}

__device__ __host__ void NeuralNetwork::apply()
{
	nem.network_apply(layers);
}

__device__ __host__ void NeuralNetwork::prepare_network()
{
	for (int i = 0; i < layers; i++) {

		//printf("%s: Initializing Layer %i: Dimension {%i, %i, %i}\n", name.c_str(), i, layer_dimensions[i].width, layer_dimensions[i].height, layer_dimensions[i].depth);

		network[i] = NeuralLayer(layer_dimensions[i], i, layer_settings[i]);
		printf("%s: Initializing Layer %i: Dimension {%i, %i, %i}\n", name.c_str(), i, network[i].get_dim().width, network[i].get_dim().height, network[i].get_dim().depth);
	}

	connect();

	update_required = false;
}

__device__ __host__ void NeuralNetwork::connect()
{
	for (int i = 0; i < layers - 1; i++) {
		ConnectionFactory::connect(network[i], network[i + 1], connections[i]);
	}

	update_required = false;
}

__device__ __host__ void NeuralNetwork::malloc() {

	std::clock_t start = std::clock();
	printf("Beginning Network Malloc %s...", name.c_str());

	nem.malloc(network, layers);

	double malloc_time = (double)(std::clock() - start) / CLOCKS_PER_SEC;
	printf("complete : %f seconds\n", malloc_time);

}

__device__ __host__ void NeuralNetwork::save()
{
	network = nem.halloc(network, layers);
}

