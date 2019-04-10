#include "NeuralNetworkFileManager.h"
#include "NeuralNetwork.cuh"
#include "DataTextFileManager.h"

#include <GLFW/glfw3.h>
GLFWwindow* window;

const int num_layers = 4;

std::string u_dir = "nnet/untrained";
std::string t_dir = "nnet/trained";

std::string file_path = "test/";

int main() {

	/*Dimension layer_dims[num_layers] = { {32,32,2}, {32,32,8}, {16,16,8}, {1,1,1} };
	LayerSettings layer_settings[num_layers] = { {true, true, false, ActivationType::SIG, {0,0,0}, layer_dims[1]},
	{true, true, false, ActivationType::ELU, {3,3,1}, {2,2,1}},
	{true, false, false, ActivationType::SIG, {2,2,1}, layer_dims[3]},
	{true, false, false, ActivationType::SIG, layer_dims[2], {0,0,0}} };

	ConnectionFormat formats[num_layers] = { ConnectionFormat::CONV,ConnectionFormat::POOL,ConnectionFormat::REG };

	NeuralNetwork nn_write = NeuralNetwork(num_layers, layer_dims, layer_settings, formats, "TEST");

	nn_write.prepare_network();

	nn_write.malloc();

	double input[10 * 10 * 10];

	for (int i = 0; i < 10 * 10 * 10; i++) {
		input[i] = 0;
	}

	nn_write.nem.input_stack.push_back(&input[0], 2);

	double preferred[1] = { 0 };

	nn_write.execute(&preferred[0]);

	nn_write.apply();

	nn_write.save();

	NeuralNetworkFileManager nnfm = NeuralNetworkFileManager(u_dir, "TEST");

	nnfm.set_to_write(nn_write, u_dir);

	nnfm.prepare_write();
	nnfm.write_network_file();
    nnfm.write_network_dimensions();
	nnfm.write_network();

	nnfm.prepare_parse();
	nnfm.read_network_file();
	nnfm.read_network_dimensions();
	nnfm.parse_network();


	NeuralNetwork parsed = nnfm.get_parsed();

	parsed.malloc();

	for (int i = 0; i < 10 * 10 * 10; i++) {
		input[i] = 0;
	}

	parsed.nem.input_stack.push_back(&input[0], 2);

	parsed.execute(&preferred[0]);*/

	
	DataTextFileManager dtfm = DataTextFileManager(file_path, "TEST");

	for (int i = 0; i < 20; i++) {
		dtfm.push_write(std::to_string(i) + "-TEST");
	}

	dtfm.write_stack();

	dtfm.set_name("TEST-2");

	dtfm.write_stack();

	dtfm.clear_read_stack();

	dtfm.read_all(7, 19);

	std::vector<std::string> read = dtfm.get_read_stack();

	for (int i = 0; i < read.size(); i++) {
		printf("%s\n", read.at(i).c_str());
	}



}