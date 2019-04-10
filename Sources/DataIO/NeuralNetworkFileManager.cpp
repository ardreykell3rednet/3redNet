#include "NeuralNetworkFileManager.h"

#include <fstream>
#include <iostream>
#include <filesystem>

NeuralNetworkFileManager::NeuralNetworkFileManager() {

}

NeuralNetworkFileManager::NeuralNetworkFileManager(std::string neural_network_directory_path, std::string name){
	this->neural_network_directory_path = neural_network_directory_path;
	this->name = name;
}

NeuralNetwork NeuralNetworkFileManager::get_parsed() {
	return parsed;
}

void NeuralNetworkFileManager::set_parsed(NeuralNetwork set){
	parsed = set;
}

NeuralNetwork NeuralNetworkFileManager::get_to_write(){
	return to_write;
}

void NeuralNetworkFileManager::set_to_write(NeuralNetwork to_write, std::string neural_network_directory_path){
	this->to_write = to_write;
	this->neural_network_directory_path = neural_network_directory_path;
}

void NeuralNetworkFileManager::set_to_write(NeuralNetwork to_write){
	this->to_write = to_write;
}

void NeuralNetworkFileManager::prepare_parse(){
	parsed = NeuralNetwork();
}

void NeuralNetworkFileManager::read_network_file(){
	std::string network_file_path = neural_network_directory_path + "/" + name + "/NETWORK_FILES/" + network_linker + name + linker_ext;

	std::ifstream file;
	file.open(network_file_path);

	printf("Reading Network: %s", network_file_path.c_str());

	while (file.good()) {
		std::string read = "";
		getline(file, read);

		link_files.push_back(read);
		printf(".");
	}

	printf("complete\n");

	file.close();
}

void NeuralNetworkFileManager::read_network_dimensions() {
	std::string dim_file_path = neural_network_directory_path + "/" + name + "/NETWORK_FILES/" + dims + name + dim_ext;
	
	std::ifstream file;
	file.open(dim_file_path);

	printf("Reading Network %s Dimensions", name.c_str());

	if (file.good()) {
		
		std::string read = "";

		getline(file, read);

		int layers = std::stoi(read.substr(0, read.find(',')));

		Dimension* dims = new Dimension[layers];

		printf("(%i layers): ", layers);

		for (int i = 0; i < layers; i++) {

			if (file.good()) {
				getline(file, read);

				unsigned int width = std::stoi(read.substr(0, read.find(',')));
				read = read.erase(0, read.find(',') + 1);

				unsigned int height = std::stoi(read.substr(0, read.find(',')));
				read = read.erase(0, read.find(',') + 1);

				unsigned int depth = std::stoi(read.substr(0, read.find(',')));
				read = read.erase(0, read.find(',') + 1);


				dims[i] = Dimension({ width, height, depth });

				printf("{%i, %i, %i} ", width, height, depth);

			}
		}

		parsed.set_network_dim(dims);
		parsed.layers = layers;

		printf("complete\n");

	}

	file.close();

}

void NeuralNetworkFileManager::parse_network(){
	
	int layer = 0;

	NeuralLayer* layers = new NeuralLayer[parsed.layers];

	for(int i = 0; i < link_files.size(); i++){
		std::string path = link_files.at(i);

		std::ifstream file;
		file.open(path);

		printf("Opening %s\n", path.c_str());


		if (path.find(l_ext) != std::string::npos) {
			std::string read = "";

			getline(file, read);
			layer = std::stoi(read);

			Dimension layer_dim = parsed.get_dim(layer);

			LayerSettings ls = {};
			getline(file, read);

			bool bias = read.substr(0, read.find(',')).find("true") != std::string::npos ? true : false;
			read = read.erase(0, read.find(',') + 1);
			bool conv = read.substr(0, read.find(',')).find("true") != std::string::npos ? true : false;
			read = read.erase(0, read.find(',') + 1);
			bool pool = read.substr(0, read.find(',')).find("true") != std::string::npos ? true : false;
			read = read.erase(0, read.find(',') + 1);

			std::string token = read.substr(0, read.find(','));

			ActivationType at;

			if (token.find("ELU")) {
				at = ActivationType::ELU;
			}
			else if (token.find("HT")) {
				at = ActivationType::HT;
			}
			else if (token.find("LIN")) {
				at = ActivationType::LIN;
			}
			else if (token.find("ReLU")) {
				at = ActivationType::ReLU;
			}
			else if (token.find("SIG")) {
				at = ActivationType::SIG;
			}
			else {
				at = ActivationType::SIG;
			}

			read = read.erase(0, read.find(',') + 1);

			Dimension weight_dim = {};

			unsigned int width = 0, height = 0, depth = 0;

			std::string width_string = read.substr(0, read.find(','));
			read = read.erase(0, read.find(',') + 1);
			std::string height_string = read.substr(0, read.find(','));
			read = read.erase(0, read.find(',') + 1);
			std::string depth_string = read.substr(0, read.find(';'));
			read = read.erase(0, read.find(';') + 1);

			//printf("%sDMA\n%sDMA\n%sDAM\n", width_string.c_str(), height_string.c_str(), depth_string.c_str());

			std::string zero = "0";

			if (strcmp(width_string.c_str(), zero.c_str()) == 0){
				width = 0;  
			}
			else {
				width = std::stoi(width_string);
			}

			if (strcmp(height_string.c_str(), zero.c_str()) == 0) {
				height = 0;
			}
			else {
				height = std::stoi(height_string);
			}

			if (strcmp(depth_string.c_str(), zero.c_str()) == 0) {
				depth = 0;
			}
			else {
				depth = std::stoi(depth_string);
			}

			weight_dim.width = width;
			weight_dim.height = height;
			weight_dim.depth = depth;

			Dimension grad_dim = {};

			width_string = read.substr(0, read.find(','));
			read = read.erase(0, read.find(',') + 1);
			height_string = read.substr(0, read.find(','));
			read = read.erase(0, read.find(',') + 1);
			depth_string = read.substr(0, read.find(';'));
			read = read.erase(0, read.find(';') + 1);

			if (strcmp(width_string.c_str(), "0") == 0) {
				width = 0;
			}
			else {
				width = std::stoi(width_string);
			}

			if (strcmp(height_string.c_str(), "0") == 0) {
				height = 0;
			}
			else {
				height = std::stoi(width_string);
			}

			if (strcmp(depth_string.c_str(), "0") == 0) {
				depth = 0;
			}
			else {
				depth = std::stoi(width_string);
			}

			
			grad_dim.width = width;
			grad_dim.height = height;
			grad_dim.depth = depth;

			ls = { bias, conv, pool, at, weight_dim, grad_dim };

			layers[layer] = NeuralLayer(layer_dim, layer, ls);
			
			/*+++++++++++++++++++++++++++++++++++++++++++++++*/
			getline(file, read);

			double* bias_deltas = new double[layer_dim.width * layer_dim.depth * layer_dim.height];

			for (int i = 0; i < layer_dim.width * layer_dim.height * layer_dim.depth; i++) {
				double to_add = std::stod(read.substr(0, read.find(',')));
				bias_deltas[i] = to_add;
				read = read.erase(0, read.find(',') + 1);

			}

			layers[layer].bias_deltas = bias_deltas;

			/*++++++++++++++++++++++++++++++++++++++++++++++*/

			Neuron* neurons = new Neuron[layer_dim.width * layer_dim.depth * layer_dim.height];

			for (int i = 0; i < layer_dim.width * layer_dim.height * layer_dim.depth; i++) {
				getline(file, read);

				LayerLocation loc = {};

				token = read.substr(0, read.find(';'));

				int n_layer = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int x = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int y = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int z = std::stoi(token.substr(0, token.find(';')));
				token = token.erase(0, token.find(';') + 1);
				

				read.erase(0, read.find(';') + 1);

				loc.layerId = n_layer;
				loc.location = { x,y,z };

				neurons[i].location = loc;

				double bias = std::stod(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);
				
				double gradient = std::stod(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);

				double prev_gradient = std::stod(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);

				double input = std::stod(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);

				double output = std::stod(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);

				int bias_updates = std::stoi(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);

				int id = std::stoi(read.substr(0, read.find(',')));
				read.erase(0, read.find(',') + 1);


				neurons[i].bias = bias;
				neurons[i].gradient = gradient;
				neurons[i].prev_gradient = prev_gradient;
				neurons[i].input = input;
				neurons[i].output = output;	
				neurons[i].activated = false;
				neurons[i].learned = false;
				neurons[i].grad_applied = false;
				neurons[i].bias_updates = bias_updates;
				neurons[i].id = id;

			}
			
			layers[layer].neurons = neurons;
			layers[layer].weight_in_ptrs = new NeuralWeightMatrix[layer_dim.width * layer_dim.depth * layer_dim.height];
			layers[layer].gradient_ptrs = new NeuralGradientMatrix[layer_dim.width * layer_dim.depth * layer_dim.height];
		}
		else if (path.find(w_mat_ext) != std::string::npos) {
			std::string read = "";
			while (file.good()) {
				getline(file, read);

				LayerLocation l_loc = {};

				std::string token = read.substr(0, read.find(';'));

				//intf("%s\n", token.c_str());

				if (strcmp(token.c_str(), "") == 0) {
					break;
				}

				int n_layer = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int x = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int y = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int z = std::stoi(token.substr(0, token.find(';')));
				token = token.erase(0, token.find(';') + 1);

				read.erase(0, read.find(';') + 1);

				l_loc = { Vector({x,y,z}), n_layer };

				int lin_loc = l_loc.location.x + l_loc.location.y * layers[n_layer].get_dim().width + l_loc.location.z * layers[n_layer].get_dim().width * layers[n_layer].get_dim().height;

				layers[n_layer].weight_in_ptrs[lin_loc].to = l_loc;

				/*++++++++++++++++++++++++++++++++++++++++++++++*/
				getline(file, read);

				Dimension weight_dim = layers[n_layer].get_obj_in_dim();

				double* weight_deltas = new double[weight_dim.width * weight_dim.depth * weight_dim.height];

				for (int i = 0; i < weight_dim.width * weight_dim.height * weight_dim.depth; i++) {
					double to_add = std::stod(read.substr(0, read.find(',')));
					weight_deltas[i] = to_add;
					read = read.erase(0, read.find(',') + 1);

				}

				layers[n_layer].weight_in_ptrs[lin_loc].weight_deltas = weight_deltas;

				/*++++++++++++++++++++++++++++++++++++++++++++++++*/

				getline(file, read);

				double* output_matrix = new double[weight_dim.width * weight_dim.depth * weight_dim.height];

				for (int i = 0; i < weight_dim.width * weight_dim.height * weight_dim.depth; i++) {
					double to_add = std::stod(read.substr(0, read.find(',')));
					output_matrix[i] = to_add;
					read = read.erase(0, read.find(',') + 1);

				}

				layers[n_layer].weight_in_ptrs[lin_loc].output_matrix = output_matrix;

				/*++++++++++++++++++++++++++++++++++++++++++++++++*/

				Weight* neurons = new Weight[weight_dim.width * weight_dim.depth * weight_dim.height];

				for (int i = 0; i < weight_dim.width * weight_dim.height * weight_dim.depth; i++) {
					getline(file, read);

					LayerLocation loc1 = {};
					LayerLocation loc2 = {};

					token = read.substr(0, read.find(';'));

					int n_layer_1 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int x_1 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int y_1 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int z_1 = std::stoi(token.substr(0, token.find(';')));
					token = token.erase(0, token.find(';') + 1);

					read.erase(0, read.find(';') + 1);

					loc1 = { Vector({x_1, y_1, z_1 }), n_layer_1 };

					token = read.substr(0, read.find(';'));

					int n_layer_2 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int x_2 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int y_2 = std::stoi(token.substr(0, token.find(',')));
					token = token.erase(0, token.find(',') + 1);

					int z_2 = std::stoi(token.substr(0, token.find(';')));
					token = token.erase(0, token.find(';') + 1);

					read.erase(0, read.find(';') + 1);

					loc2 = { Vector({x_2, y_2, z_2 }), n_layer_2 };

					Connection conn = { loc1, loc2 };

					neurons[i].conn = conn;

					double weight = std::stod(read.substr(0, read.find(',')));
					read.erase(0, read.find(',') + 1);

					double prev_weight = std::stod(read.substr(0, read.find(',')));
					read.erase(0, read.find(',') + 1);

					int weight_updates = std::stoi(read.substr(0, read.find(',')));
					read.erase(0, read.find(',') + 1);

					neurons[i].weight = weight;
					neurons[i].prev_weight = prev_weight;
					neurons[i].weight_updates = weight_updates;
					neurons[i].learned = false;
				}

				layers[n_layer].weight_in_ptrs[lin_loc].weights = neurons;
				layers[n_layer].weight_in_ptrs[lin_loc].set_dimension(layers[n_layer].get_obj_in_dim());

				getline(file, read);

				while (!read.find("-")) { getline(file, read); }

			}


		}
		else if (path.find(g_mat_ext) != std::string::npos) {
			std::string read = "";
			
			while (!file.eof()) {
				getline(file, read);

				LayerLocation l_loc = {};

				std::string token = read.substr(0, read.find(';'));

				if (strcmp(token.c_str(), "") == 0) {
					break;
				}

				int n_layer = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int x = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int y = std::stoi(token.substr(0, token.find(',')));
				token = token.erase(0, token.find(',') + 1);

				int z = std::stoi(token.substr(0, token.find(';')));
				token = token.erase(0, token.find(';') + 1);

				read.erase(0, read.find(';') + 1);

				l_loc = { Vector({x,y,z}), n_layer };

				int lin_loc = l_loc.location.x + l_loc.location.y * layers[n_layer].get_dim().width + l_loc.location.z * layers[n_layer].get_dim().width * layers[n_layer].get_dim().height;

				layers[n_layer].gradient_ptrs[lin_loc].to = l_loc;

				/*++++++++++++++++++++++++++++++++++++++++++++++*/
				getline(file, read);

				Dimension grad_dim = layers[n_layer].get_obj_grad_dim();

				double* gradient_matrix = new double[grad_dim.width * grad_dim.depth * grad_dim.height];

				for (int i = 0; i < grad_dim.width * grad_dim.height * grad_dim.depth; i++) {

					try {
						double to_add = std::stod(read.substr(0, read.find(',')));
						gradient_matrix[i] = to_add;
						read = read.erase(0, read.find(',') + 1);
					}
					catch (std::invalid_argument &ia) {
						gradient_matrix[i] = 0;
					}


				}

				layers[n_layer].gradient_ptrs[lin_loc].gradient_matrix = gradient_matrix;

				/*++++++++++++++++++++++++++++++++++++++++++++++++*/
			
				getline(file, read);

				double* beta_matrix = new double[grad_dim.width * grad_dim.depth * grad_dim.height];

				for (int i = 0; i < grad_dim.width * grad_dim.height * grad_dim.depth; i++) {
					try {
						double to_add = std::stod(read.substr(0, read.find(',')));
						beta_matrix[i] = to_add;
						read = read.erase(0, read.find(',') + 1);
					}
					catch (std::invalid_argument &ia) {
						beta_matrix[i] = 0;
					}

				}

				layers[n_layer].gradient_ptrs[lin_loc].beta_matrix = beta_matrix;

				getline(file, read);

				layers[n_layer].gradient_ptrs[lin_loc].set_dimension(layers[n_layer].get_obj_grad_dim());

				while (!read.find(",")) { getline(file, read); }

			}
		}

		file.close();
	}

	parsed.set_network(layers);

	LayerSettings* net_sets = new LayerSettings[parsed.layers];
	
	for (int i = 0; i < parsed.layers; i++) {
		net_sets[i] = layers[i].ls;
	}

	parsed.set_network_layer_settings(net_sets);

}

void NeuralNetworkFileManager::prepare_write() {
	
	std::string path = neural_network_directory_path + "/" + name + "/NETWORK_FILES/";

	printf("Initializing Dir: %s   ", path.c_str());

	if (make_dir(path.c_str(), "") != 0) { printf("Success\n"); }
	else { printf("Failed\n"); }

	path = neural_network_directory_path + "/" + name + "/LAYER_FILES/";

	printf("Initializing Dir: %s   ", path.c_str());

	if (make_dir(path.c_str(), "") != 0) { printf("Success\n"); }
	else { printf("Failed\n"); }

	path = neural_network_directory_path + "/" + name + "/W_MAT_FILES/";

	printf("Initializing Dir: %s   ", path.c_str());

	if (make_dir(path.c_str(), "") != 0) { printf("Success\n"); }
	else { printf("Failed\n"); }

	path = neural_network_directory_path + "/" + name + "/G_MAT_FILES/";

	printf("Initializing Dir: %s   ", path.c_str());

	if (make_dir(path.c_str(), "") != 0) { printf("Success\n"); }
	else { printf("Failed\n"); }
}

void NeuralNetworkFileManager::write_network_file() {
	
	std::string path = neural_network_directory_path + "/" + name + "/NETWORK_FILES/" + network_linker + name + linker_ext;
	std::ofstream file;
	
	file.open(path, std::ios::beg);

	printf("Writing Network File.");

	for (int i = 0; i < to_write.layers; i++) {

		file << neural_network_directory_path + "/" + name + "/LAYER_FILES/LAYER_" + std::to_string(i) + l_ext << std::endl;
		file << neural_network_directory_path + "/" + name + "/W_MAT_FILES/W_MAT_" + std::to_string(i) + w_mat_ext << std::endl;
		file << neural_network_directory_path + "/" + name + "/G_MAT_FILES/G_MAT_" + std::to_string(i) + g_mat_ext << std::endl;

		printf(".");

	}

	printf("complete\n");

	file.close();
}

void NeuralNetworkFileManager::write_network_dimensions(){
	std::string path = neural_network_directory_path + "/" + name + "/NETWORK_FILES/" + dims + name + dim_ext;
	std::ofstream file;

	file.open(path, std::ios::beg);

	printf("Writing Dimension File.");

	file << std::to_string(to_write.layers) + "," << std::endl;

	for (int i = 0; i < to_write.layers; i++) {
		file << std::to_string(to_write.get_dim(i).width) + "," + std::to_string(to_write.get_dim(i).height) + "," +
			std::to_string(to_write.get_dim(i).depth) + "," << std::endl;

		printf(".");

	}

	printf("complete\n");

	file.close();
}

void NeuralNetworkFileManager::write_network(){

	printf("Writing Network\n");

	for (int i = 0; i < to_write.layers; i++) {

		printf(":> Layer %i.", i);

		std::string path = neural_network_directory_path + "/" + name + "/LAYER_FILES/LAYER_" + std::to_string(i) + l_ext;

		std::ofstream file;
		file.open(path);

		LayerSettings w = to_write.get_layer_settings(i);

		file << std::to_string(i) + "," << std::endl;
	
		std::string b = (w.includeBias ? "true," : "false,");
		std::string c = (w.convolutional ? "true," : "false,");
		std::string p = (w.pool ? "true," : "false,");

		file << b + c + p;
		std::string at;

		if (w.activation == ActivationType::ELU) {
			at = "ELU,";
		} 
		else if (w.activation == ActivationType::HT) {
			at = "HT,";
		}
		else if (w.activation == ActivationType::ReLU) {
			at = "ReLU,";
		}
		else if (w.activation == ActivationType::LIN) {
			at = "LIN,";
		}
		else if (w.activation == ActivationType::SIG) {
			at = "SIG,";
		}
		else {
			at = "SIG,";
		}

		file << at;

		Dimension weight = w.weight_in_dim;

		file << std::to_string(weight.width) + "," + std::to_string(weight.height) + "," + std::to_string(weight.depth) + ";";

		weight = w.grad_dim;

		file << std::to_string(weight.width) + "," + std::to_string(weight.height) + "," + std::to_string(weight.depth) + ";" << std::endl;

		for (int j = 0; j < to_write.get_dim(i).width * to_write.get_dim(i).height * to_write.get_dim(i).depth; j++) {
			file << std::to_string(to_write.get_layer(i).bias_deltas[j]) + ",";
		}

		file << std::endl;

		for (int j = 0; j < to_write.get_dim(i).width * to_write.get_dim(i).height * to_write.get_dim(i).depth; j++) {
			Neuron write = to_write.get_layer(i).neurons[j];

			file << std::to_string(write.location.layerId) + "," + std::to_string(write.location.location.x) + "," + std::to_string(write.location.location.y) + "," + std::to_string(write.location.location.z) + ";" +
				std::to_string(write.bias) + "," + std::to_string(write.gradient) + "," + std::to_string(write.prev_gradient) + "," + std::to_string(write.input) + "," + std::to_string(write.output) + "," +
				std::to_string(write.bias_updates) + "," + std::to_string(write.id) + "," << std::endl;

		}

		file.close();

		printf(".");

		path = neural_network_directory_path + "/" + name + "/W_MAT_FILES/W_MAT_" + std::to_string(i) + w_mat_ext;

		file.open(path);

		for (int j = 0; j < to_write.get_dim(i).width * to_write.get_dim(i).height * to_write.get_dim(i).depth; j++) {

			NeuralWeightMatrix write = to_write.get_layer(i).weight_in_ptrs[j];

			file << std::to_string(write.to.layerId) + "," + std::to_string(write.to.location.x) + "," + std::to_string(write.to.location.y) + "," + std::to_string(write.to.location.z) + ";" << std::endl;
			
			for (int k = 0; k < write.get_dimension().width * write.get_dimension().height * write.get_dimension().depth; k++) {
				file << std::to_string(write.weight_deltas[k]) + ",";
			}

			file << std::endl;

			for (int k = 0; k < write.get_dimension().width * write.get_dimension().height * write.get_dimension().depth; k++) {
				file << std::to_string(write.output_matrix[k]) + ",";
			}

			file << std::endl;

			for (int k = 0; k < write.get_dimension().width * write.get_dimension().height * write.get_dimension().depth; k++) {
				Weight wr = write.weights[k];
				
				file << std::to_string(wr.conn.from.layerId) + "," + std::to_string(wr.conn.from.location.x) + "," + std::to_string(wr.conn.from.location.y) + "," + std::to_string(wr.conn.from.location.z) + ";" +
					std::to_string(wr.conn.to.layerId) + "," + std::to_string(wr.conn.to.location.x) + "," + std::to_string(wr.conn.to.location.y) + "," + std::to_string(wr.conn.to.location.z) + ";" +
					std::to_string(wr.weight) + "," + std::to_string(wr.prev_weight) + "," + std::to_string(wr.weight_updates) + "," << std::endl;
			}

			if (j % 10 == 0) {
				printf(".");
			}

			file << ":---------------------------------------------------------------: WEIGHT-END: " + std::to_string(j) << std::endl;

		}

		file.close();

		path = neural_network_directory_path + "/" + name + "/G_MAT_FILES/G_MAT_" + std::to_string(i) + g_mat_ext;

		file.open(path);

		for (int j = 0; j < to_write.get_dim(i).width * to_write.get_dim(i).height * to_write.get_dim(i).depth; j++) {


			NeuralGradientMatrix ngm = to_write.get_layer(i).gradient_ptrs[j];
		
			file << std::to_string(ngm.to.layerId) + "," + std::to_string(ngm.to.location.x) + "," + std::to_string(ngm.to.location.y) + "," + std::to_string(ngm.to.location.z) + ";" << std::endl;

			for (int k = 0; k < ngm.get_dimension().width * ngm.get_dimension().height * ngm.get_dimension().depth; k++) {
				file << std::to_string(ngm.gradient_matrix[k]) + ",";
			}

			file << std::endl;

			for (int k = 0; k < ngm.get_dimension().width * ngm.get_dimension().height * ngm.get_dimension().depth; k++) {
				file << std::to_string(ngm.beta_matrix[k]) + ",";
			}

			file << std::endl;

			if (j % 10 == 0) {
				printf(".");
			}

			file << ":---------------------------------------------------------------: GRAD-END: " + std::to_string(j) << std::endl;
		}

		file.close();

		printf("complete\n");

	}
}

bool NeuralNetworkFileManager::make_dir(std::string path, std::string non_erased) {

	if (path.find('/') == std::string::npos) {
		return true;
	}
	else {
		std::string dir = non_erased + path.substr(0, path.find('/'));

		//printf("Creating directory %s\n", dir.c_str());

		std::experimental::filesystem::create_directory(dir.c_str());

		non_erased += path.substr(0, path.find('/')) + "/";

		std::string token = path.erase(0, path.find('/') + 1);
		make_dir(token, non_erased);
	}

	
}