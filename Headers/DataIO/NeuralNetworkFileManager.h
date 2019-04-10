#ifndef NEURAL_NETWORK_FILE_MANAGER_H
#define NEURAL_NETWORK_FILE_MANAGER_H

#include "FileType.cpp"
#include <string>
#include <vector>

#include "VectorStructs.cuh"
#include "NeuralNetwork.cuh"



class NeuralNetworkFileManager {

	public:
		NeuralNetworkFileManager();
		NeuralNetworkFileManager(std::string neural_network_directory_path, std::string name);
		
		NeuralNetwork get_parsed();
		void set_parsed(NeuralNetwork set);

		NeuralNetwork get_to_write();
		void set_to_write(NeuralNetwork to_write, std::string neural_network_directory_path);
		void set_to_write(NeuralNetwork to_write);

		void prepare_parse();
		void read_network_file();
		void read_network_dimensions();
		void parse_network();
		
		void prepare_write();
		void write_network_file();
		void write_network_dimensions();
		void write_network();
		
		bool make_dir(std::string path, std::string non_erased);

	private:

		std::string neural_network_directory_path;
		std::string name;
	
		NeuralNetwork parsed;
		NeuralNetwork to_write;

		std::vector<std::string> link_files;

		std::string network_linker = "NETWORK_";
		std::string dims = "NET_DIMS_";
		std::string layer = "LAYER_";
		std::string weight_matrix = "W_MAT_";
		std::string gradient_matrix = "G_MAT_";

		std::string linker_ext = ".nn";
		std::string l_ext = ".layer";
		std::string w_mat_ext = ".weight";
		std::string g_mat_ext = ".gradient";
		std::string dim_ext = ".dim";


		
		

};

#endif


