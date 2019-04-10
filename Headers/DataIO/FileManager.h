#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <string>
#include <vector>

#include "FileType.cpp"
#include "NeuralNetworkFileManager.h"
#include "DataIOFileManager.h"

class FileManager{

	public:

		FileManager();
		


	private:

		NeuralNetworkFileManager nnfm;
		DataIOFileManager dfm;

		const std::string nn_file_ext = ".nn";
		const std::string trained_nn_file_path = "nnet//trained//";
		const std::string untrained_file_path = "nnet//untrained//";

		const std::string data_location = "data//";
		const std::string data_file_ext = ".csv";
		
		std::string dir;

};

#endif
