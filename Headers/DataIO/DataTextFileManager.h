#ifndef DATA_TEXT_FILE_MANAGER_H
#define DATA_TEXT_FILE_MANAGER_H

#include "FileType.cpp"
#include <string>
#include <vector>

#include "VectorStructs.cuh"
#include "NeuralNetwork.cuh"

#include <direct.h>

#include <iostream>
#include <fstream>
#include <filesystem>

class DataTextFileManager {

public:

	DataTextFileManager();
	DataTextFileManager(std::string file_path, std::string name);

	std::string get_file_path();
	void set_file_path(std::string file_path);

	std::string get_name();
	void set_name(std::string file_path);

	void read_stack();

	void read(int line);
	void read_all(int s_line, int end_line);

	void read_until(std::string delimiter);

	void write(int s_line);
	void write_stack();
	void write_all(int s_line, int end_line);

	void clear_read_stack();
	void clear_write_stack();

	std::string get_next_read_index();
	std::vector<std::string> get_read_stack();

	std::string get_next_write_index();
	std::vector<std::string> get_write_stack();
	void push_write(std::string to_write);

	bool make_dir(std::string path, std::string non_erased);

private:

	std::string file_path;
	std::vector<std::string> data_read;
	std::vector<std::string> data_write;

	std::string data_file_format = ".csv";

	std::string name;

	std::vector<int> char_lengths;

	std::streampos curr_pos;

	int index = 0;
};

#endif