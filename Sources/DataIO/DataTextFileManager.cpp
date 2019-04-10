#include "DataTextFileManager.h"


DataTextFileManager::DataTextFileManager() {

}

DataTextFileManager::DataTextFileManager(std::string file_path, std::string name) {
	this->file_path = file_path;
	this->name = name;

	make_dir(file_path, "");
}

std::string DataTextFileManager::get_file_path()
{
	return file_path;
}

void DataTextFileManager::set_file_path(std::string file_path) {
	this->file_path = file_path;
	make_dir(file_path, "");

}

std::string DataTextFileManager::get_name()
{
	return name;
}

void DataTextFileManager::set_name(std::string name) {
	this->name = name;
	//make_dir(file_path, "");

}

void DataTextFileManager::read_stack() {
	std::ifstream reader;
	reader.open(file_path + name + data_file_format, std::ios::beg);

	std::string push = "";

	while (reader.good()) {
		getline(reader, push);
		data_read.push_back(push);
	}

	reader.close();
}

void DataTextFileManager::read(int line) {

	std::ifstream reader;
	reader.open(file_path + name + data_file_format, std::ios::beg);

	std::string push = "";

	for (int i = 0; i < line - 1; i++) {

		if (i < char_lengths.size()) {
			reader.ignore(char_lengths.at(i));
		}
		else {
			getline(reader, push);
			char_lengths.push_back(push.size());
		}
	}

	getline(reader, push);
	//printf("%s\n", push.c_str());
	data_read.push_back(push);
	reader.close();

}



void DataTextFileManager::read_all(int s_line, int end_line) {
	std::ifstream reader;
	reader.open(file_path + name + data_file_format, std::ios::beg);

	std::string push = "";

	for (int i = 0; i < s_line - 1; i++) {

		if (i < char_lengths.size()) {
			reader.ignore(char_lengths.at(i));
		}
		else {
			getline(reader, push);
			char_lengths.push_back(push.size());
		}

	}

	for (int i = s_line; i < end_line; i++) {
		getline(reader, push);
		data_read.push_back(push);
	}

	reader.close();

}

void DataTextFileManager::read_until(std::string delimiter){

	std::ifstream g_if;
	
	g_if.open(file_path + name + data_file_format, std::ios::beg);
	g_if.seekg(curr_pos,  std::ios::beg);

	int i = 0;

	while (true) {
		std::string push = "";

		getline(g_if, push);

		if (push.find(delimiter) != std::string::npos && i != 0) {
			break;
		}

		printf("%s\n", push.substr(0, 50));

		data_read.push_back(push);

		i++;

	}
	
	curr_pos = g_if.tellg();

	g_if.close();

}

void DataTextFileManager::write(int s_line) {

	std::ifstream reader;
	reader.open(file_path + name + data_file_format, std::ios::beg);

	std::string push = "";

	int chars = 0;

	for (int i = 0; i < s_line - 1; i++) {
		chars += char_lengths.at(i);
	}

	reader.close();

	std::ofstream writer;
	writer.open(file_path + name + data_file_format);
	writer.seekp(chars + 1);

	writer << data_write.at(0) << std::endl;

	writer.close();
}

void DataTextFileManager::write_stack() {

	std::ofstream writer;
	writer.open(file_path + name + data_file_format, std::ios::app);

	for (int i = 0; i < data_write.size(); i++) {
		writer << data_write.at(i) << std::endl;
	}

	writer.close();

}

void DataTextFileManager::write_all(int s_line, int end_line) {

	std::ifstream reader;
	reader.open(file_path + name + data_file_format, std::ios::beg);

	std::string push = "";

	int chars = 0;

	for (int i = 0; i < s_line - 1; i++) {
		getline(reader, push);

		chars += push.size();

	}

	reader.close();

	std::ofstream writer;
	writer.open(file_path + name + data_file_format);
	writer.seekp(chars + 1);

	for (int i = s_line; i < end_line; i++) {
		if (i - s_line < data_write.size()) {
			writer << data_write.at(i - s_line) << std::endl;
		}
		else {
			writer << std::endl;
		}
	}

	writer.close();

}

void DataTextFileManager::clear_read_stack() {
	data_read.clear();
}

void DataTextFileManager::clear_write_stack() {
	data_write.clear();
}

std::string DataTextFileManager::get_next_read_index()
{
	return data_read.at(0);
}

std::vector<std::string> DataTextFileManager::get_read_stack()
{
	return data_read;
}

std::string DataTextFileManager::get_next_write_index()
{
	return data_write.at(0);
}

std::vector<std::string> DataTextFileManager::get_write_stack()
{
	return data_write;
}

void DataTextFileManager::push_write(std::string to_write) {
	data_write.push_back(to_write);
}

bool DataTextFileManager::make_dir(std::string path, std::string non_erased) {

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
