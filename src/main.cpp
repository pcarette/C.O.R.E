#include <chrono>
#include <iomanip>
#include <iostream>

#include "encoder.hpp"
#include "loader.hpp"

void print_bits(const uint64_t block) {
	for (int i = 0; i < 32; ++i) {
		const uint8_t val = (block >> (i * 2)) & 0x3;
		char c = 'A';
		if (val == 1)
			c = 'C';
		if (val == 2)
			c = 'G';
		if (val == 3)
			c = 'T';
		std::cout << c;
	}
}

int main(const int argc, char **argv) {
	if (argc < 2) {
		std::cerr << "Usage: ./core_runner <file.fasta>" << std::endl;
		return 1;
	}

	const std::string filepath = argv[1];
	std::cout << "[CORE] Starting engine benchmark..." << std::endl;
	std::cout << "[CORE] Target: " << filepath << std::endl;

	try {
		const auto start_load = std::chrono::high_resolution_clock::now();
		const GenomeLoader loader(filepath);
		const auto end_load = std::chrono::high_resolution_clock::now();

		const double size_mb = loader.size() / 1024.0 / 1024.0;
		const double time_load_ms =
			std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load).count() / 1000.0;

		std::cout << "[CORE] IO Load Complete." << std::endl;
		std::cout << "       Size: " << std::fixed << std::setprecision(2) << size_mb << " MB" << std::endl;
		std::cout << "       Time: " << time_load_ms << " ms" << std::endl;
		std::cout << "[CORE] Executing Encoding..." << std::endl;

		const auto start_enc = std::chrono::high_resolution_clock::now();
		const auto encoded_data = loader.encode();
		const auto end_enc = std::chrono::high_resolution_clock::now();
		const double enc_time_ms =
			std::chrono::duration_cast<std::chrono::microseconds>(end_enc - start_enc).count() / 1000.0;
		const double gbps = (loader.size() / 1e9) / (enc_time_ms / 1000.0);

		std::cout << "[CORE] Encoding Complete." << std::endl;
		std::cout << "       Time: " << enc_time_ms << " ms" << std::endl;
		std::cout << "       Throughput: " << gbps << " Gb/s" << std::endl;

		std::cout << "\n[DEBUG] First Block Preview (32 bases):" << std::endl;
		if (!encoded_data.empty()) {
			std::cout << "       Bin: ";
			print_bits(encoded_data[0]);
			std::cout << "\n       Hex: 0x" << std::hex << encoded_data[0] << std::dec << std::endl;
		}
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
