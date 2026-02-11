#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "encoder.hpp"
#include "loader.hpp"

static volatile char sink;

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
		const GenomeLoader loader(filepath);
		std::cout << "[CORE] File mapped. Size: " << (loader.size() / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "[CORE] Warming up RAM (Pre-faulting)..." << std::endl;
		const auto start_warm = std::chrono::high_resolution_clock::now();

		const uint8_t *data = loader.data();
		const size_t size = loader.size();
		constexpr size_t page_size = 4096;
		for (size_t i = 0; i < size; i += page_size)
			sink = data[i];

		const auto end_warm = std::chrono::high_resolution_clock::now();
		const double warm_ms =
			std::chrono::duration_cast<std::chrono::microseconds>(end_warm - start_warm).count() / 1000.0;
		std::cout << "[CORE] Warmup Complete. Disk/OS Latency: " << warm_ms << " ms" << std::endl;

		std::cout << "[CORE] Executing AVX2 Encoding (RAM Resident)..." << std::endl;
		const auto start_enc = std::chrono::high_resolution_clock::now();
		const auto encoded_data = encode_sequence_avx2(reinterpret_cast<const uint8_t *>(data), size);
		const auto end_enc = std::chrono::high_resolution_clock::now();
		const double enc_time_ms =
			std::chrono::duration_cast<std::chrono::microseconds>(end_enc - start_enc).count() / 1000.0;
		const double size_gb = loader.size() * 8.0 / 1e9;
		const double throughput_gbps = size_gb / (enc_time_ms / 1000.0);

		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "  Encoding Time : " << enc_time_ms << " ms" << std::endl;
		std::cout << "  CPU Throughput: " << throughput_gbps << " Gb/s" << std::endl;
		std::cout << "------------------------------------------------" << std::endl;

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
