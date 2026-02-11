#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "kernel.hpp"
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

uint64_t make_pattern(const std::string &seq) {
	uint64_t pat = 0;
	for (size_t i = 0; i < seq.length() && i < 32; ++i) {
		uint64_t val = 0;
		if (const char c = seq[i]; c == 'C')
			val = 1;
		else if (c == 'G')
			val = 2;
		else if (c == 'T')
			val = 3;
		pat |= (val << (i * 2));
	}
	return pat;
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
		const double warm_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_warm - start_warm).count() / 1000.0;
		std::cout << "[CORE] Warmup Complete. Disk/OS Latency: " << warm_ms << " ms" << std::endl;

		std::cout << "[CORE] Executing AVX2 Encoding (RAM Resident)..." << std::endl;
		const auto start_enc = std::chrono::high_resolution_clock::now();
		auto encoded_data = encode_sequence_avx2(data, size);
		const auto end_enc = std::chrono::high_resolution_clock::now();
		const double enc_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_enc - start_enc).count() / 1000.0;
		const double size_gb = loader.size() * 8.0 / 1e9;
		const double throughput_gbps = size_gb / (enc_time_ms / 1000.0);

		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "  Encoding Time : " << enc_time_ms << " ms" << std::endl;
		std::cout << "  CPU Throughput: " << throughput_gbps << " Gb/s" << std::endl;
		std::cout << "------------------------------------------------" << std::endl;

		if (!encoded_data.empty()) {
			const std::string signature = "ACGTACGTACGTACGTACGTACGTACGTACGT";
			const uint64_t unique_pattern = make_pattern(signature);
			encoded_data[0] = unique_pattern;
			std::cout << "\n[CORE] --- SECRET WEAPON TEST: BULGE/INDEL ---" << std::endl;
			std::cout << "Target (Index 0)     : ";
			print_bits(unique_pattern);
			std::cout << std::endl;
			const uint64_t bulged_pattern = unique_pattern << 2;
			std::cout << "Query (Shifted << 1) : ";
			print_bits(bulged_pattern);
			std::cout << std::endl;
			std::cout << "\n[TEST A] Exact Search on Shifted Pattern..." << std::endl;
			SearchResults res1 = launch_exact_search(encoded_data.data(), encoded_data.size(), bulged_pattern);
			std::cout << " -> Matches: " << res1.count << " (Expected 0)" << std::endl;
			free_search_results(res1);
			std::cout << "\n[TEST B] Bulge Search (Auto-Shift Detection)..." << std::endl;
			SearchResults res2 = launch_bulge_search(encoded_data.data(), encoded_data.size(), bulged_pattern, 2);
			std::cout << "------------------------------------------------" << std::endl;
			std::cout << "  Matches Found : " << res2.count << std::endl;
			std::cout << "  GPU Time      : " << res2.time_ms << " ms" << std::endl;
			if (res2.count > 0) {
				std::cout << "  First Index   : " << res2.matches[0] << " (Target: 0)" << std::endl;
				if (res2.matches[0] == 0) {
					std::cout << "  [SUCCESS] TARGET LOCKED. Engine Validated." << std::endl;
				} else {
					std::cout << "  [FAIL] Target Missed." << std::endl;
				}
			} else {
				std::cout << "  [FAIL] No Match Found." << std::endl;
			}
			std::cout << "------------------------------------------------" << std::endl;
			free_search_results(res2);
		}
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
