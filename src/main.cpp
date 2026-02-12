#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

static volatile unsigned char sink;

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
		const auto loader_d_size = static_cast<double>(loader.size());
		std::cout << "[CORE] File mapped. Size: " << (loader_d_size / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "[CORE] Warming up RAM (Pre-faulting)..." << std::endl;

		const uint8_t *data = loader.data();
		const size_t size = loader.size();
		constexpr size_t page_size = 4096;
		for (size_t i = 0; i < size; i += page_size)
			sink = data[i];

		std::cout << "[CORE] Allocating Pinned Memory (DMA Ready)..." << std::endl;
		const size_t num_blocks = (size + 31) / 32;
		PinnedHostBuffer<uint64_t> pinned_genome(num_blocks);
		std::cout << "[CORE] Executing AVX2 Encoding (Direct to Pinned)..." << std::endl;
		const auto start_enc = std::chrono::high_resolution_clock::now();
		encode_sequence_avx2(data, size, pinned_genome.data());
		const auto end_enc = std::chrono::high_resolution_clock::now();
		const double enc_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_enc - start_enc).count() / 1000.0;
		const double throughput_gbps = (loader_d_size * 8.0 / 1e9) / (enc_time_ms / 1000.0);

		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "  Encoding Time : " << enc_time_ms << " ms" << std::endl;
		std::cout << "  Throughput    : " << throughput_gbps << " Gb/s" << std::endl;
		std::cout << "------------------------------------------------" << std::endl;
		if (pinned_genome.size() > 0) {
			const std::string signature = "ACGTACGTACGTACGTACGTAGGACGTACGT";
			const uint64_t unique_pattern = make_pattern(signature);
			pinned_genome[0] = unique_pattern;
			std::cout << "\n[CORE] --- VALIDATION: PINNED MEMORY SEARCH ---" << std::endl;
			std::cout << "Target (Index 0)     : ";
			print_bits(unique_pattern);
			std::cout << std::endl;
			const uint64_t bulged_pattern = unique_pattern << 2;
			SearchResults res = launch_bulge_search(pinned_genome.data(), pinned_genome.size(), bulged_pattern, 2, 0);
			std::cout << "  Matches Found : " << res.count << std::endl;
			std::cout << "  GPU Time      : " << res.time_ms << " ms" << std::endl;
			if (res.count > 0 && res.matches[0] == 0) {
				std::cout << "  [SUCCESS] TARGET LOCKED via DMA + PAM Validated." << std::endl;
			} else {
				std::cout << "  [FAIL] Target Missed (Check PAM logic)." << std::endl;
			}
			free_search_results(res);
		}
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
