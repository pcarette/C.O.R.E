#include <bitset>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

static volatile uint8_t sink;

void print_bits(const uint64_t block) {
	for (int i = 0; i < 32; ++i) {
		const uint8_t val = (block >> (i * 2)) & 0x3;
		char c = '?';
		if (val == 0)
			c = 'A';
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
		if (const char c = std::toupper(seq[i]); c == 'C')
			val = 1;
		else if (c == 'G')
			val = 2;
		else if (c == 'T')
			val = 3;
		pat |= (val << (i * 2));
	}
	return pat;
}

std::string get_reverse_complement(const std::string &seq) {
	std::string rc;
	for (int i = seq.length() - 1; i >= 0; i--) {
		if (const char c = std::toupper(seq[i]); c == 'A')
			rc += 'T';
		else if (c == 'T')
			rc += 'A';
		else if (c == 'C')
			rc += 'G';
		else if (c == 'G')
			rc += 'C';
		else
			rc += 'N';
	}
	return rc;
}

void run_sanity_check() {
	std::cout << "\n[DIAGNOSTICS] --- AVX2 MEMORY DUMP START ---" << std::endl;
	std::string seq = "ACGTACGTACGTACGTACGTACGTACGTACGT";
	const std::vector<uint8_t> raw(seq.begin(), seq.end());

	PinnedHostBuffer<uint64_t> out(1);
	encode_sequence_avx2(raw.data(), raw.size(), out.data());

	uint64_t val = out[0];
	std::cout << "Input    : " << seq.substr(0, 4) << "..." << std::endl;
	std::cout << "Hex Raw  : 0x" << std::hex << val << std::dec << std::endl;
	std::cout << "Bin Raw  : " << std::bitset<64>(val) << " (MSB ... LSB)" << std::endl;

	std::cout << "Decoding LSB (Bytes Order Check):" << std::endl;
	for (int i = 0; i < 4; i++) {
		const uint8_t bits = (val >> (i * 2)) & 3;
		const char c = seq[i];
		std::cout << "  Base[" << i << "] '" << c << "' -> Bits: " << std::bitset<2>(bits) << " (" << static_cast<int>(bits)
				  << ")" << std::endl;
		bool ok = false;
		if (c == 'A' && bits == 0)
			ok = true;
		if (c == 'C' && bits == 1)
			ok = true;
		if (c == 'G' && bits == 2)
			ok = true;
		if (c == 'T' && bits == 3)
			ok = true;
		if (!ok)
			std::cout << "    [CRITICAL ERROR] Encoding Mismatch! CPU maps '" << c << "' to " << static_cast<int>(bits)
					  << std::endl;
		else
			std::cout << "    [OK]" << std::endl;
	}

	std::string weird = "acgtNNNN";
	while (weird.length() < 32)
		weird += "A";

	const std::vector<uint8_t> raw_weird(weird.begin(), weird.end());
	encode_sequence_avx2(raw_weird.data(), raw_weird.size(), out.data());
	val = out[0];

	std::cout << "\nInput    : " << weird.substr(0, 8) << "..." << std::endl;
	std::cout << "Hex Raw  : 0x" << std::hex << val << std::dec << std::endl;
	std::cout << "Decoding :" << std::endl;
	for (int i = 0; i < 8; i++) {
		const uint8_t bits = (val >> (i * 2)) & 3;
		const char c = weird[i];
		std::cout << "  Base[" << i << "] '" << c << "' -> " << static_cast<int>(bits) << std::endl;
	}

	std::cout << "[DIAGNOSTICS] --- AVX2 MEMORY DUMP END ---\n" << std::endl;
}

std::vector<uint8_t> sanitize_genome(const uint8_t *raw_data, const size_t raw_size, double &duration_ms) {
	const auto start = std::chrono::high_resolution_clock::now();
	std::vector<uint8_t> clean_data;
	clean_data.reserve(raw_size);

	bool in_header = false;
	for (size_t i = 0; i < raw_size; ++i) {
		uint8_t c = raw_data[i];
		if (in_header) {
			if (c == '\n')
				in_header = false;
			continue;
		}
		if (c == '>') {
			in_header = true;
			continue;
		}
		if (c == '\n' || c == '\r' || c == ' ' || c == '\t')
			continue;
		clean_data.push_back(c);
	}

	const auto end = std::chrono::high_resolution_clock::now();
	duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
	return clean_data;
}

int main(const int argc, char **argv) {
	run_sanity_check();
	if (argc < 3) {
		std::cerr << "Usage: ./core_runner <file.fasta> <TARGET_SEQUENCE>" << std::endl;
		return 1;
	}

	const std::string filepath = argv[1];
	const std::string target_seq = argv[2];

	if (target_seq.length() >= 23) {
		const char pam1 = std::toupper(target_seq[21]);
		if (const char pam2 = std::toupper(target_seq[22]); pam1 != 'G' || pam2 != 'G') {
			std::cout << "[WARN] Target does not end with 'GG' (PAM). Kernel filter will likely reject it." << std::endl;
		}
	}

	std::cout << "[CORE] Starting benchmark..." << std::endl;
	std::cout << "[CORE] Target File : " << filepath << std::endl;
	std::cout << "[CORE] Query Seq   : " << target_seq << " (" << target_seq.length() << " bp)" << std::endl;
	std::cout << "================================================================" << std::endl;

	try {
		const auto t_global_start = std::chrono::high_resolution_clock::now();

		std::cout << "[STEP 1] Memory Mapping..." << std::endl;
		const auto t_load_start = std::chrono::high_resolution_clock::now();
		const GenomeLoader loader(filepath);
		const auto t_load_end = std::chrono::high_resolution_clock::now();
		const double load_time =
			std::chrono::duration_cast<std::chrono::microseconds>(t_load_end - t_load_start).count() / 1000.0;

		std::cout << "  > Mapped Size   : " << std::fixed << std::setprecision(2) << (loader.size() / 1024.0 / 1024.0) << " MB"
				  << std::endl;
		std::cout << "  > IO Time       : " << load_time << " ms" << std::endl;

		const uint8_t *raw_data = loader.data();
		const size_t raw_size = loader.size();
		for (size_t i = 0; i < raw_size; i += 4096)
			sink = raw_data[i];

		std::cout << "[STEP 2] Sanitizing FASTA..." << std::endl;
		double sanitize_time = 0;
		const std::vector<uint8_t> clean_data = sanitize_genome(raw_data, raw_size, sanitize_time);
		const size_t clean_size = clean_data.size();

		const double ratio = static_cast<double>(clean_size) / raw_size * 100.0;
		std::cout << "  > Raw Size      : " << (raw_size / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "  > Clean Size    : " << (clean_size / 1024.0 / 1024.0) << " MB (" << ratio << "% genetic content)"
				  << std::endl;
		std::cout << "  > Sanitize Time : " << sanitize_time << " ms" << std::endl;

		std::cout << "[STEP 3] Allocating Pinned Memory (DMA)..." << std::endl;
		const auto t_alloc_start = std::chrono::high_resolution_clock::now();
		const size_t num_blocks = (clean_size + 31) / 32;
		PinnedHostBuffer<uint64_t> pinned_genome(num_blocks);
		const auto t_alloc_end = std::chrono::high_resolution_clock::now();
		const double alloc_time =
			std::chrono::duration_cast<std::chrono::microseconds>(t_alloc_end - t_alloc_start).count() / 1000.0;

		std::cout << "  > Buffer Size   : " << (pinned_genome.size() * 8.0 / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "  > Alloc Time    : " << alloc_time << " ms" << std::endl;

		std::cout << "[STEP 4] Executing AVX2 Encoding..." << std::endl;
		const auto t_enc_start = std::chrono::high_resolution_clock::now();
		encode_sequence_avx2(clean_data.data(), clean_size, pinned_genome.data());
		const auto t_enc_end = std::chrono::high_resolution_clock::now();

		const double enc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_enc_end - t_enc_start).count() / 1000.0;
		const double throughput = (clean_size * 8.0 / 1e9) / (enc_time / 1000.0);

		std::cout << "  > Encoding Time : " << enc_time << " ms" << std::endl;
		std::cout << "  > Throughput    : " << throughput << " Gb/s" << std::endl;

		if (pinned_genome.size() > 0) {
			const std::vector targets = {target_seq};
			for (size_t t = 0; t < targets.size(); ++t) {
				std::cout << "\n[STEP 5] Searching..." << std::endl;
				const uint64_t pattern = make_pattern(targets[t]);
				SearchResults res = launch_bulge_search(pinned_genome.data(), pinned_genome.size(), pattern, 3, 0);
				std::cout << "  > Matches Found : " << res.count << std::endl;
				if (res.count > 0) {
					const uint32_t limit = (res.count < 5) ? res.count : 5;
					for (uint32_t i = 0; i < limit; ++i) {
						const uint32_t idx = res.matches[i];
						std::cout << "    Hit Block " << idx << ": ";
						print_bits(pinned_genome[idx]);
						std::cout << std::endl;
					}
				}
				free_search_results(res);
			}
		}

		const auto t_global_end = std::chrono::high_resolution_clock::now();
		std::cout << "[CORE] Done in "
				  << std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global_start).count() << " ms"
				  << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
