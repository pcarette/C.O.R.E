#include <bitset>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <immintrin.h>

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
	for (size_t i = 0; i < seq.size() && i < 32; ++i) {
		uint8_t code = 0;
		if (const char c = seq[i]; c == 'C' || c == 'c')
			code = 1;
		else if (c == 'G' || c == 'g')
			code = 2;
		else if (c == 'T' || c == 't')
			code = 3;
		else
			code = 0;
		pat |= static_cast<uint64_t>(code) << (i * 2);
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

std::vector<uint8_t> sanitize_genome(const uint8_t *raw_data, const size_t raw_size, double &duration_ms) {
	const auto start = std::chrono::high_resolution_clock::now();

	std::vector<uint8_t> clean_data;
	clean_data.resize(raw_size);

	uint8_t *dst = clean_data.data();
	const uint8_t *src = raw_data;
	const uint8_t *end = raw_data + raw_size;
	const __m256i thresh = _mm256_set1_epi8(32);
	const __m256i gt_char = _mm256_set1_epi8('>');

	while (src < end) {
		if (*src == '>') {
			if (const void *newline_pos = std::memchr(src, '\n', end - src))
				src = static_cast<const uint8_t *>(newline_pos) + 1;
			else
				break;
			continue;
		}

		while (src + 32 <= end) {
			if (*src == '>')
				break;

			const __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
			const __m256i is_valid = _mm256_cmpgt_epi8(chunk, thresh);
			const int mask = _mm256_movemask_epi8(is_valid);
			const __m256i is_header = _mm256_cmpeq_epi8(chunk, gt_char);
			const int header_mask = _mm256_movemask_epi8(is_header);
			if (header_mask != 0)
				break;

			if (mask == -1) {
				_mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), chunk);
				dst += 32;
				src += 32;
			} else {
				const uint8_t *limit = src + 32;
				while (src < limit) {
					if (*src == '>')
						break;
					if (*src > 32) {
						*dst++ = *src;
					}
					src++;
				}
			}
		}

		while (src < end && *src != '>') {
			if (*src > 32) {
				*dst++ = *src;
			}
			src++;
		}
	}

	const size_t actual_size = dst - clean_data.data();
	clean_data.resize(actual_size);

	const auto end_time = std::chrono::high_resolution_clock::now();
	duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count() / 1000.0;
	return clean_data;
}

int main(const int argc, char **argv) {
	if (argc < 4) {
		std::cerr << "Usage: ./core_runner <genome.fasta> <epigenome.epi> <TARGET_SEQUENCE>" << std::endl;
		return 1;
	}

	const std::string filepath = argv[1];
	const std::string epigenome_path = argv[2];
	const std::string target_seq = argv[3];

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

		std::cout << "[STEP 2] Loading Epigenome Atlas..." << std::endl;
		const BinaryLoader epi_loader(epigenome_path);
		std::cout << "  > Epigenome Size: " << (epi_loader.size() / 1024.0 / 1024.0) << "MB" << std::endl;

		std::cout << "[STEP 3] Sanitizing FASTA..." << std::endl;
		double sanitize_time = 0;
		const std::vector<uint8_t> clean_data = sanitize_genome(raw_data, raw_size, sanitize_time);
		const size_t clean_size = clean_data.size();

		const double ratio = static_cast<double>(clean_size) / raw_size * 100.0;
		std::cout << "  > Raw Size      : " << (raw_size / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "  > Clean Size    : " << (clean_size / 1024.0 / 1024.0) << " MB (" << ratio << "% genetic content)"
				  << std::endl;
		std::cout << "  > Sanitize Time : " << sanitize_time << " ms" << std::endl;

		std::cout << "[STEP 4] Allocating Pinned Memory (DMA)..." << std::endl;
		const auto t_alloc_start = std::chrono::high_resolution_clock::now();
		const size_t num_blocks = (clean_size + 31) / 32;
		PinnedHostBuffer<uint64_t> pinned_genome(num_blocks);
		const auto t_alloc_end = std::chrono::high_resolution_clock::now();
		const double alloc_time =
			std::chrono::duration_cast<std::chrono::microseconds>(t_alloc_end - t_alloc_start).count() / 1000.0;

		std::cout << "  > Buffer Size   : " << (pinned_genome.size() * 8.0 / 1024.0 / 1024.0) << " MB" << std::endl;
		std::cout << "  > Alloc Time    : " << alloc_time << " ms" << std::endl;

		std::cout << "[STEP 5] Executing AVX2 Encoding..." << std::endl;
		const auto t_enc_start = std::chrono::high_resolution_clock::now();
		encode_sequence_avx2(clean_data.data(), clean_size, pinned_genome.data());
		const auto t_enc_end = std::chrono::high_resolution_clock::now();

		const double enc_time = std::chrono::duration_cast<std::chrono::microseconds>(t_enc_end - t_enc_start).count() / 1000.0;
		const double throughput = (clean_size * 8.0 / 1e9) / (enc_time / 1000.0);

		std::cout << "  > Encoding Time : " << enc_time << " ms" << std::endl;
		std::cout << "  > Throughput    : " << throughput << " Gb/s" << std::endl;

		if (pinned_genome.size() > 0) {
			for (const std::vector targets = {target_seq}; const auto &target : targets) {
				std::cout << "\n[STEP 5] Searching..." << std::endl;
				const uint64_t pattern = make_pattern(target);
				SearchResults res = launch_bulge_search(
					pinned_genome.data(), epi_loader.data(), pinned_genome.size(), epi_loader.size(), pattern, 3, 0
				);
				std::cout << "  > Matches Found : " << res.count << std::endl;
				if (res.count > 0) {
					const uint32_t limit = (res.count < 15) ? res.count : 15;
					for (uint32_t i = 0; i < limit; ++i) {
						const uint32_t idx = res.matches[i] / 32;
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
