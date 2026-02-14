#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

std::string decode_block_32bp(const uint64_t block) {
	std::string s;
	s.reserve(32);
	for (int i = 0; i < 32; ++i) {
		const uint8_t val = (block >> (i * 2)) & 0x3;
		char c = 'A';
		if (val == 1)
			c = 'C';
		else if (val == 2)
			c = 'G';
		else if (val == 3)
			c = 'T';
		s += c;
	}
	return s;
}

std::string reverse_complement(const std::string &seq) {
	std::string rc = seq;
	std::ranges::reverse(rc);
	for (char &c : rc) {
		if (c == 'A')
			c = 'T';
		else if (c == 'T')
			c = 'A';
		else if (c == 'C')
			c = 'G';
		else if (c == 'G')
			c = 'C';
	}
	return rc;
}

uint64_t encode_dna_string(const std::string &seq) {
	uint64_t pattern_enc = 0;
	for (size_t i = 0; i < seq.size(); ++i) {
		uint8_t c = 0;
		if (const char base = std::toupper(seq[i]); base == 'C')
			c = 1;
		else if (base == 'G')
			c = 2;
		else if (base == 'T')
			c = 3;
		pattern_enc |= static_cast<uint64_t>(c) << (i * 2);
	}
	return pattern_enc;
}

void run_search_pass(
	const std::string &label,
	const std::string &query,
	const PinnedHostBuffer<uint64_t> &genome_buf,
	const BinaryLoader &epi_loader,
	const std::vector<ChromosomeRange> &chr_index
) {
	const uint64_t pattern = encode_dna_string(query);
	SearchResults results =
		launch_bulge_search(genome_buf.data(), epi_loader.data(), genome_buf.size(), epi_loader.size(), pattern, 3, 0);

	for (size_t i = 0; i < results.count; ++i) {
		const uint32_t raw_pos = results.matches[i];
		const uint32_t block_idx = raw_pos / 32;

		if (block_idx >= genome_buf.size())
			continue;

		std::string raw_seq = decode_block_32bp(genome_buf[block_idx]);
		size_t global_base_pos = raw_pos;

		auto it = std::upper_bound(
			chr_index.begin(), chr_index.end(), global_base_pos, [](const size_t pos, const ChromosomeRange &range) {
				return pos < range.start_idx;
			}
		);

		std::string loc = "Unknown";
		if (it != chr_index.begin()) {
			const auto prev = std::prev(it);
			const size_t relative_pos = global_base_pos - prev->start_idx + 1;
			loc = prev->name + ":" + std::to_string(relative_pos);
		}

		std::cout << "    Hit Block " << std::left << std::setw(10) << block_idx << ": " << raw_seq
				  << "    Position: " << std::setw(20) << loc << " Strand: " << label
				  << std::endl;
	}
	free_search_results(results);
}

int main(const int argc, char **argv) {
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <fasta_file> <epi_file> <sequence>" << std::endl;
		return 1;
	}

	const std::string fasta_path = argv[1];
	const std::string epi_path = argv[2];
	const std::string query_seq = argv[3];

	std::cout << "[CORE] Starting benchmark..." << std::endl;
	std::cout << "[CORE] Query: " << query_seq << " (" << query_seq.length() << " bp)" << std::endl;
	std::cout << "================================================================" << std::endl;

	try {
		const GenomeLoader loader(fasta_path);
		const BinaryLoader epi_loader(epi_path);

		std::cout << "[STEP 3] Sanitizing/Loading Cache..." << std::endl;
		std::vector<ChromosomeRange> chr_index;
		double sanitize_time;
		const std::vector<uint8_t> clean_data =
			sanitize_genome(fasta_path, loader.data(), loader.size(), chr_index, sanitize_time);

		std::cout << "  > Clean Genome Size : " << clean_data.size() / (1024.0 * 1024.0) << " MB" << std::endl;

		std::cout << "[STEP 4] Encoding (AVX2)..." << std::endl;
		size_t num_blocks = (clean_data.size() + 31) / 32;
		const auto pinned_genome = std::make_unique<PinnedHostBuffer<uint64_t>>(num_blocks);
		encode_sequence_avx2(clean_data.data(), clean_data.size(), pinned_genome->data());

		std::cout << "[STEP 5] GPU Search (Dual Strand)..." << std::endl;
		std::cout << "----------------------------------------------------------------" << std::endl;

		run_search_pass("(+)", query_seq, *pinned_genome, epi_loader, chr_index);
		const std::string rc_query = reverse_complement(query_seq);
		run_search_pass("(-)", rc_query, *pinned_genome, epi_loader, chr_index);
		std::cout << "----------------------------------------------------------------" << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
