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
		const char c = "ACGT"[val];
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

uint64_t generate_care_mask(const size_t len, const bool is_forward) {
	uint64_t mask = 0;
	for (size_t i = 0; i < len; ++i) {
		bool care = true;
		if (is_forward) {
			if (i == 20)
				care = false;
		} else {
			if (i == 2)
				care = false;
		}

		if (care) {
			mask |= 3ULL << (i * 2);
		}
	}

	return mask;
}

void run_search_pass(
	const std::string &label,
	const std::string &query,
	const bool is_forward,
	const PinnedHostBuffer<uint64_t> &genome_buf,
	const std::vector<ChromosomeRange> &chr_index
) {
	const uint64_t pattern = encode_dna_string(query);
	const uint64_t mask = generate_care_mask(query.length(), is_forward);

	SearchResults results = launch_bulge_search(genome_buf.data(), nullptr, genome_buf.size(), 0, pattern, mask, 3, 0);

	for (size_t i = 0; i < results.count; ++i) {
		const uint32_t raw_pos = results.matches[i];
		const uint32_t block_idx = raw_pos / 32;

		if (block_idx >= genome_buf.size())
			continue;

		std::string raw_block = decode_block_32bp(genome_buf[block_idx]);
		if (block_idx + 1 < genome_buf.size()) {
			raw_block += decode_block_32bp(genome_buf[block_idx + 1]);
		} else {
			raw_block += "NNNNNNNN";
		}

		int best_mm = 999;
		int best_offset = -1;
		std::string matched_seq;

		for (int off = 0; off < 32; ++off) {
			std::string sub = raw_block.substr(off, 23);
			if (sub.length() < 23)
				break;

			bool pam_ok = false;
			if (is_forward) {
				if (sub[21] == query[21] && sub[22] == query[22]) {
					pam_ok = true;
				}
			} else {
				if (sub[0] == query[0] && sub[1] == query[1]) {
					pam_ok = true;
				}
			}

			if (!pam_ok)
				continue;

			int mm = 0;
			if (is_forward) {
				for (int k = 0; k < 20; ++k) {
					if (sub[k] != query[k])
						mm++;
				}
			} else {
				for (int k = 3; k < 23; ++k) {
					if (sub[k] != query[k])
						mm++;
				}
			}

			if (mm <= 3) {
				if (mm < best_mm) {
					best_mm = mm;
					best_offset = off;
					matched_seq = sub;
				}
			}
		}

		if (best_offset == -1)
			continue;

		size_t global_base_pos = static_cast<size_t>(block_idx) * 32 + best_offset;
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

		std::cout << "    Hit Block " << std::left << std::setw(10) << block_idx << ": " << matched_seq
				  << "    Position: " << std::setw(20) << loc << " Strand: " << label << " [Mis: " << best_mm << "]" << std::endl;
	}
	free_search_results(results);
}

int main(const int argc, char **argv) {
	if (argc < 4) {
		std::cerr << "Usage: " << argv[0] << " <fasta> <epi> <query>" << std::endl;
		return 1;
	}

	const std::string fasta_path = argv[1];
	const std::string query_seq = argv[3];
	std::cout << "[CORE] Starting Engine..." << std::endl;
	std::cout << "[CORE] Query: " << query_seq << std::endl;

	try {
		const GenomeLoader loader(fasta_path);
		std::vector<ChromosomeRange> chr_index;
		double t;
		const std::vector<uint8_t> clean_data = sanitize_genome(fasta_path, loader.data(), loader.size(), chr_index, t);
		size_t num_blocks = (clean_data.size() + 31) / 32;
		const auto pinned_genome = std::make_unique<PinnedHostBuffer<uint64_t>>(num_blocks);
		encode_sequence_avx2(clean_data.data(), clean_data.size(), pinned_genome->data());
		std::cout << "----------------------------------------------------------------" << std::endl;
		run_search_pass("(+)", query_seq, true, *pinned_genome, chr_index);
		const std::string rc_query = reverse_complement(query_seq);
		run_search_pass("(-)", rc_query, false, *pinned_genome, chr_index);
		std::cout << "----------------------------------------------------------------" << std::endl;
	} catch (const std::exception &e) {
		std::cerr << "[FATAL] " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
