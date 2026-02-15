#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.hpp"
#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

namespace py = pybind11;

static std::string decode_32bp(const uint64_t b) {
	std::string s(32, 'A');
	for (int i = 0; i < 32; ++i) {
		if (const uint8_t v = (b >> (i * 2)) & 3; v == 1)
			s[i] = 'C';
		else if (v == 2)
			s[i] = 'G';
		else if (v == 3)
			s[i] = 'T';
	}
	return s;
}

static std::string get_rc(const std::string &s) {
	std::string r = s;
	std::ranges::reverse(r);
	for (char &c : r) {
		if (const char u = std::toupper(static_cast<unsigned char>(c)); u == 'A')
			c = 'T';
		else if (u == 'T')
			c = 'A';
		else if (u == 'C')
			c = 'G';
		else if (u == 'G')
			c = 'C';
		else if (u == 'R')
			c = 'Y';
		else if (u == 'Y')
			c = 'R';
		else if (u == 'N')
			c = 'N';
	}
	return r;
}

static uint64_t encode_dna_string(const std::string &seq) {
	uint64_t p = 0;
	for (size_t i = 0; i < std::min(seq.size(), static_cast<size_t>(32)); ++i) {
		uint64_t v = 0;
		if (const char b = std::toupper(seq[i]); b == 'C')
			v = 1;
		else if (b == 'G')
			v = 2;
		else if (b == 'T')
			v = 3;
		p |= (v << (i * 2));
	}
	return p;
}

static EnzymeConfig generate_reverse_config(const EnzymeConfig &fwd) {
	EnzymeConfig rev = fwd;
	uint64_t new_pat = 0;
	uint64_t new_mask = 0;

	for (int i = 0; i < fwd.pam_len; ++i) {
		const uint64_t shift_in = i * 2;
		const uint64_t base_val = (fwd.pam_pattern >> shift_in) & 3;
		const uint64_t mask_val = (fwd.pam_care_mask >> shift_in) & 3;
		uint64_t rc_base = 3 - base_val;
		const uint64_t rc_mask = mask_val;
		if (rc_mask == 0)
			rc_base = 0;

		const int shift_out = (fwd.pam_len - 1 - i) * 2;
		new_pat |= (rc_base << shift_out);
		new_mask |= (rc_mask << shift_out);
	}

	rev.pam_pattern = new_pat;
	rev.pam_care_mask = new_mask;
	if (fwd.pam_offset_correction == 0)
		rev.pam_offset_correction = -fwd.target_len;
	else
		rev.pam_offset_correction = 0;

	return rev;
}

struct PyHit {
	std::string chromosome;
	uint32_t position;
	std::string sequence;
	int mismatches;
	std::string strand;
};

class CoreEngine {
	std::unique_ptr<GenomeLoader> loader;
	std::unique_ptr<PinnedHostBuffer<uint64_t>> encoded_genome;
	std::vector<ChromosomeRange> chr_index;
	std::vector<bool> n_mask;
	size_t num_blocks;

	void process_hits(
		const SearchResults &raw,
		const std::string &q_seq,
		const std::string &strand,
		const EnzymeConfig &cfg,
		std::vector<PyHit> &out
	) {

		for (size_t i = 0; i < raw.count; ++i) {
			uint32_t pos = raw.matches[i];
			const uint32_t b_idx = pos / 32;
			if (b_idx >= num_blocks)
				continue;

			std::string ctx = decode_32bp((*encoded_genome)[b_idx]);
			if (b_idx + 1 < num_blocks)
				ctx += decode_32bp((*encoded_genome)[b_idx + 1]);

			const size_t local_off = pos % 32;
			if (local_off + q_seq.size() > ctx.size())
				continue;

			std::string cand = ctx.substr(local_off, q_seq.size());
			int mm = 0;
			for (size_t k = 0; k < q_seq.size(); ++k)
				if (cand[k] != q_seq[k])
					mm++;

			if (mm <= cfg.max_mismatches) {
				auto it = std::upper_bound(chr_index.begin(), chr_index.end(), pos, [](const size_t p, const ChromosomeRange &r) {
					return p < r.start_idx;
				});

				if (it != chr_index.begin()) {
					const auto prev = std::prev(it);
					PyHit hit;
					hit.chromosome = prev->name;
					hit.position = pos - prev->start_idx;
					hit.sequence = cand;
					hit.mismatches = mm;
					hit.strand = strand;
					out.push_back(hit);
				}
			}
		}
	}

public:
	explicit CoreEngine(const std::string &fasta_path) {
		py::print("[C++] Loading Genome from:", fasta_path);
		loader = std::make_unique<GenomeLoader>(fasta_path);
		double dummy;
		const auto clean_data = sanitize_genome(fasta_path, loader->data(), loader->size(), chr_index, n_mask, dummy);
		num_blocks = (clean_data.size() + 31) / 32;
		encoded_genome = std::make_unique<PinnedHostBuffer<uint64_t>>(num_blocks);
		py::print("[C++] Encoding Genome (AVX2)...");
		encode_sequence_avx2(clean_data.data(), clean_data.size(), encoded_genome->data());
		py::print("[C++] Ready.");
	}

	std::vector<PyHit> search(const std::string &query, const std::string &config_json_path) {
		std::vector<PyHit> results;
		const EnzymeConfig fwd_cfg = ConfigLoader::load_from_json(config_json_path);
		const EnzymeConfig rev_cfg = generate_reverse_config(fwd_cfg);
		const std::string rc_query = get_rc(query);
		SearchResults res_fwd, res_rev;
		{
			py::gil_scoped_release release;
			const uint64_t p_fwd = encode_dna_string(query);
			uint64_t m_fwd = 0;
			for (size_t i = 0; i < query.size(); ++i)
				m_fwd |= 3ULL << (i * 2);
			res_fwd = launch_pipelined_search(encoded_genome->data(), num_blocks, fwd_cfg, p_fwd, m_fwd);
			const uint64_t p_rev = encode_dna_string(rc_query);
			uint64_t m_rev = 0;
			for (size_t i = 0; i < rc_query.size(); ++i)
				m_rev |= 3ULL << (i * 2);
			res_rev = launch_pipelined_search(encoded_genome->data(), num_blocks, rev_cfg, p_rev, m_rev);
		}
		process_hits(res_fwd, query, "(+)", fwd_cfg, results);
		process_hits(res_rev, rc_query, "(-)", rev_cfg, results);
		free_search_results(res_fwd);
		free_search_results(res_rev);
		return results;
	}
};

PYBIND11_MODULE(core_engine, m) {
	m.doc() = "C.O.R.E High-Performance CRISPR Search Engine";

	py::class_<PyHit>(m, "Hit")
		.def_readonly("chrom", &PyHit::chromosome)
		.def_readonly("pos", &PyHit::position)
		.def_readonly("sequence", &PyHit::sequence)
		.def_readonly("mismatches", &PyHit::mismatches)
		.def_readonly("strand", &PyHit::strand)
		.def("__repr__", [](const PyHit &h) {
			return "<Hit " + h.chromosome + ":" + std::to_string(h.position) + " " + h.strand + " (" +
				   std::to_string(h.mismatches) + "mm)>";
		});

	py::class_<CoreEngine>(m, "Engine")
		.def(py::init<const std::string &>())
		.def("search", &CoreEngine::search);
}
