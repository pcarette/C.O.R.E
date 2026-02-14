#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

namespace py = pybind11;

uint64_t encode_pattern_python(const std::string &seq) {
	uint64_t pat = 0;
	for (size_t i = 0; i < seq.size() && i < 32; ++i) {
		uint8_t code = 0;
		const uint8_t c = std::toupper(seq[i]);
		if (c == 'C')
			code = 1;
		else if (c == 'G')
			code = 2;
		else if (c == 'T')
			code = 3;
		else
			code = 0;
		pat |= static_cast<uint64_t>(code) << (i * 2);
	}
	return pat;
}

uint64_t generate_care_mask(const std::string &seq) {
	uint64_t mask = 0;
	for (size_t i = 0; i < seq.size() && i < 32; ++i)
		mask |= 3ULL << (i * 2);
	if (seq.size() == 23)
		mask &= ~(3ULL << (20 * 2));
	return mask;
}

std::string decode_2bit(const uint64_t block) {
	std::string s;
	s.reserve(32);
	for (int i = 0; i < 32; ++i) {
		if (const uint8_t val = (block >> (i * 2)) & 0x3; val == 0)
			s += 'A';
		else if (val == 1)
			s += 'C';
		else if (val == 2)
			s += 'G';
		else
			s += 'T';
	}
	return s;
}

class CoreEngine {
	std::unique_ptr<PinnedHostBuffer<uint64_t>> pinned_genome_;
	std::unique_ptr<BinaryLoader> epi_loader_;
	std::vector<ChromosomeRange> chr_index_;

public:
	explicit CoreEngine(const std::string &fasta_path, const std::string &epi_path) {
		const GenomeLoader loader(fasta_path);
		epi_loader_ = std::make_unique<BinaryLoader>(epi_path);
		double dummy_time;
		const uint8_t *raw = loader.data();
		volatile uint8_t sink = 0;
		for (size_t i = 0; i < loader.size(); i += 4096)
			sink = raw[i];
		const std::vector<uint8_t> clean_data = sanitize_genome(fasta_path, loader.data(), loader.size(), chr_index_, dummy_time);
		size_t num_blocks = (clean_data.size() + 31) / 32;
		pinned_genome_ = std::make_unique<PinnedHostBuffer<uint64_t>>(num_blocks);
		encode_sequence_avx2(clean_data.data(), clean_data.size(), pinned_genome_->data());
	}

	py::array_t<uint32_t> search(const std::string &pattern_seq, const int max_mismatches = 3) {
		if (pattern_seq.length() > 32) {
			throw std::runtime_error("Pattern length must be <= 32bp");
		}

		const uint64_t pattern = encode_pattern_python(pattern_seq);
		const uint64_t mask = generate_care_mask(pattern_seq);
		SearchResults results = launch_bulge_search(
			pinned_genome_->data(),
			epi_loader_->data(),
			pinned_genome_->size(),
			epi_loader_->size(),
			pattern,
			mask,
			max_mismatches,
			0
		);
		auto result_array = py::array_t<uint32_t>(results.count);
		if (results.count > 0) {
			const py::buffer_info buffer = result_array.request();
			auto *ptr = static_cast<uint32_t *>(buffer.ptr);
			std::memcpy(ptr, results.matches, results.count * sizeof(uint32_t));
		}

		free_search_results(results);
		return result_array;
	}

	std::pair<std::string, std::string> resolve_location(const uint32_t base_idx) {
		const size_t global_base_pos = base_idx;
		const auto it = std::upper_bound(
			chr_index_.begin(), chr_index_.end(), global_base_pos, [](const size_t pos, const ChromosomeRange &range) {
				return pos < range.start_idx;
			}
		);

		std::string loc = "Unknown";
		if (it != chr_index_.begin()) {
			const auto prev = std::prev(it);
			loc = prev->name + ":" + std::to_string(global_base_pos - prev->start_idx + 1);
		}

		const uint32_t block_idx = base_idx / 32;
		if (block_idx >= pinned_genome_->size())
			return {loc, "ERROR_OUT_OF_BOUNDS"};

		return {loc, decode_2bit((*pinned_genome_)[block_idx])};
	}

	size_t get_genome_size_blocks() const {
		return pinned_genome_->size();
	}
};

PYBIND11_MODULE(core_engine, m) {
	m.doc() = "C.O.R.E Engine";
	py::class_<CoreEngine>(m, "CoreEngine")
		.def(
			py::init<const std::string &, const std::string &>(),
			py::arg("fasta_path"),
			py::arg("epi_path"),
			"Init engine: load, clean and encode genome"
		)
		.def(
			"search",
			&CoreEngine::search,
			py::arg("sequence"),
			py::arg("max_mismatches") = 3,
			"Search a sequence with a mismatch tolerance. Return a NumPy array of blocks ID's"
		)
		.def(
			"resolve_location",
			&CoreEngine::resolve_location,
			py::arg("block_idx"),
			"Convert a Block ID to genomic coordinates (chr:pos)"
		)
		.def_property_readonly("total_blocks", &CoreEngine::get_genome_size_blocks);
}
