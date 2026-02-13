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
		const std::vector<uint8_t> clean_data = sanitize_genome(loader.data(), loader.size(), chr_index_, dummy_time);
		size_t num_blocks = (clean_data.size() + 31) / 32;
		pinned_genome_ = std::make_unique<PinnedHostBuffer<uint64_t>>(num_blocks);
		encode_sequence_avx2(clean_data.data(), clean_data.size(), pinned_genome_->data());
	}

	py::array_t<uint32_t> search(const std::string &pattern_seq, int max_mismatches = 3) {
		if (pattern_seq.length() > 32) {
			throw std::runtime_error("Pattern length must be <= 32bp");
		}

		uint64_t pattern = encode_pattern_python(pattern_seq);
		SearchResults results = launch_bulge_search(
			pinned_genome_->data(), epi_loader_->data(), pinned_genome_->size(), epi_loader_->size(), pattern, max_mismatches, 0
		);
		auto result_array = py::array_t<uint32_t>(results.count);
		if (results.count > 0) {
			py::buffer_info buffer = result_array.request();
			uint32_t *ptr = static_cast<uint32_t *>(buffer.ptr);
			std::memcpy(ptr, results.matches, results.count * sizeof(uint32_t));
		}

		free_search_results(results);
		return result_array;
	}

	std::string resolve_location(uint32_t block_idx) {
		size_t global_base_pos = static_cast<size_t>(block_idx) * 32;
		auto it =
			std::upper_bound(chr_index_.begin(), chr_index_.end(), global_base_pos, [](size_t pos, const ChromosomeRange &range) {
				return pos < range.start_idx;
			});

		if (it != chr_index_.begin()) {
			--it;
			const size_t offset = global_base_pos - it->start_idx;
			return it->name + ":" + std::to_string(offset + 1);
		}

		return "Unknown: " + std::to_string(global_base_pos);
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
