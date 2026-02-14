#pragma once

#include <cstdint>
#include <vector>

#include "config.hpp"

struct SearchResults {
	uint32_t *matches;
	size_t count;
};

SearchResults launch_pipelined_search(
	const uint64_t *d_genome, size_t genome_num_u64, const EnzymeConfig &config, uint64_t query_pattern, uint64_t query_mask
);

void free_search_results(SearchResults &res);
