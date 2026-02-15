#pragma once

#include <cstdint>
#include <vector>

#include "config.hpp"

struct SearchResults {
	uint32_t *matches;
	size_t count;
};

SearchResults launch_pipelined_search(
	const uint64_t *h_gen, const uint8_t *h_epi, size_t n, const EnzymeConfig &cfg, uint64_t pat, uint64_t mask
);

void free_search_results(SearchResults &res);
