#pragma once

#include <cstdint>
#include <vector>

struct SearchResults {
	uint32_t *matches;
	uint32_t count;
	uint32_t capacity;
	float time_ms;
};

SearchResults launch_exact_search(const uint64_t *genome_data, size_t num_elements, uint64_t pattern);

SearchResults launch_bulge_search(
	const uint64_t *genome_data, size_t num_elements, uint64_t pattern, int max_mismatches, int max_seed_mismatches
);

void free_search_results(SearchResults &results);
