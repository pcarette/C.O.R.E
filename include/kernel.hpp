#pragma once

#include <cstdint>
#include <vector>

struct SearchResults {
	uint32_t *matches;
	uint32_t count;
	uint32_t capacity;
	float time_ms;
};

SearchResults launch_bulge_search(
	const uint64_t *genome_data,
	const uint8_t *host_epigenome,
	size_t num_elements,
	size_t max_epi_size,
	uint64_t pattern,
	uint64_t care_mask,
	int max_mismatches,
	int max_seed_mismatches
);

void free_search_results(SearchResults &results);
