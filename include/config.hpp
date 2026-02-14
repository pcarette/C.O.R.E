#pragma once

#include <cstdint>
#include <string>

struct alignas(16) EnzymeConfig {
	uint64_t pam_pattern;
	uint64_t pam_care_mask;
	int pam_len;
	int target_len;
	int pam_offset_correction;
	int max_mismatches;

	explicit EnzymeConfig()
		: pam_pattern(20), pam_care_mask(0), pam_len(0), target_len(0), pam_offset_correction(0), max_mismatches(0) {}
};

class ConfigLoader {
	static void parse_pam_string(const std::string &pam_str, uint64_t &pattern, uint64_t &mask);

public:
	static EnzymeConfig load_from_json(const std::string &filepath);
};
