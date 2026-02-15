#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "config.hpp"

static std::string clean_token(const std::string &str) {
	const std::string junk = " \t\n\r\",{}";
	const size_t first = str.find_first_not_of(junk);
	if (std::string::npos == first)
		return "";
	const size_t last = str.find_last_not_of(junk);
	return str.substr(first, (last - first + 1));
}

void ConfigLoader::parse_pam_string(const std::string &pam_str, uint64_t &pattern, uint64_t &mask) {
	pattern = 0;
	mask = 0;
	for (size_t i = 0; i < pam_str.length(); ++i) {
		const char c = std::toupper(pam_str[i]);
		const int shift = i * 2;
		uint64_t p = 0, m = 0;
		switch (c) {
		case 'A':
			p = 0;
			m = 3;
			break;
		case 'C':
			p = 1;
			m = 3;
			break;
		case 'G':
			p = 2;
			m = 3;
			break;
		case 'T':
			p = 3;
			m = 3;
			break;
		case 'R':
			p = 0;
			m = 1;
			break;
		case 'N':
			p = 0;
			m = 0;
			break;
		default:
			throw std::runtime_error("Invalid PAM base");
		}
		pattern |= (p << shift);
		mask |= (m << shift);
	}
}

EnzymeConfig ConfigLoader::load_from_json(const std::string &filepath) {
	std::ifstream file(filepath);
	if (!file.is_open())
		throw std::runtime_error("Config file missing");
	EnzymeConfig cfg;
	std::string line, pam_str, pam_dir;
	while (std::getline(file, line)) {
		size_t colon = line.find(':');
		if (colon == std::string::npos)
			continue;
		std::string k = clean_token(line.substr(0, colon));
		std::string v = clean_token(line.substr(colon + 1));
		if (k == "pam_pattern")
			pam_str = v;
		else if (k == "pam_direction")
			pam_dir = v;
		else if (k == "target_len")
			cfg.target_len = std::stoi(v);
		else if (k == "max_mismatches")
			cfg.max_mismatches = std::stoi(v);
		else if (k == "seed_len")
			cfg.seed_len = std::stoi(v);
		else if (k == "max_seed_mismatches")
			cfg.max_seed_mismatches = std::stoi(v);
	}
	cfg.pam_len = pam_str.length();
	parse_pam_string(pam_str, cfg.pam_pattern, cfg.pam_care_mask);
	if (pam_dir == "3_prime")
		cfg.pam_offset_correction = -cfg.target_len;
	else
		cfg.pam_offset_correction = 0;
	return cfg;
}
