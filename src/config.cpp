#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "config.hpp"

static std::string clean_json_token(const std::string &str) {
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

	if (pam_str.empty()) {
		throw std::runtime_error("Config Error: PAM Pattern is empty !");
	}

	for (size_t i = 0; i < pam_str.length(); ++i) {
		const char c = std::toupper(pam_str[i]);
		const int shift = i * 2;

		uint64_t p_bits = 0;
		uint64_t m_bits = 0;

		switch (c) {
		case 'A':
			p_bits = 0;
			m_bits = 3;
			break;
		case 'C':
			p_bits = 1;
			m_bits = 3;
			break;
		case 'G':
			p_bits = 2;
			m_bits = 3;
			break;
		case 'T':
			p_bits = 3;
			m_bits = 3;
			break;
		case 'N':
			p_bits = 0;
			m_bits = 0;
			break;
		case 'R':
			p_bits = 0;
			m_bits = 1;
			break;
		case 'Y':
			p_bits = 1;
			m_bits = 1;
			break;
		default:
			throw std::runtime_error("Unknown nucleotide in the PAM: " + std::string(1, c));
		}

		pattern |= (p_bits << shift);
		mask |= (m_bits << shift);
	}
}

EnzymeConfig ConfigLoader::load_from_json(const std::string &filepath) {
	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open the configuration file: " + filepath);
	}

	EnzymeConfig config;
	std::string line;
	std::string pam_str = "";
	std::string pam_dir = "";
	config.target_len = -1;
	config.max_mismatches = -1;

	std::cout << "[DEBUG] Reading the configuration: " << filepath << std::endl;

	while (std::getline(file, line)) {
		size_t colon_pos = line.find(':');
		if (colon_pos == std::string::npos)
			continue;

		std::string raw_key = line.substr(0, colon_pos);
		std::string raw_val = line.substr(colon_pos + 1);
		std::string key = clean_json_token(raw_key);
		std::string val = clean_json_token(raw_val);

		if (key == "enzyme_name") {
			// Optional, only for log for the moment
		} else if (key == "pam_pattern") {
			pam_str = val;
		} else if (key == "pam_direction") {
			pam_dir = val;
		} else if (key == "target_len") {
			try {
				config.target_len = std::stoi(val);
			} catch (...) {
				throw std::runtime_error("Invalid target_len (must be an integer): " + val);
			}
		} else if (key == "max_mismatches") {
			try {
				config.max_mismatches = std::stoi(val);
			} catch (...) {
				throw std::runtime_error("Invalid max_mismatches (must be an integer): " + val);
			}
		}
	}

	if (pam_str.empty())
		throw std::runtime_error("Invalid configuration: ‘pam_pattern’ missing.");
	if (pam_dir.empty())
		throw std::runtime_error("Invalid configuration: ‘pam_direction’ missing.");
	if (config.target_len <= 0)
		throw std::runtime_error("Invalid configuration: ‘target_len’ missing or <= 0.");

	config.pam_len = pam_str.length();
	parse_pam_string(pam_str, config.pam_pattern, config.pam_care_mask);

	if (pam_dir == "3_prime") {
		config.pam_offset_correction = -config.target_len;
	} else if (pam_dir == "5_prime") {
		config.pam_offset_correction = config.pam_len;
	} else {
		throw std::runtime_error("pam_direction unknown (expected: 3_prime or 5_prime): " + pam_dir);
	}

	std::cout << "[CONFIG] Loaded Enzyme Config:" << std::endl;
	std::cout << "  > Enzyme PAM      : " << pam_str << " (" << pam_dir << ")" << std::endl;
	std::cout << "  > Target Length   : " << config.target_len << " bp" << std::endl;
	std::cout << "  > Max Mismatches  : " << config.max_mismatches << std::endl;
	std::cout << "  > PAM Pattern Hex : 0x" << std::hex << config.pam_pattern << std::dec << std::endl;
	std::cout << "  > PAM Mask Hex    : 0x" << std::hex << config.pam_care_mask << std::dec << std::endl;
	std::cout << "  > Offset Align    : " << config.pam_offset_correction << std::endl;

	return config;
}