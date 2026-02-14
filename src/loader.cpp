#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "loader.hpp"

#include <utility>

namespace fs = std::filesystem;

constexpr size_t CHROM_PADDING_BYTES = 16;

void save_cache(const std::string &base_path, const std::vector<uint8_t> &data, const std::vector<ChromosomeRange> &index) {
	std::string bin_path = base_path + ".cache.bin";
	std::string idx_path = base_path + ".cache.idx";
	std::ofstream out_data(bin_path, std::ios::binary);
	out_data.write(reinterpret_cast<const char *>(data.data()), data.size());
	std::ofstream out_idx(idx_path, std::ios::binary);
	size_t count = index.size();
	out_idx.write(reinterpret_cast<const char *>(&count), sizeof(count));
	for (const auto &chr : index) {
		size_t name_len = chr.name.size();
		out_idx.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
		out_idx.write(chr.name.data(), name_len);
		out_idx.write(reinterpret_cast<const char *>(&chr.start_idx), sizeof(chr.start_idx));
	}
}

bool load_cache(const std::string &base_path, std::vector<uint8_t> &data, std::vector<ChromosomeRange> &index) {
	std::string bin_path = base_path + ".cache.bin";
	std::string idx_path = base_path + ".cache.idx";
	if (!fs::exists(bin_path) || !fs::exists(idx_path))
		return false;

	std::ifstream in_data(bin_path, std::ios::binary | std::ios::ate);
	size_t data_size = in_data.tellg();
	in_data.seekg(0);
	data.resize(data_size);
	in_data.read(reinterpret_cast<char *>(data.data()), data_size);

	std::ifstream in_idx(idx_path, std::ios::binary);
	size_t count = 0;
	in_idx.read(reinterpret_cast<char *>(&count), sizeof(count));
	index.clear();
	for (size_t i = 0; i < count; ++i) {
		size_t name_len;
		in_idx.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
		std::string name(name_len, '\0');
		in_idx.read(name.data(), name_len);
		size_t start_idx;
		in_idx.read(reinterpret_cast<char *>(&start_idx), sizeof(start_idx));
		index.push_back({name, start_idx});
	}
	return true;
}

std::vector<uint8_t> sanitize_genome(
	const std::string &filepath,
	const uint8_t *raw_data,
	size_t raw_size,
	std::vector<ChromosomeRange> &index,
	double &duration_ms
) {
	const auto start = std::chrono::high_resolution_clock::now();
	std::vector<uint8_t> clean_data;

	if (load_cache(filepath, clean_data, index)) {
		duration_ms = 0.0;
		std::cout << "[LOADER] Cache loaded (" << clean_data.size() / 1024 / 1024 << " MB)" << std::endl;
		return clean_data;
	}

	std::cout << "[LOADER] Sanitizing Genome (Removing Ns, Adding Padding)..." << std::endl;

	clean_data.reserve(raw_size / 2);

	const uint8_t *src = raw_data;
	const uint8_t *end = raw_data + raw_size;

	while (src < end) {
		if (*src == '>') {
			if (!clean_data.empty()) {
				for (size_t p = 0; p < CHROM_PADDING_BYTES; ++p)
					clean_data.push_back(0);
			}

			const char *name_start = reinterpret_cast<const char *>(src) + 1;
			const char *name_end = name_start;
			while (name_end < reinterpret_cast<const char *>(end) && *name_end != '\n' && *name_end != ' ') {
				name_end++;
			}

			size_t current_idx_bases = clean_data.size();
			index.push_back({std::string(name_start, name_end), current_idx_bases});
			src = static_cast<const uint8_t *>(std::memchr(src, '\n', end - src));
			if (src)
				src++;
			else
				break;
			continue;
		}

		while (src < end && *src != '>') {
			if (uint8_t c = *src; c != '\n' && c != '\r') {
				uint8_t val = 0;
				c = c & 0xDF;
				if (c == 'C')
					val = 1;
				else if (c == 'G')
					val = 2;
				else if (c == 'T')
					val = 3;

				clean_data.push_back(val);
			}
			src++;
		}
	}

	save_cache(filepath, clean_data, index);
	const auto end_time = std::chrono::high_resolution_clock::now();
	duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start).count();
	return clean_data;
}

GenomeLoader::GenomeLoader(const std::string &filepath) : data_(nullptr), size_(0), fd_(-1) {
	fd_ = open(filepath.c_str(), O_RDONLY);
	if (fd_ == -1)
		throw std::runtime_error("GenomeLoader: Unable to open file " + filepath);
	struct stat sb{};
	if (fstat(fd_, &sb) == -1) {
		close(fd_);
		throw std::runtime_error("GenomeLoader: fstat error");
	}
	size_ = static_cast<size_t>(sb.st_size);
	data_ = static_cast<uint8_t *>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
	if (data_ == MAP_FAILED) {
		close(fd_);
		throw std::runtime_error("GenomeLoader: mmap error");
	}
	madvise(data_, size_, MADV_SEQUENTIAL);
}

GenomeLoader::~GenomeLoader() {
	if (data_ && data_ != MAP_FAILED)
		munmap(data_, size_);
	if (fd_ != -1)
		close(fd_);
}

GenomeLoader::GenomeLoader(GenomeLoader &&other) noexcept
	: data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)), fd_(std::exchange(other.fd_, -1)) {}

GenomeLoader &GenomeLoader::operator=(GenomeLoader &&other) noexcept {
	if (this != &other) {
		this->~GenomeLoader();
		data_ = std::exchange(other.data_, nullptr);
		size_ = std::exchange(other.size_, 0);
		fd_ = std::exchange(other.fd_, -1);
	}
	return *this;
}

std::string_view GenomeLoader::get_view() const {
	return {reinterpret_cast<const char *>(data_), size_};
}

BinaryLoader::BinaryLoader(const std::string &filepath) : m_fd_(-1), m_size_(0), m_data_(nullptr) {
	m_fd_ = open(filepath.c_str(), O_RDONLY);
	if (m_fd_ == -1)
		throw std::runtime_error("BinaryLoader: Cannot open file " + filepath);

	struct stat sb{};
	if (fstat(m_fd_, &sb) == -1) {
		close(m_fd_);
		throw std::runtime_error("BinaryLoader: Cannot stat file " + filepath);
	}

	m_size_ = static_cast<size_t>(sb.st_size);
	m_data_ = static_cast<uint8_t *>(mmap(nullptr, m_size_, PROT_READ, MAP_PRIVATE, m_fd_, 0));
	if (m_data_ == MAP_FAILED) {
		close(m_fd_);
		throw std::runtime_error("BinaryLoader: mmap failed");
	}

	madvise(m_data_, m_size_, MADV_SEQUENTIAL);
}

BinaryLoader::~BinaryLoader() {
	if (m_data_ && m_data_ != MAP_FAILED)
		munmap(m_data_, m_size_);
	if (m_fd_ != -1)
		close(m_fd_);
}
