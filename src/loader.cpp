#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>

#include <immintrin.h>

#include "loader.hpp"

#include <filesystem>

namespace fs = std::filesystem;

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
	std::cout << "[LOADER] Cache created: " << bin_path << std::endl;
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
	index.reserve(count);
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
	const size_t raw_size,
	std::vector<ChromosomeRange> &index,
	double &duration_ms
) {
	const auto start = std::chrono::high_resolution_clock::now();
	std::vector<uint8_t> clean_data;

	if (load_cache(filepath, clean_data, index)) {
		const auto end_time = std::chrono::high_resolution_clock::now();
		duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count() / 1000.0;
		std::cout << "[LOADER] Hot Cache Hit! (IO-Bound only)" << std::endl;
		return clean_data;
	}

	std::cout << "[LOADER] Cache miss. Starting AVX2 sanitization..." << std::endl;
	clean_data.resize(raw_size);

	uint8_t *dst = clean_data.data();
	const uint8_t *dst_start = clean_data.data();
	const uint8_t *src = raw_data;
	const uint8_t *end = raw_data + raw_size;

	const __m256i thresh = _mm256_set1_epi8(32);
	const __m256i gt_char = _mm256_set1_epi8('>');

	while (src < end) {
		if (*src == '>') {
			const char *name_start = reinterpret_cast<const char *>(src) + 1;
			const void *sep_space = std::memchr(name_start, ' ', end - src);
			const void *sep_newline = std::memchr(name_start, '\n', end - src);
			const char *name_end = reinterpret_cast<const char *>(end);
			if (sep_space && sep_space < name_end)
				name_end = static_cast<const char *>(sep_space);
			if (sep_newline && sep_newline < name_end)
				name_end = static_cast<const char *>(sep_newline);
			const size_t current_offset = dst - dst_start;
			index.push_back({std::string(name_start, name_end), current_offset});
			if (const void *newline_pos = std::memchr(src, '\n', end - src))
				src = static_cast<const uint8_t *>(newline_pos) + 1;
			else
				break;
			continue;
		}

		while (src + 32 <= end) {
			if (*src == '>')
				break;

			const __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src));
			const __m256i is_valid = _mm256_cmpgt_epi8(chunk, thresh);
			const int mask = _mm256_movemask_epi8(is_valid);
			const __m256i is_header = _mm256_cmpeq_epi8(chunk, gt_char);

			if (const int header_mask = _mm256_movemask_epi8(is_header); header_mask != 0)
				break;

			if (mask == -1) {
				_mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), chunk);
				dst += 32;
				src += 32;
			} else {
				const uint8_t *limit = src + 32;
				while (src < limit) {
					if (*src == '>')
						break;
					if (*src > 32)
						*dst++ = *src;
					src++;
				}
			}
		}

		while (src < end && *src != '>') {
			if (*src > 32)
				*dst++ = *src;
			src++;
		}
	}

	const size_t actual_size = dst - clean_data.data();
	clean_data.resize(actual_size);
	save_cache(filepath, clean_data, index);
	const auto end_time = std::chrono::high_resolution_clock::now();
	duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start).count() / 1000.0;
	return clean_data;
}

GenomeLoader::GenomeLoader(const std::string &filepath) : data_(nullptr), size_(0), fd_(-1) {
	fd_ = open(filepath.c_str(), O_RDONLY);
	if (fd_ == -1)
		throw std::runtime_error("GenomeLoader: Unable to open file " + filepath);

	struct stat sb{};
	if (fstat(fd_, &sb) == -1) {
		close(fd_);
		throw std::runtime_error("GenomeLoader: fstat error on " + filepath);
	}
	size_ = static_cast<size_t>(sb.st_size);
	data_ = static_cast<uint8_t *>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
	if (data_ == MAP_FAILED) {
		close(fd_);
		throw std::runtime_error("GenomeLoader: mmap error (OOM?)");
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
		throw std::runtime_error("BinaryLoader: mmap failed for " + filepath);
	}

	madvise(m_data_, m_size_, MADV_SEQUENTIAL);
}

BinaryLoader::~BinaryLoader() {
	if (m_data_ && m_data_ != MAP_FAILED)
		munmap(m_data_, m_size_);
	if (m_fd_ != -1)
		close(m_fd_);
}
