#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>

#include "loader.hpp"

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
	: data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
	  fd_(std::exchange(other.fd_, -1)) {}

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
