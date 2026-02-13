#pragma once

#include <cstdint>
#include <string_view>

class GenomeLoader {
	uint8_t *data_;
	size_t size_;
	int fd_;

public:
	explicit GenomeLoader(const std::string &filepath);

	~GenomeLoader();

	GenomeLoader(const GenomeLoader &) = delete;

	GenomeLoader &operator=(const GenomeLoader &) = delete;

	GenomeLoader(GenomeLoader &&other) noexcept;

	GenomeLoader &operator=(GenomeLoader &&other) noexcept;

	[[nodiscard]]
	std::string_view get_view() const;

	[[nodiscard]]
	const uint8_t *data() const {
		return data_;
	}

	[[nodiscard]]
	size_t size() const {
		return size_;
	}
};

class BinaryLoader {
	int m_fd_;
	size_t m_size_;
	uint8_t *m_data_;

public:
	explicit BinaryLoader(const std::string &filepath);

	~BinaryLoader();

	BinaryLoader(const BinaryLoader &) = delete;

	BinaryLoader &operator=(const BinaryLoader &) = delete;

	[[nodiscard]] const uint8_t *data() const {
		return m_data_;
	}

	[[nodiscard]] size_t size() const {
		return m_size_;
	}
};
