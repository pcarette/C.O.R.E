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
