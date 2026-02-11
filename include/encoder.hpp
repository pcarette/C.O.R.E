#pragma once

#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

template <typename T> struct AlignedAllocator {
	static constexpr std::size_t ALIGNMENT = 64;

	using value_type = T;

	explicit AlignedAllocator() = default;

	template <class U> explicit constexpr AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

	static T *allocate(const std::size_t n) {
		if (n > static_cast<std::size_t>(-1) / sizeof(T))
			throw std::bad_alloc();
		void *ptr = std::aligned_alloc(ALIGNMENT, n * sizeof(T));
		if (!ptr)
			throw std::bad_alloc();
		return static_cast<T *>(ptr);
	}

	static void deallocate(T *p, std::size_t) noexcept {
		std::free(p);
	}
};

template <class T, class U> bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
	return true;
}

template <class T, class U> bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) {
	return false;
}

using AlignedVector = std::vector<uint64_t, AlignedAllocator<uint64_t>>;

AlignedVector encode_sequence_avx2(const uint8_t  *__restrict__ data, size_t size);
