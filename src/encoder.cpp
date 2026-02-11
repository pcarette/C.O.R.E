#include <cstring>
#include <immintrin.h>

#include "encoder.hpp"

alignas(32) static const __m256i SHUFFLE_LUT =
	_mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0);

static constexpr uint64_t MASK_ODD = 0x5555555555555555ULL;
static constexpr uint64_t MASK_EVEN = 0xAAAAAAAAAAAAAAAAULL;

static __always_inline uint8_t encode_char_scalar(const char c) {
	switch (c & 0xDF) {
	case 'C':
		return 1;
	case 'G':
		return 2;
	case 'T':
		return 3;
	default:
		return 0;
	}
}

static __always_inline uint64_t process_block_32(const __m256i &raw) {
	const __m256i vals = _mm256_shuffle_epi8(SHUFFLE_LUT, raw);
	const uint32_t mask0 = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(vals, 7)));
	const uint32_t mask1 = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(vals, 6)));
	return _pdep_u64(mask0, MASK_ODD) | _pdep_u64(mask1, MASK_EVEN);
}

AlignedVector encode_sequence_avx2(const uint8_t *__restrict__ data, const size_t size) {
	const size_t num_u64 = (size + 31) / 32;
	AlignedVector result(num_u64);
	uint64_t *__restrict__ res_ptr = result.data();
	const size_t chunks_256 = size / 256;

#pragma omp parallel for schedule(dynamic, 1024)
	for (size_t i = 0; i < chunks_256; ++i) {
		const size_t in_offset = i * 256;
		const size_t out_offset = i * 8;
		__m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset));
		__m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 32));
		__m256i r2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 64));
		__m256i r3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 96));
		__m256i r4 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 128));
		__m256i r5 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 160));
		__m256i r6 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 192));
		__m256i r7 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + in_offset + 224));
		const uint64_t v0 = process_block_32(r0);
		const uint64_t v1 = process_block_32(r1);
		const uint64_t v2 = process_block_32(r2);
		const uint64_t v3 = process_block_32(r3);
		const uint64_t v4 = process_block_32(r4);
		const uint64_t v5 = process_block_32(r5);
		const uint64_t v6 = process_block_32(r6);
		const uint64_t v7 = process_block_32(r7);
		const __m256i out_vec_a = _mm256_set_epi64x(v3, v2, v1, v0);
		const __m256i out_vec_b = _mm256_set_epi64x(v7, v6, v5, v4);
		_mm256_stream_si256(res_ptr + out_offset, out_vec_a);
		_mm256_stream_si256(res_ptr + out_offset + 4, out_vec_b);
	}

	size_t processed_bytes = chunks_256 * 256;
	size_t block_idx = processed_bytes / 32;

	for (size_t k = processed_bytes; k + 32 <= size; k += 32) {
		__m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + k));
		result[block_idx++] = process_block_32(r);
		processed_bytes += 32;
	}

	if (processed_bytes < size) {
		uint64_t current_block = 0;
		int bit_pos = 0;
		for (size_t k = processed_bytes; k < size; ++k) {
			if (data[k] == '\n' || data[k] == '\r')
				continue;
			const uint8_t val = encode_char_scalar(data[k]);
			current_block |= (static_cast<uint64_t>(val) << bit_pos);
			bit_pos += 2;
		}

		result[block_idx] = current_block;
	}

	_mm_sfence();

	return result;
}
