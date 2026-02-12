#include <cstdint>
#include <immintrin.h>
#include <omp.h>

#include "encoder.hpp"

alignas(32) static const __m256i SHUFFLE_LUT = _mm256_setr_epi8(
	0x00,
	0x00,
	0x00,
	0x40, // 0-3 (3=C)
	0xC0,
	0x00,
	0x00,
	0x80, // 4-7 (4=T, 7=G)
	0x00,
	0x00,
	0x00,
	0x00, // 8-11
	0x00,
	0x00,
	0x00,
	0x00, // 12-15 (14=N -> 00)
	0x00,
	0x00,
	0x00,
	0x40,
	0xC0,
	0x00,
	0x00,
	0x80,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00
);

static constexpr uint64_t MASK_ODD_BITS = 0xAAAAAAAAAAAAAAAAULL;
static constexpr uint64_t MASK_EVEN_BITS = 0x5555555555555555ULL;

static __always_inline uint8_t encode_char_scalar(const uint8_t c) {
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
	const auto mask_hi = static_cast<uint32_t>(_mm256_movemask_epi8(vals));
	const auto mask_lo = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_slli_epi16(vals, 1)));
	return _pdep_u64(mask_hi, MASK_ODD_BITS) | _pdep_u64(mask_lo, MASK_EVEN_BITS);
}

size_t encode_sequence_avx2(const uint8_t *__restrict__ data, const size_t size, uint64_t *__restrict__ output_buffer) {
	const size_t num_u64 = (size + 31) / 32;
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
		_mm256_stream_si256(output_buffer + out_offset, _mm256_set_epi64x(v3, v2, v1, v0));
		_mm256_stream_si256(output_buffer + out_offset + 4, _mm256_set_epi64x(v7, v6, v5, v4));
	}

	size_t processed_bytes = chunks_256 * 256;
	size_t block_idx = processed_bytes / 32;

	for (size_t k = processed_bytes; k + 32 <= size; k += 32) {
		__m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + k));
		output_buffer[block_idx++] = process_block_32(r);
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
			if (bit_pos == 64) {
				output_buffer[block_idx++] = current_block;
				current_block = 0;
				bit_pos = 0;
			}
		}

		if (bit_pos > 0)
			output_buffer[block_idx] = current_block;
	}

	_mm_sfence();

	return num_u64;
}
