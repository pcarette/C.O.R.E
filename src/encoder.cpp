#include <cstring>
#include <immintrin.h>

#include "encoder.hpp"

const __m512i SHUFFLE_LUT = _mm512_set_epi8(
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // 15-8
	2,
	0,
	0,
	3,
	1,
	0,
	0,
	0, // 7 (G), 6, 5, 4 (T), 3 (C), 2, 1 (A), 0
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0, // Repeat for others lanes (AVX512 repeats the 128b pattern)
	2,
	0,
	0,
	3,
	1,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	2,
	0,
	0,
	3,
	1,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	0,
	2,
	0,
	0,
	3,
	1,
	0,
	0,
	0
);

static __always_inline uint8_t encode_char_scalar(const char c) {
	switch (c & 0xDF) {
	case 'A':
		return 0;
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

AlignedVector encode_sequence_avx512(const uint8_t *data, const size_t size) {
	const size_t num_blocks = (size + 31) / 32;
	AlignedVector result(num_blocks);

	size_t i = 0;
	for (; i + 64 <= size; i += 64) {
		const __m512i raw_ascii = _mm512_loadu_si512(data + i);
		const __m512i vals = _mm512_shuffle_epi8(SHUFFLE_LUT, raw_ascii);
		const __mmask64 mask_bit_0 = _mm512_test_epi8_mask(vals, _mm512_set1_epi8(0x01));
		const __mmask64 mask_bit_1 = _mm512_test_epi8_mask(vals, _mm512_set1_epi8(0x02));
		const uint64_t lsb_0 = static_cast<uint32_t>(mask_bit_0);
		const uint64_t hsb_0 = static_cast<uint32_t>(mask_bit_1);
		const uint64_t encoded_chunk_0 =
			_pdep_u64(lsb_0, 0x5555555555555555ULL) | _pdep_u64(hsb_0, 0xAAAAAAAAAAAAAAAAULL);
		const uint64_t lsb_1 = static_cast<uint32_t>(mask_bit_0 >> 32);
		const uint64_t hsb_1 = static_cast<uint32_t>(mask_bit_1 >> 32);
		const uint64_t encoded_chunk_1 =
			_pdep_u64(lsb_1, 0x5555555555555555ULL) | _pdep_u64(hsb_1, 0xAAAAAAAAAAAAAAAAULL);
		result[i / 32] = encoded_chunk_0;
		result[i / 32 + 1] = encoded_chunk_1;
	}

	if (i < size) {
		size_t block_idx = i / 32;
		uint64_t current_block = 0;
		int bit_pos = 0;
		for (; i < size; ++i) {
			if (data[i] == '\n')
				continue;
			const uint8_t val = encode_char_scalar(data[i]);
			current_block |= (static_cast<uint64_t>(val) << bit_pos);
			bit_pos += 2;
			if (bit_pos == 64) {
				result[block_idx++] = current_block;
				current_block = 0;
				bit_pos = 0;
			}
		}

		if (bit_pos > 0)
			result[block_idx] = current_block;
	}

	return result;
}
