#pragma once

#include <cstdint>
#include <vector>

size_t encode_sequence_avx2(const uint8_t  *__restrict__ data, size_t size, uint64_t *__restrict__ output_buffer);
