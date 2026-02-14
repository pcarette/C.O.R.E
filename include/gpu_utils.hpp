#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CHECK_CUDA(call)                                                                                                         \
	{                                                                                                                            \
		cudaError_t err = call;                                                                                                  \
		if (err != cudaSuccess) {                                                                                                \
			fprintf(stderr, "[GPU FATAL] CUDA Error at %s:%d -> %s\n", __FILE__, __LINE__, cudaGetErrorString(err));             \
			exit(1);                                                                                                             \
		}                                                                                                                        \
	}

template <typename T> class DeviceBuffer {
	T *ptr_ = nullptr;
	size_t size_ = 0;

public:
	explicit DeviceBuffer(const size_t size) : size_(size) {
		if (size > 0) {
			CHECK_CUDA(cudaMalloc(&ptr_, size * sizeof(T)));
			CHECK_CUDA(cudaMemset(ptr_, 0, size * sizeof(T)));
		}
	}

	~DeviceBuffer() {
		if (ptr_)
			cudaFree(ptr_);
	}

	DeviceBuffer(const DeviceBuffer &) = delete;

	DeviceBuffer &operator=(const DeviceBuffer &) = delete;

	T *data() const {
		return ptr_;
	}

	[[nodiscard]] size_t size() const {
		return size_;
	}

	void copyFromHost(const T *host_ptr, const size_t count) {
		CHECK_CUDA(cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copyToHost(T *host_ptr, const size_t count) {
		CHECK_CUDA(cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
	}

	void copyFromHostAsync(const T *host_ptr, const size_t count, const cudaStream_t stream) {
		CHECK_CUDA(cudaMemcpyAsync(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream));
	}

	void copyToHostAsync(T *host_ptr, const size_t count, const cudaStream_t stream) {
		CHECK_CUDA(cudaMemcpyAsync(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}
};

class CudaStream {
	cudaStream_t stream_ = nullptr;

public:
	explicit CudaStream() {
		CHECK_CUDA(cudaStreamCreate(&stream_));
	}

	~CudaStream() {
		cudaStreamDestroy(stream_);
	}

	[[nodiscard]] cudaStream_t get() const {
		return stream_;
	}

	void synchronize() const {
		CHECK_CUDA(cudaStreamSynchronize(stream_));
	}
};

template <typename T> class PinnedHostBuffer {
	T *ptr_ = nullptr;
	size_t size_ = 0;

public:
	explicit PinnedHostBuffer(const size_t size) : size_(size) {
		if (size > 0)
			CHECK_CUDA(cudaMallocHost(&ptr_, size * sizeof(T)));
	}

	~PinnedHostBuffer() {
		if (ptr_)
			cudaFreeHost(ptr_);
	}

	PinnedHostBuffer(const PinnedHostBuffer &) = delete;

	PinnedHostBuffer &operator=(const PinnedHostBuffer &) = delete;

	T *data() const {
		return ptr_;
	}

	[[nodiscard]] size_t size() const {
		return size_;
	}

	T &operator[](size_t index) {
		return ptr_[index];
	}

	const T &operator[](size_t index) const {
		return ptr_[index];
	}
};

class CudaEvent {
	cudaEvent_t event_ = nullptr;

public:
	explicit CudaEvent() {
		CHECK_CUDA(cudaEventCreate(&event_));
	}

	~CudaEvent() {
		cudaEventDestroy(event_);
	}

	void record(const cudaStream_t stream = nullptr) const {
		CHECK_CUDA(cudaEventRecord(event_, stream));
	}

	void synchronize() const {
		CHECK_CUDA(cudaEventSynchronize(event_));
	}

	static float elapsed(CudaEvent &start, CudaEvent &stop) {
		float ms = 0;
		CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, stop.event_));
		return ms;
	}
};

__device__ __forceinline__ uint64_t load_window_unaligned(const uint64_t *genome, const size_t s_bit_idx, const size_t max_bits) {
	const size_t word_idx = s_bit_idx / 64;
	const size_t bit_offset = s_bit_idx % 64;
	const uint64_t w1 = genome[word_idx];
	if (bit_offset == 0)
		return w1;
	uint64_t w2 = 0;
	if ((word_idx + 1) * 64 < max_bits)
		w2 = genome[word_idx + 1];
	return ((w1 >> bit_offset) | (w2 << (64 - bit_offset)));
}
