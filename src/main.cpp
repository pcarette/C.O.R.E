#include <algorithm>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "config.hpp"
#include "encoder.hpp"
#include "gpu_utils.hpp"
#include "kernel.hpp"
#include "loader.hpp"

struct WriteJob {
	std::string data;
	bool done = false;
};

std::deque<WriteJob> q;
std::mutex mtx;
std::condition_variable cv;

void writer_thread(const std::string &path) {
	std::ofstream f(path);
	if (!f.is_open()) {
		std::cerr << "[ERROR] Cannot open output file: " << path << std::endl;
		return;
	}
	f << "LOCATION\tCANDIDATE_SEQ\tMISMATCHES\tSTRAND\n";
	while (true) {
		std::unique_lock lk(mtx);
		cv.wait(lk, [] { return !q.empty(); });
		auto [data, done] = q.front();
		q.pop_front();
		lk.unlock();
		if (done)
			break;
		f << data;
	}
}

void submit_write(const std::string &s) {
	if (s.empty())
		return;
	std::lock_guard lk(mtx);
	q.push_back({s, false});
	cv.notify_one();
}

// ReSharper disable once CppDFAConstantFunctionResult
std::string decode_32bp(const uint64_t b) {
	std::string s(32, 'A');
	for (int i = 0; i < 32; ++i) {
		if (const uint8_t v = (b >> (i * 2)) & 3; v == 1)
			s[i] = 'C';
		else if (v == 2)
			s[i] = 'G';
		else if (v == 3)
			s[i] = 'T';
	}
	return s;
}

std::string get_rc(const std::string &s) {
	std::string r = s;
	std::ranges::reverse(r);
	for (char &c : r) {
		if (const char u = std::toupper(static_cast<unsigned char>(c)); u == 'A')
			c = 'T';
		else if (u == 'T')
			c = 'A';
		else if (u == 'C')
			c = 'G';
		else if (u == 'G')
			c = 'C';
		else if (u == 'R')
			c = 'Y';
		else if (u == 'Y')
			c = 'R';
		else if (u == 'N')
			c = 'N';
	}
	return r;
}

uint64_t encode_dna_string(const std::string &seq) {
	uint64_t p = 0;
	for (size_t i = 0; i < std::min(seq.size(), static_cast<size_t>(32)); ++i) {
		uint64_t v = 0;
		if (const char b = std::toupper(seq[i]); b == 'C')
			v = 1;
		else if (b == 'G')
			v = 2;
		else if (b == 'T')
			v = 3;
		p |= (v << (i * 2));
	}
	return p;
}

EnzymeConfig generate_reverse_config(const EnzymeConfig &fwd) {
	EnzymeConfig rev = fwd;
	uint64_t new_pat = 0;
	uint64_t new_mask = 0;
	for (int i = 0; i < fwd.pam_len; ++i) {
		const uint64_t shift_in = i * 2;
		const uint64_t base_val = (fwd.pam_pattern >> shift_in) & 3;
		const uint64_t mask_val = (fwd.pam_care_mask >> shift_in) & 3;
		uint64_t rc_base = 3 - base_val;
		const uint64_t rc_mask = mask_val;
		const int shift_out = (fwd.pam_len - 1 - i) * 2;
		if (rc_mask == 0)
			rc_base = 0;
		new_pat |= (rc_base << shift_out);
		new_mask |= (rc_mask << shift_out);
	}

	rev.pam_pattern = new_pat;
	rev.pam_care_mask = new_mask;
	if (fwd.pam_offset_correction == 0) {
		rev.pam_offset_correction = -fwd.target_len;
	} else {
		rev.pam_offset_correction = 0;
	}

	return rev;
}

std::vector<uint8_t> load_epigenome(const std::string &path, const size_t expected_size) {
	std::cout << "[LOADER] Loading Epigenome atlas: " << path << std::endl;
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		std::cerr << "[ERROR] Epigenome file not found!" << std::endl;
		exit(1);
	}

	const size_t size = f.tellg();
	if (std::abs(static_cast<long long>(size) - static_cast<long long>(expected_size)) > 1024 * 1024) {
		std::cerr << "[WARN] Size mismatch! Epi: " << size << " vs Genome: " << expected_size << std::endl;
		std::cerr << "[WARN] Ensure chrom.sizes used for .epi matches the .fa file!" << std::endl;
	}

	f.seekg(0, std::ios::beg);
	std::vector<uint8_t> buffer(size);
	if (!f.read(reinterpret_cast<char *>(buffer.data()), size)) {
		std::cerr << "[ERROR] Failed to read epigenome data." << std::endl;
		exit(1);
	}
	std::cout << "[LOADER] Epigenome loaded (" << size / 1024 / 1024 << " MB). Ready for GPU." << std::endl;
	return buffer;
}

int main(const int argc, char **argv) {
	if (argc < 5) {
		std::cerr << "Usage: ./core_runner <genome.fa> <epi.bin> <query> <config.json>" << std::endl;
		return 1;
	}

	const EnzymeConfig cfg = ConfigLoader::load_from_json(argv[4]);
	const GenomeLoader l(argv[1]);
	std::vector<ChromosomeRange> idx;
	std::vector<bool> nm;
	double st;
	const auto clean = sanitize_genome(argv[1], l.data(), l.size(), idx, nm, st);
	const auto epi_buffer = load_epigenome(argv[2], clean.size());
	const size_t nb = (clean.size() + 31) / 32;
	PinnedHostBuffer<uint64_t> gb(nb);
	encode_sequence_avx2(clean.data(), clean.size(), gb.data());
	std::thread wt(writer_thread, "results.tsv");

	auto run_pass = [&](const std::string &q_seq, const std::string &lbl, const EnzymeConfig &current_cfg) {
		const uint64_t p = encode_dna_string(q_seq);
		uint64_t m = 0;
		for (size_t i = 0; i < q_seq.size(); ++i)
			m |= 3ULL << (i * 2);
		auto res = launch_pipelined_search(gb.data(), epi_buffer.data(), nb, current_cfg, p, m);
		std::string buf;
		buf.reserve(4 * 1024 * 1024);
		for (size_t i = 0; i < res.count; ++i) {
			uint32_t pos = res.matches[i];
			const uint32_t b_idx = pos / 32;
			if (b_idx >= nb)
				continue;
			std::string ctx = decode_32bp(gb[b_idx]);
			if (b_idx + 1 < nb)
				ctx += decode_32bp(gb[b_idx + 1]);
			const size_t local_off = pos % 32;
			if (local_off + q_seq.size() > ctx.size())
				continue;
			std::string cand = ctx.substr(local_off, q_seq.size());
			int mm = 0;
			for (size_t k = 0; k < q_seq.size(); ++k)
				if (cand[k] != q_seq[k])
					mm++;

			if (mm <= current_cfg.max_mismatches) {
				auto it = std::upper_bound(idx.begin(), idx.end(), pos, [](const size_t ps, const ChromosomeRange &r) {
					return ps < r.start_idx;
				});
				if (it != idx.begin()) {
					const auto prev = std::prev(it);
					std::string context_str = "Closed";
					if (pos < epi_buffer.size()) {
						if (epi_buffer[pos] & 1)
							context_str = "ATAC_OPEN";
					}
					buf += prev->name + ":" + std::to_string(pos - prev->start_idx) + "\t" + cand + "\t" + std::to_string(mm) +
						   "\t" + lbl + "\t" + context_str + "\n";
				}

				if (buf.size() > 2 * 1024 * 1024) {
					submit_write(buf);
					buf.clear();
				}
			}
		}
		submit_write(buf);
		std::cout << "[RESULT] " << lbl << " Pass Complete. GPU Hits: " << res.count << std::endl;
		free_search_results(res);
	};

	std::cout << "[CORE] Processing Forward Strand..." << std::endl;
	run_pass(argv[3], "(+)", cfg);
	std::cout << "[CORE] Processing Reverse Strand..." << std::endl;
	const EnzymeConfig rev_cfg = generate_reverse_config(cfg);
	run_pass(get_rc(argv[3]), "(-)", rev_cfg);
	{
		std::lock_guard lk(mtx);
		q.push_back({"", true});
		cv.notify_one();
	}
	wt.join();

	std::cout << "[CORE] Finalized. Check results.tsv." << std::endl;
	return 0;
}
