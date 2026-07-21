// SPDX-License-Identifier: Apache-2.0
// SRI-style suffix-path store for the optional SGLang forest backend.

#pragma once

#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

struct TokenKeyHash {
  size_t operator()(const std::vector<int32_t>& key) const;
};

struct SuffixCandidate {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> parents;
  std::vector<float> probs;
  float score = 0.0F;
  int32_t match_len = 0;
};

class SuffixTree {
 public:
  explicit SuffixTree(int32_t max_depth);

  int32_t num_seqs() const;
  void append(int32_t seq_id, int32_t token);
  void extend(int32_t seq_id, const std::vector<int32_t>& tokens);
  void remove(int32_t seq_id);
  SuffixCandidate speculate(const std::vector<int32_t>& pattern,
                            int32_t max_spec_tokens,
                            float max_spec_factor,
                            float max_spec_offset,
                            float min_token_prob) const;

 private:
  int32_t max_depth_;
  std::unordered_map<int32_t, std::vector<int32_t>> sequences_;
  std::unordered_map<std::vector<int32_t>, std::map<int32_t, int32_t>,
                     TokenKeyHash>
      transitions_;

  void add_transition(const std::vector<int32_t>& sequence, int32_t next_index,
                      int32_t delta);
};
