// SPDX-License-Identifier: Apache-2.0
//
// SRI's suffix-tree design stores suffix contexts once and votes for the next
// token from all matching histories. This compact implementation keeps the
// same path-speculation behavior using an incrementally maintained
// token-context -> next-token-count index. It avoids scanning every cached
// completion for every proposal, which is essential for online use.

#include "suffix_tree.h"

#include <algorithm>
#include <cmath>
#include <utility>

size_t TokenKeyHash::operator()(const std::vector<int32_t>& key) const {
  size_t hash = 1469598103934665603ULL;
  for (const int32_t token : key) {
    hash ^= static_cast<uint32_t>(token);
    hash *= 1099511628211ULL;
  }
  return hash;
}

SuffixTree::SuffixTree(int32_t max_depth) : max_depth_(max_depth) {}

int32_t SuffixTree::num_seqs() const {
  return static_cast<int32_t>(sequences_.size());
}

void SuffixTree::add_transition(const std::vector<int32_t>& sequence,
                                int32_t next_index, int32_t delta) {
  const int32_t max_context = std::min<int32_t>(max_depth_, next_index);
  for (int32_t length = 1; length <= max_context; ++length) {
    std::vector<int32_t> context(sequence.begin() + next_index - length,
                                 sequence.begin() + next_index);
    auto transition = transitions_.find(context);
    if (delta > 0) {
      if (transition == transitions_.end()) {
        transition = transitions_.emplace(std::move(context), std::map<int32_t, int32_t>{}).first;
      }
      transition->second[sequence[next_index]] += delta;
      continue;
    }
    if (transition == transitions_.end()) {
      continue;
    }
    auto count = transition->second.find(sequence[next_index]);
    if (count == transition->second.end()) {
      continue;
    }
    count->second += delta;
    if (count->second <= 0) {
      transition->second.erase(count);
    }
    if (transition->second.empty()) {
      transitions_.erase(transition);
    }
  }
}

void SuffixTree::append(int32_t seq_id, int32_t token) {
  auto& sequence = sequences_[seq_id];
  sequence.push_back(token);
  const int32_t next_index = static_cast<int32_t>(sequence.size()) - 1;
  if (next_index > 0) {
    add_transition(sequence, next_index, 1);
  }
}

void SuffixTree::extend(int32_t seq_id, const std::vector<int32_t>& tokens) {
  for (const int32_t token : tokens) {
    append(seq_id, token);
  }
}

void SuffixTree::remove(int32_t seq_id) {
  const auto sequence = sequences_.find(seq_id);
  if (sequence == sequences_.end()) {
    return;
  }
  for (int32_t next_index = 1;
       next_index < static_cast<int32_t>(sequence->second.size()); ++next_index) {
    add_transition(sequence->second, next_index, -1);
  }
  sequences_.erase(sequence);
}

SuffixCandidate SuffixTree::speculate(const std::vector<int32_t>& pattern,
                                      int32_t max_spec_tokens,
                                      float max_spec_factor,
                                      float max_spec_offset,
                                      float min_token_prob) const {
  SuffixCandidate best;
  if (pattern.empty() || max_spec_tokens <= 0 || transitions_.empty()) {
    return best;
  }

  const int32_t begin = std::max<int32_t>(
      0, static_cast<int32_t>(pattern.size()) - max_depth_);
  for (int32_t start = begin; start < static_cast<int32_t>(pattern.size()); ++start) {
    std::vector<int32_t> context(pattern.begin() + start, pattern.end());
    const auto initial = transitions_.find(context);
    if (initial == transitions_.end()) {
      continue;
    }
    const int32_t match_len = static_cast<int32_t>(context.size());
    const int32_t allowed = std::clamp<int32_t>(
        static_cast<int32_t>(std::floor(match_len * max_spec_factor +
                                        max_spec_offset + 1e-6F)),
        0, max_spec_tokens);
    if (allowed == 0) {
      continue;
    }

    SuffixCandidate candidate;
    candidate.match_len = match_len;
    float cumulative_prob = 1.0F;
    for (int32_t step = 0; step < allowed; ++step) {
      const auto transition = transitions_.find(context);
      if (transition == transitions_.end() || transition->second.empty()) {
        break;
      }
      int32_t total = 0;
      for (const auto& item : transition->second) {
        total += item.second;
      }
      const auto choice = std::max_element(
          transition->second.begin(), transition->second.end(),
          [](const auto& left, const auto& right) {
            return left.second < right.second;
          });
      const float conditional = static_cast<float>(choice->second) / total;
      cumulative_prob *= conditional;
      if (cumulative_prob < min_token_prob) {
        break;
      }
      candidate.token_ids.push_back(choice->first);
      candidate.parents.push_back(
          static_cast<int32_t>(candidate.token_ids.size()) - 2);
      candidate.probs.push_back(cumulative_prob);
      candidate.score += cumulative_prob;
      context.push_back(choice->first);
      if (static_cast<int32_t>(context.size()) > max_depth_) {
        context.erase(context.begin());
      }
    }
    if (candidate.score > best.score) {
      best = std::move(candidate);
    }
  }
  return best;
}
