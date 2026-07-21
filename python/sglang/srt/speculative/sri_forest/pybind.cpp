// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "suffix_tree.h"

namespace py = pybind11;

PYBIND11_MODULE(_sglang_sri_suffix_tree, module) {
  py::class_<SuffixCandidate>(module, "SuffixCandidate")
      .def_readonly("token_ids", &SuffixCandidate::token_ids)
      .def_readonly("parents", &SuffixCandidate::parents)
      .def_readonly("probs", &SuffixCandidate::probs)
      .def_readonly("score", &SuffixCandidate::score)
      .def_readonly("match_len", &SuffixCandidate::match_len);

  py::class_<SuffixTree>(module, "SuffixTree")
      .def(py::init<int32_t>())
      .def("num_seqs", &SuffixTree::num_seqs)
      .def("append", &SuffixTree::append)
      .def("extend", &SuffixTree::extend)
      .def("remove", &SuffixTree::remove)
      .def("speculate", &SuffixTree::speculate);
}
