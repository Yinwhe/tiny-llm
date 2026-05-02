#include <torch/extension.h>

#include "tiny_llm_ext.h"

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
    m.doc() = "tiny-llm Torch reference extensions";

    m.def("load_library", &tiny_llm_ext_torch_ref::load_library, py::arg("device"), py::arg("path"));

    m.def("quantized_matmul",
          &tiny_llm_ext_torch_ref::quantized_matmul,
          py::arg("scales"),
          py::arg("zeros"),
          py::arg("group_size"),
          py::arg("bits"),
          py::arg("a"),
          py::arg("b"),
          py::arg("transpose_b") = false,
          R"(
        AWQ-style int4 quantized matmul.
      )");
}
