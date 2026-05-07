#include <torch/extension.h>

#include "tiny_llm_ext.h"
#include "axpby.h"

namespace py = pybind11;

PYBIND11_MODULE(_ext, m) {
    m.doc() = "tiny-llm extensions for Torch";

    m.def("load_library", &tiny_llm_ext::load_library, py::arg("device"), py::arg("path"));

    m.def("axpby", &tiny_llm_ext::axpby, py::arg("x"), py::arg("y"), py::arg("alpha"), py::arg("beta"),
          R"(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``

        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (Tensor): Input tensor.
            y (Tensor): Input tensor.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            Tensor: ``alpha * x + beta * y``
      )");
}
