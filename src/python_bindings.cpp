#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(core_engine, m) {
	m.doc() = "C.O.R.E Engine";
	py::class_<GenomeLoader>(m, "GenomeLoader")
		.def(py::init<const std::string &>())
		.def("size", &GenomeLoader::size)
		.def("encode", &GenomeLoader::encode, py::return_value_policy::move);
}
