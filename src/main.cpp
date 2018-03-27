#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Problem.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libmf_py, m)
{
  m.doc() = "libmf_py";

  py::class_<Problem>(m, "Problem")
  .def(py::init<>())
  .def("set_ratings", &Problem::set_ratings, "set_ratings")
  .def("set_param_int", &Problem::set_param_int, "set_param_int")
  .def("set_param_float", &Problem::set_param_float, "set_param_float")
  .def("set_param_bool", &Problem::set_param_bool, "set_param_bool")
  .def("train", &Problem::train, "train")
  .def("train_cv", &Problem::train_cv, "train_cv")
  .def("get_data_preparation_time", &Problem::get_data_preparation_time, "get_data_preparation_time")
  .def("get_training_time", &Problem::get_training_time, "get_training_time")
  .def("get_cv_time", &Problem::get_cv_time, "get_cv_time")
  .def("predict", &Problem::predict, "predict")
  .def("get_P", &Problem::get_P, "get_P")
  .def("get_Q", &Problem::get_Q, "get_Q")
  ;

}
