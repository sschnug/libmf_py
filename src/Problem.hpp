#ifndef PROBLEM_HPP
#define PROBLEM_HPP
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "libmf/mf.h"

namespace py = pybind11;

class Problem
{
    // Internal timings
    long data_preparation_time = -1;
    long training_time = -1;
    long cv_time = -1;

    // Internal data
    mf::mf_parameter mf_param = mf::mf_get_default_param();
    mf::mf_problem mf_problem_train;
    mf::mf_model* mf_model;

public:
    Problem()
    {
        mf_problem_train = {};
    };

    // Problem definition
    void set_ratings(std::vector<int>& u,
                     std::vector<int>& v,
                     std::vector<float>& r);

    // Parameters
    // TODO: any chance of pybind11 + boost::any / std::any (C++17)?
    void set_param_int(std::string param, int value);
    void set_param_float(std::string param, float value);
    void set_param_bool(std::string param, bool value);

    // Training
    void train();
    double train_cv(int nfolds);

    // Output
    float get_data_preparation_time();
    float get_training_time();
    float get_cv_time();

    float predict(int u, int v);

    py::array_t<float> get_P();
    py::array_t<float> get_Q();

};
#endif
