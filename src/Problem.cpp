#include <chrono>
#include "Problem.hpp"
#include "libmf/mf.h"

// TODO debug
#include <pybind11/pybind11.h>
namespace py = pybind11;

using namespace std::chrono;
typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;

// Problem definition
void Problem::set_ratings(std::vector<int>& u,
                          std::vector<int>& v,
                          std::vector<float>& r)
{
    auto t0 = Time::now();

    long long N = u.size();
    this->mf_problem_train.nnz = N;

    mf::mf_node *R = new mf::mf_node[this->mf_problem_train.nnz];

    for(long long i=0; i<N; ++i)
    {
        int u_ = u[i];
        int v_ = v[i];
        float r_ = r[i];

        if( (u_+1) > (this->mf_problem_train.m))
            this->mf_problem_train.m = (u_+1);

        if( (v_+1) > (this->mf_problem_train.n))
            this->mf_problem_train.n = (v_+1);

        mf::mf_node node;
        node.u = u_;
        node.v = v_;
        node.r = r_;
        R[i] = node;
    }

    this->mf_problem_train.R = R;

    auto t1 = Time::now();
    ms d = std::chrono::duration_cast<ms>(t1 - t0);
    this->data_preparation_time = d.count();
}

// Parameters
void Problem::set_param_int(std::string param, int value)
{
    if(param == "fun")
        this->mf_param.fun = value;
    else if(param == "k")
        this->mf_param.k = value;
    else if(param == "nr_threads")
        this->mf_param.nr_threads = value;
    else if(param == "nr_bins")
        this->mf_param.nr_bins = value;
    else if(param == "nr_iters")
        this->mf_param.nr_iters = value;
}

void Problem::set_param_float(std::string param, float value)
{
    if(param == "lambda_p1")
        this->mf_param.lambda_p1 = value;
    else if(param == "lambda_p2")
        this->mf_param.lambda_p2 = value;
    else if(param == "lambda_q1")
        this->mf_param.lambda_q1 = value;
    else if(param == "lambda_q2")
        this->mf_param.lambda_q2 = value;
    else if(param == "eta")
        this->mf_param.eta = value;
}

void Problem::set_param_bool(std::string param, bool value)
{
    if(param == "do_nmf")
        this->mf_param.do_nmf = value;
    else if(param == "quiet")
        this->mf_param.quiet = value;
    else if(param == "copy_data")
        this->mf_param.copy_data = value;
}

// Training
void Problem::train()
{
    auto t0 = Time::now();

    this->mf_model = mf::mf_train(&this->mf_problem_train, this->mf_param);

    auto t1 = Time::now();
    ms d = std::chrono::duration_cast<ms>(t1 - t0);
    this->training_time = d.count();
}

double Problem::train_cv(int nfolds)
{
    auto t0 = Time::now();

    double rmse = mf::mf_cross_validation(&this->mf_problem_train, nfolds,
                                          this->mf_param);

    auto t1 = Time::now();
    ms d = std::chrono::duration_cast<ms>(t1 - t0);
    this->cv_time = d.count();
    return rmse;
}

// Output
float Problem::get_data_preparation_time()
{
    return float(this->data_preparation_time) / 1000.0;  // secs
}

float Problem::get_training_time()
{
    return float(this->data_preparation_time) / 1000.0;  // secs
}

float Problem::get_cv_time()
{
    return float(this->cv_time) / 1000.0;  // secs
}

float Problem::predict(int u, int v)
{
    return mf::mf_predict(this->mf_model, u, v);
}

py::array_t<float> Problem::get_P()
{
    int m = this->mf_model->m;
    int k = this->mf_model->k;

    std::vector<float> P(m*k);

    for(int i=0; i<m; ++i)
        for(int j=0; j<k; ++j)
        {
          int off = i * k + j;
          P[off] = *(this->mf_model->P + off);
        }

    return py::array(py::buffer_info(
                      P.data(),
                      sizeof(float),
                      py::format_descriptor<float>::format(),
                      2,
                      {m, k},
                      {sizeof(float) * k,
                       sizeof(float)}));
}

py::array_t<float> Problem::get_Q()
{
    int n = this->mf_model->n;
    int k = this->mf_model->k;

    std::vector<float> P(k*n);

    for(int i=0; i<n; ++i)
        for(int j=0; j<k; ++j)
        {
          int off = i * k + j;
          P[off] = *(this->mf_model->Q + off);
        }

    return py::array(py::buffer_info(
                      P.data(),
                      sizeof(float),
                      py::format_descriptor<float>::format(),
                      2,
                      {n, k},
                      {sizeof(float) * k,
                       sizeof(float)}));
}
