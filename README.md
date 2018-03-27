# libmf_py
This is a basic python-wrapper for [libmf](https://www.csie.ntu.edu.tw/~cjlin/libmf/), an open-source tool for approximating an incomplete matrix using the product of two matrices in latent space, which is shipped.

The wrapping is done using [pybind11](https://github.com/pybind/pybind11).

Only tested with python3.

## What's wrapped
- Model-definition: ```set_ratings```
- Parameter-selection: ```set_param_int```, ```set_param_float```, ```set_param_bool```
- Training: ```train```, ```train_cv```
- Prediction: ```predict```
- Factorization-matrices: ```get_P```, ```get_Q```

## What's not wrapped
- training with validation set
- disk-based training
- loss-calculations
- load/save model
- read txt-based problem

## Remarks
No error-checking is done!

Compiler-configuration is hard-coded within ```setup.py```:

    os.environ['CFLAGS'] = '-O3 -Wall -pthread -march=native -mavx'
    os.environ['DFLAG'] = '-DUSEAVX'

## Usage
See ```/examples``` folder (Movielens-dataset ```ml-latest-small``` is shipped).

```python test_2.py``` outputs:


    n ratings:  100004
    n users:  671
    n items:  9066
    time (read data):  1.0741573290142696
    time (id-mapping):  0.7628628929960541
    time (data-preparation):  0.0
    iter      tr_rmse          obj
       0       1.3283   2.4933e+05
       1       0.9067   1.5543e+05
       2       0.8770   1.4940e+05
       3       0.8639   1.4662e+05
       4       0.8551   1.4464e+05
       5       0.8497   1.4348e+05
       6       0.8452   1.4255e+05
       7       0.8417   1.4152e+05
       8       0.8386   1.4081e+05
       9       0.8366   1.4049e+05
      10       0.8344   1.3985e+05
      11       0.8325   1.3954e+05
      12       0.8308   1.3930e+05
      13       0.8291   1.3885e+05
      14       0.8277   1.3862e+05
      15       0.8265   1.3844e+05
      16       0.8248   1.3822e+05
      17       0.8234   1.3796e+05
      18       0.8213   1.3768e+05
      19       0.8205   1.3746e+05
    Prediction-subset: libmf
    [3.46397352 3.50003219 3.94744539 4.2361002  4.42462969 1.83410966
     3.73325968 0.91650927 2.60829639 3.39821815]
    Prediction-subset: python on P*Q.T
    [3.4639733 3.5000322 3.9474456 4.2361    4.42463   1.8341098 3.73326
     0.9165092 2.6082964 3.3982182]


## Alternative wrappers
[python-libmf](https://github.com/PorkShoulderHolder/python-libmf)

- ctypes-based
- similar functionality
- more error-checking
