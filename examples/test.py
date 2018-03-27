from libmf_py import Problem
from time import perf_counter as pc
from itertools import count
import numpy as np

""" Obtain data: "Movielens" """
s_time = pc()
data = np.loadtxt('ml-latest-small/ratings.csv',
# data = np.loadtxt('ml-20m//ratings.csv',
                  dtype={'names': ('userId', 'movieId', 'rating'),
                         'formats': ('i4', 'i4', 'f4')},
                  skiprows=1,
                  usecols=(0,1,2),
                  delimiter=',')

print('n ratings: ', data.shape[0])
print('n users: ', np.unique(data['userId']).shape[0])
print('n items: ', np.unique(data['movieId']).shape[0])
print('time (read data): ', pc() - s_time)

"""" Preprocess data: map to [0,n-1] """
s_time = pc()
user_map, user_map_inv = {}, {}
item_map, item_map_inv = {}, {}
user_counter = count(0)
item_counter = count(0)

for u, v, r in data:
    if u not in user_map:
        new_u_id = next(user_counter)
        user_map[u] = new_u_id
        user_map_inv[new_u_id] = u
    if v not in item_map:
        new_v_id = next(item_counter)
        item_map[v] = new_v_id
        item_map_inv[new_v_id] = v

mapped_u = np.vectorize(user_map.__getitem__)(data['userId'])
mapped_v = np.vectorize(item_map.__getitem__)(data['movieId'])
print('time (id-mapping): ', pc() - s_time)

""" Train """
problem = Problem()
problem.set_ratings(mapped_u, mapped_v, data['rating'])
print('time (data-preparation): ', problem.get_data_preparation_time())
problem.set_param_int('nr_threads', 4)
problem.set_param_int('nr_iters', 50)

results = []

for reg in [0.01, 0.05, 0.1, 0.15]:
    for k in [5, 10, 20, 30]:
        print('REGULARIZATION L2: ', reg)
        print('K: ', k)
        problem.set_param_float('lambda_p', reg)
        problem.set_param_float('lambda_q', reg)
        problem.set_param_int('k', k)

        rmse = problem.train_cv(5)
        print(rmse)
        results.append((reg, k, rmse))
        print('time (cv): ', problem.get_cv_time())

print('Results')
sorted_results = sorted(results, key=lambda x: x[2])
for i in sorted_results:
    print(i)

# FINAL FIT
problem.set_param_int('k', sorted_results[0][1])
problem.set_param_float('lambda_p', sorted_results[0][0])
problem.set_param_float('lambda_q', sorted_results[0][0])
problem.set_param_int('nr_iters', 50)
problem.train()
