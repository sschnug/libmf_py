from libmf_py import Problem
from time import perf_counter as pc
from itertools import count
import numpy as np

""" Obtain data: "Movielens" """
s_time = pc()
data = np.loadtxt('ml-latest-small/ratings.csv',
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

""" Split data """
rand_perm = np.random.permutation(len(mapped_u))
u_train = mapped_u[rand_perm[:99000]]
v_train = mapped_v[rand_perm[:99000]]
r_train = data['rating'][rand_perm[:99000]]

u_test = mapped_u[rand_perm[99000:]]
v_test = mapped_v[rand_perm[99000:]]
r_test = data['rating'][rand_perm[99000:]]

""" Train """
problem = Problem()
problem.set_ratings(u_train, v_train, r_train)
print('time (data-preparation): ', problem.get_data_preparation_time())
problem.set_param_int('nr_threads', 4)
problem.set_param_int('nr_iters', 20)
problem.train()
P = problem.get_P()
Q = problem.get_Q()
pred = P.dot(Q.T)

""" Predict test using libmf """
prediction_native = []
for i in range(len(u_test)):
    r_pred = problem.predict(u_test[i], v_test[i])
    error = abs(r_test[i] - r_pred)
    prediction_native.append(r_pred)

""" Predict using obtained P,Q -> full pred-matrix already computed (inefficient) """
prediction_py = []
for i in range(len(u_test)):
    r_pred = pred[u_test[i], v_test[i]]
    error = abs(r_test[i] - r_pred)
    prediction_py.append(r_pred)

print('Prediction-subset: libmf')
print(np.array(prediction_native)[:10])
print('Prediction-subset: python on P*Q.T')
print(np.array(prediction_py)[:10])
