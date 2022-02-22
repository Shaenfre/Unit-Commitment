import numpy as np
import matplotlib.pyplot as plt
import math

fo = 50
del_t = 0.01
H = 8

fault_init_time = 0.1
fault_removal_time = 0.3

Pm = 1.5

Pmax1 = 3.0568
Pmax2 = 1.1038
Pmax3 = 2.48375

tf = 2

no_of_iter = int(tf // del_t) + 1
print(no_of_iter)

fault_init_iterno = int(fault_init_time // del_t)
fault_remvl_iterno = int(fault_removal_time // del_t)

print(fault_init_iterno)
print(fault_remvl_iterno)

delta = np.zeros(no_of_iter)
delta_w = np.zeros(no_of_iter)
t_arr = np.zeros(no_of_iter)

delta[0] = 29.3875 * math.pi / 180
delta_w[0] = 0

d_delta_dt_1 = 0
d_deltaw_dt_1 = math.pi * fo * (Pm - Pmax1 * math.sin(delta[0])) / H
print(d_deltaw_dt_1)
Pmax = Pmax1

for i in range(1, no_of_iter):
    if i < fault_init_iterno:
        delta[i] = delta[i - 1]
        t_arr[i] = del_t + t_arr[i - 1]
        continue
    elif i >= fault_init_iterno and i < fault_remvl_iterno:
        Pmax = Pmax2
    else:
        Pmax = Pmax3
    
    delta1_p = delta[i - 1] + d_delta_dt_1 * del_t
    del_w1_p = delta_w[i - 1] + d_deltaw_dt_1 * del_t

    d_delta_dt_2 = del_w1_p
    d_deltaw_dt_2 = math.pi * fo * (Pm - Pmax * math.sin(delta1_p)) / H

    delta[i] = delta[i - 1] + 0.5 * (d_delta_dt_1 + d_delta_dt_2) * del_t
    delta_w[i] = delta_w[i - 1] + 0.5 * (d_deltaw_dt_1 + d_deltaw_dt_2) * del_t

    d_delta_dt_1 = delta_w[i]
    d_deltaw_dt_1 = math.pi * fo * (Pm - Pmax * math.sin(delta[i])) / H

    t_arr[i] = del_t + t_arr[i - 1]

# print(delta)
# print(delta_w)

plt.plot(t_arr, delta)
plt.show()
