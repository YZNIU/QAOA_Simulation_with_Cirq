import cirq
import numpy as np
from matplotlib import pyplot as plt
from qaoa_functions import qaoa_to_p, output_state_prob_energy, initial_guess_sweep_bobyqa, parameter_sweep_ground_prob, parameter_sweep, angles_trotter_steps


def fixed_instance_1():
    h = np.array([[ 0.094,  0.5213,  0.992],
       [ 0.5998, -0.2527,  0.2481],
       [ 0.9243, -0.4946,  0.4319]])
    jr = np.array([[-0.69,  0.56],
       [ 0.09,  0.93],
       [-0.98,  0.50]])
    jc = np.array([[0.111, 0.96, 0.082],
       [0.92, 0.64, 0.325]])
    return (h, jr, jc)

length = 3
qubits = [cirq.GridQubit(i, j) for i in range(length) for j in
          range(length)]

h, jr, jc = fixed_instance_1()
# print(h)
# print(jr)
# print(jc)

x_min = 0.0
x_max = 1.0
z_min = 0.0
z_max = 4.0

# best_angles, best_ground_prob, best_cost, result_gnd_prob, result_cost_func = initial_guess_sweep_bobyqa(h, jr, jc, qubits, p = 1, num_trials = 30000)

# parameter_sweep_ground_prob(x_min, x_max, z_min, z_max, h, jr, jc, qubits, num_points=10)
# parameter_sweep(x_min, x_max, z_min, z_max, h, jr, jc, qubits, num_points=10, num_trials=3000)
# minimum_strings, maximum_strings, minimum, maximum = energy_extrema(length, h, jr, jc)
# print(minimum_strings)
# print(maximum_strings)
# print(minimum)
# print(maximum)
# best_angles = [0.80508571, 0.20362419, 0.87883107, 0.38059593, 0.92986663, 0.44366917]
# best_angles_p30 = [0.75627212, 0.06643678, 0.77414841, 0.13653974, 0.80661742,
#        0.16238658, 0.81808738, 0.19476923, 0.82804397, 0.22159164,
#        0.83292317, 0.24467722, 0.83822807, 0.26495925, 0.85315192,
#        0.28524411, 0.86642848, 0.30275963, 0.86983192, 0.31629966,
#        0.87372073, 0.31956511, 0.87872797, 0.32986414, 0.88084987,
#        0.33997162, 0.88795388, 0.35190463, 0.89479287, 0.36088928,
#        0.89486997, 0.36633857, 0.90325528, 0.37670665, 0.90459847,
#        0.3821479 , 0.91285654, 0.38257757, 0.92041257, 0.37992637,
#        0.92752407, 0.39668008, 0.93217917, 0.40348479, 0.94325762,
#        0.40376649, 0.95399383, 0.41188407, 0.96247752, 0.41803497,
#        0.96797653, 0.42452269, 0.9726096 , 0.43454641, 0.97622434,
#        0.42975456, 0.97953791, 0.43514055, 0.9917838 , 0.41323836]

# probs, ratios, steps, new_angles = qaoa_to_p(h, jr, jc, best_angles, 2, qubits, num_trials = 1000)
#
# plt.plot(steps, probs, 'b-o')
# plt.xlabel("p")
# plt.ylabel("Ground State Prob")
# plt.show()

steps = 30
time = 30
local_strength = 0.5

new_angles = angles_trotter_steps(steps, time, local_strength)
# p = np.linspace(1,steps,steps)
# plt.plot(p, new_angles[0::2], 'r-o')
# plt.xlabel("p")
# plt.ylabel(r"$\gamma / \pi$")
# plt.show()

time_min = 0.2
time_max = 40.0
local_strength_min = -1.0
local_strength_max = 8.0
steps = 3

def parameter_sweep_trotter(time_min, time_max, local_strength_min, local_strength_max, h, jr, jc, qubits, steps, num_points=20):
    results_map_gnd = np.zeros((num_points, num_points))
    results_map_exp = np.zeros((num_points, num_points))
    time_values = np.linspace(time_min, time_max, num_points)
    strength_values = np.linspace(local_strength_min, local_strength_max, num_points)
    for i, time in np.ndenumerate(time_values):
        for j, strength in np.ndenumerate(strength_values):
            new_angles = angles_trotter_steps(steps, time, strength)
            gnd_prob, expected_energy = output_state_prob_energy(new_angles, h,
                                                                 jr, jc, qubits,
                                                                 p=steps, sorted = False)
            results_map_gnd[j, i] = gnd_prob
            results_map_exp[j, i] = (14.576 - expected_energy)/(14.576 - 6.957)

    print(np.amax(results_map_gnd))
    print(np.amax(results_map_exp))

    plt.figure()
    plt.imshow(results_map_gnd, origin='lower', extent=[time_min, time_max, local_strength_min, local_strength_max],
               aspect= (time_max - time_min) / (local_strength_max - local_strength_min))
    plt.xlabel(r"$T$")
    plt.ylabel(r"$A$")
    plt.colorbar()
    plt.title("Ground State Probability")

    plt.figure()
    plt.imshow(results_map_exp, origin='lower', extent=[time_min, time_max, local_strength_min, local_strength_max],
               aspect= (time_max - time_min) / (local_strength_max - local_strength_min))
    plt.xlabel(r"$T$")
    plt.ylabel(r"$A$")
    plt.colorbar()
    plt.title("Cost function 1")

    plt.show()


parameter_sweep_trotter(time_min, time_max, local_strength_min, local_strength_max, h, jr, jc, qubits, steps, num_points = 10)

# gnd_prob, expected_energy = output_state_prob_energy(new_angles, h, jr, jc, qubits, p = steps)
# print(gnd_prob, expected_energy)