"""Example for simulating the optimal approximation ratio of a 1D Ising model
with and without control errors."""

from typing import Tuple, List, Sequence

import numpy
import pybobyqa
from matplotlib import pyplot as plt
import pdb
from qubit_ring import QubitRing


def optimal_angle_evolutionary(qubit_ring: QubitRing, sigma: float, alpha: float, p_val: int, batch_size: int=30, optimization_size: int=100, num_inits: int = 20
                               )-> Tuple[float, List[Tuple[float, float]]]:
    """Solves for the optimal approximation ratio at a given circuit depth (p) using evolutionary algorithm.

        Args:
            qubit_ring: Stores information of the Ising Hamiltonian. See
                implementation of the QubitRing class.
            p_val: Circuit depth (number of (\gamma, \beta) pairs).
            num_inits: How many restarts with random initial guesses.

        Returns:
            best_ratio: The best approximation ratio at circuit depth p.
            best_angles: The optimal angles that give the best approximation ratio.
        """
    energy_extrema, indices, e_list = qubit_ring.get_all_energies()

    def cost_function(angles_list: Sequence[float]) -> float:
        angles_list = [(angles_list[k], angles_list[k + 1]) for k in
                       numpy.array(range(p_val)) * 2]
        _, state_probs = qubit_ring.compute_wavefunction(angles_list)
        return -float(numpy.sum(e_list * state_probs))

    lower_bounds = numpy.array([0.0] * (2 * p_val))
    upper_bounds = numpy.array([1.0, 2.0] * p_val) * 16

    best_cost = None
    best_angles = None  # record the rotation angles gamma_1, beta_1, gamma_2, beta_2,

    def rescale_w(win, pmax, pmin):
        #win = win / numpy.absolute(win + 10e-12)
        wnew = numpy.absolute(win * (pmax - pmin) /2 )
        return wnew

    npop = p_val * batch_size + 20 #batch_size= 30

    num_inits =p_val * optimization_size # optimization_size = 100
    w = numpy.random.rand(2 * p_val)

    for i in range(num_inits):
        N = numpy.random.randn(npop, 2 * p_val)
        R = numpy.zeros(npop)
        for kk in range(npop):
            w_try = w + sigma * N[kk]
            w_try = rescale_w(w_try, upper_bounds, lower_bounds)
            R[kk] =  - cost_function(w_try)

        print("the average reward this batch:", numpy.average(R))
        print("the reward before update is ", cost_function(rescale_w(w, upper_bounds, lower_bounds)))

        A = (R - numpy.mean(R)) / (numpy.std(R) + 1 / 10 ** 12)
        w = w + alpha / (npop * sigma) * numpy.dot(N.T, A)
        wrescale = rescale_w(w, upper_bounds, lower_bounds)
        cost = cost_function(wrescale)
        #print(numpy.mean(R))
        # print("w is ", w)
        #
        # print("w-rescale current is ", wrescale)



        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_angles = wrescale
            print("best cost is ", best_cost)
            print("best angle is ", wrescale)


    best_angles = [(best_angles[i], best_angles[i + 1]) for i in
                   numpy.array(range(p_val)) * 2]

    e_max, e_min = energy_extrema['E_max'], energy_extrema['E_min']
    best_ratio = (-best_cost - e_min) / (e_max - e_min)

    return best_ratio, best_angles


def optimal_angle_solver(qubit_ring: QubitRing, p_val: int, num_inits: int = 20
                         ) -> Tuple[float, List[Tuple[float, float]]]:
    """Solves for the optimal approximation ratio at a given circuit depth (p).

    Args:
        qubit_ring: Stores information of the Ising Hamiltonian. See
            implementation of the QubitRing class.
        p_val: Circuit depth (number of (\gamma, \beta) pairs).
        num_inits: How many restarts with random initial guesses.

    Returns:
        best_ratio: The best approximation ratio at circuit depth p.
        best_angles: The optimal angles that give the best approximation ratio.
    """
    energy_extrema, indices, e_list = qubit_ring.get_all_energies()

    def cost_function(angles_list: Sequence[float]) -> float:
        angles_list = [(angles_list[k], angles_list[k + 1]) for k in
                       numpy.array(range(p_val)) * 2]
        _, state_probs = qubit_ring.compute_wavefunction(angles_list)
        return -float(numpy.sum(e_list * state_probs))

    lower_bounds = numpy.array([0.0] * (2 * p_val))
    upper_bounds = numpy.array([1.0, 2.0] * p_val) * 16

    best_cost = None
    best_angles = None

    for i in range(num_inits):
        guess = numpy.random.uniform(0, 4, p_val * 2)
        guess[0::2] = guess[0::2] / 2.0
        res = pybobyqa.solve(cost_function, guess,
                             bounds=(lower_bounds, upper_bounds),
                             maxfun=1000)
        cost = res.f
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_angles = res.x

    best_angles = [(best_angles[i], best_angles[i + 1]) for i in
                   numpy.array(range(p_val)) * 2]

    e_max, e_min = energy_extrema['E_max'], energy_extrema['E_min']
    best_ratio = (-best_cost - e_min) / (e_max - e_min)

    return best_ratio, best_angles

# Specify two qubit rings with and without control noise.
quiet_ring = QubitRing(8)
noisy_ring = QubitRing(8, noise_j=0.3)

approx_ratios = []
approx_ratios_with_noise = []

# Find and plot the maximum approximation ratios vs p (up to p = 3) for both
# cases.
p_vals = range(1, 4)
for p_val in p_vals:


    sigma = 0.2  # alpha = 0.85: 1.0: 0.749  0.3: 0.732;  0.2: 0.748; 0.1:  0.7431; 0.8: 0.74415; 1.8: 0.7248 ; 1.3:  0.719
    alpha = 0.85 # sigma=1, alpha =0.55: 0.7439; 0.85: 0.749; 0.755: 0.742; 0.705: 0.7487; 0.55: 0.7493; 0.25:  0.741; 1.85: 0.7446; 1.25: 0.7313
    ratio, _ = optimal_angle_evolutionary(quiet_ring, sigma, alpha, p_val, 100, 300) # 55, 70: 0.7135 with sigma=1, alpha =0.85
    print('Best approximation ratio for evolutionary QAOA at p = {} is {}, without control '
          'noise'.format(p_val, ratio))


#best so far: sigma = 1.0, alpha = 9.85, 30, 100
    sigma = 0.2 # 1.5: 0.6506, 0.8: 0.627
    alpha = 9.85  #sigma = 1.0, 9.95: 0.654; 10.05: 0.671;  9.65: 0.592; 9.85: 0.668; 9.35: 0.680;  12.35: 0.62; 10.35: 0.665; 8.35: 0.624;  1.35: 0.62941;  5.35: 0.631582604975; 20:  0.6287; 12: 0.627
    ratio_with_noise, _ = optimal_angle_evolutionary(noisy_ring, sigma, alpha, p_val, 150, 100) # 70, 50: 0.6029


    print('Best approximation ratio for evolutionary QAOA  at p = {} is {}, with control '
          'noise \n'.format(p_val, ratio_with_noise))
    approx_ratios.append(ratio)
    approx_ratios_with_noise.append(ratio_with_noise)


    ratio, _ = optimal_angle_solver(quiet_ring, p_val)
    ratio_with_noise, _ = optimal_angle_solver(noisy_ring, p_val)
    approx_ratios.append(ratio)
    approx_ratios_with_noise.append(ratio_with_noise)
    print('Best approximation ratio at p = {} is {}, without control '
          'noise'.format(p_val, ratio))
    print('Best approximation ratio at p = {} is {}, with control '
          'noise \n'.format(p_val, ratio_with_noise))

# fig = plt.figure()
# plt.plot(p_vals, approx_ratios, 'ro-', figure=fig,
#          label='Without control noise')
# plt.plot(p_vals, approx_ratios_with_noise, 'bo-',
#          figure=fig, label='With control noise')
# plt.xlabel('Circuit depth, p')
# plt.ylabel('Optimal approximation ratio')
# plt.legend()
# plt.show()
