from qutip import *
import numpy as np
import matplotlib.pyplot as plt

#structure density_matrix = reservoir tensor three level system
#three level system defined as |e> |g> |0> basis
wr = 4 * 2 * np.pi
t = 1.1 * 2 * np.pi
delta = np.sqrt(wr ** 2 - 4 * t ** 2)
kappa = 0.04 * 2 * np.pi
kin = 0.038 * 2 * np.pi
K = 0.01 * 2 * np.pi
g0 = 0.1 * 2 * np.pi
gamma_r = 0.03 * 2 * np.pi
gamma_phi = 0.06 * 2 * np.pi
gamma = 2.5 * 2 * np.pi
gamma = 0.25 * 2 * np.pi
e, N = 1.602e-19, 30
g = g0 * 2 * t / wr * 2 * np.pi
def compute_lindblad(K):

    # Power range to sweep (test values)
    P_values = np.linspace(0, 1000, 30) #fw # From 1 to 10 in 5 steps
    #P_values_large = np.linspace(1,50,20)
    #P_values = np.hstack((P_values, P_values_large))
    # System setup (constant for all simulations)
    a = destroy(N)
    rho_up, rho_down = destroy(2), create(2)
    rho_z = rho_up * rho_down - rho_down * rho_up

    # 3-level operators
    def extend(op):
        a = np.block([[op.full(), np.zeros((2, 1))], [np.zeros((1, 3))]])
        #a[2,2] = 1
        return Qobj(a)

    def system_hamiltonian(w):
        Delta_c = wr - w
        Delta_q = np.sqrt(delta ** 2 + 4 * t ** 2) - w
        f = np.sqrt(kin * n_dot)
        print('f',f)
        print('g',g)
        print('Delta_q',Delta_q)
        print('cavity_photon number:', n_dot)
        #print((tensor(a, rho_up)))
        #print(tensor(a.dag(), rho_down))
        return (Delta_c * tensor(a.dag() * a, qeye(3)) + \
                (Delta_q * tensor(qeye(N), rho_z) / 2) + \
                K * (tensor(a.dag()*a*a.dag()*a, qeye(3))) + \
                g * (tensor(a, rho_up) + tensor(a.dag(), rho_down)) + \
                f * (tensor(a, qeye(3)) + tensor(a.dag(), qeye(3))))

    rho_up = extend(rho_up) # rho +
    print(rho_up)
    rho_down = extend(rho_down) # rho -
    rho_z = extend(rho_z)
    rho_0e = basis(3, 2) * basis(3, 0).dag()
    rho_g0 = basis(3, 1) * basis(3, 2).dag()
    #print(rho_up)
    # Collapse operators (constant for all simulations)
    c_ops = [
        np.sqrt(gamma) * tensor(qeye(N), rho_0e),
        np.sqrt(2 * gamma) * tensor(qeye(N), rho_g0),
        np.sqrt(gamma_r) * tensor(qeye(N), rho_down),
        np.sqrt(gamma_phi) * tensor(qeye(N), rho_z),
        np.sqrt(kappa) * tensor(a, qeye(3))
    ]

    # Common simulation parameters
    tlist = np.linspace(0, 30, 30)
    psi0 = tensor(fock(N, 0), fock(3, 2))
    observables = [
        tensor(qeye(N), basis(3, 0) * basis(3, 0).dag()),
        tensor(qeye(N), basis(3, 2) * basis(3, 2).dag())
    ]

    # Store results
    currents = []
    efficiencies = []
    cavity_photon_number = []
    saturation_current = []
    for P in P_values:
        # Update power-dependent parameters
        n_dot = P * 1e-15 / (1.055e-34 * wr * 1e18)  # photons/s # Simplified relationship for testing
        cavity_photon_number.append(n_dot)
        H = system_hamiltonian(wr)  # Rebuild Hamiltonian with new P

        # Run simulation
        output = mcsolve(H, psi0, tlist, c_ops, observables)

        gamma_l_in = gamma * (wr + delta) / (wr)
        gamma_r_in = gamma * (wr - delta) / (wr)
        gamma_r_out = gamma_l_in/2
        gamma_l_out = gamma_r_in/2

        current = e * (output.expect[0] * gamma_r_out -
                       output.expect[1] * gamma_r_in)*10**9
        #time_averaged_currents = np.trapezoid(current, tlist)/(tlist[-1]-tlist[0])
        #currents.append(time_averaged_currents)  # Average steady-state current
        print(current[-1])
        currents.append(current[-1]) #steady state current, roughtly after 15 ps

        efficiencies.append(current[-1]/(1.602*10**-19*n_dot*10**9))
        pf = gamma_l_in*gamma_r_out/(2*gamma**2)-gamma_r_in*gamma_l_out/(2*gamma**2)

        N_saturation = 1.602 * 10 ** -19 * 2 / 5 * gamma * pf *10**9
        saturation_current.append(N_saturation)

    return cavity_photon_number, currents, P_values, efficiencies,saturation_current

non_linearlity = [0.01, 0.05, 0.1, 0.5, 1]
non_linearlity = [0]
plt.figure(figsize=(16, 6))
for i in non_linearlity:
    cavity_photon_number, currents, P_values, efficiencies, saturation_current = compute_lindblad(i)
    # Plot results
    # Create a figure with two subplots side by side
    currents[0] = 0
    cavity_photon_number[0] = 0
    plt.subplot(1, 2, 1)
    plt.plot(cavity_photon_number, currents, 'o-')
    plt.plot(cavity_photon_number, saturation_current, 'o-')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Cavity photon number', fontsize=12)
    plt.ylabel('Time averaged current (A)', fontsize=12)
    plt.title('Numerical Simulation of Ids', fontsize=14)
    plt.grid(True, which="both", ls="--")

    # Second subplot (Efficiency vs Power)
    plt.subplot(1, 2, 2)
    plt.plot(P_values, efficiencies, 'o-', label=f'Kerr = {i}Ghz')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Power (fW)', fontsize=12)
    plt.ylabel('Electron-photon conversion efficiency', fontsize=12)
    plt.title('Conversion Efficiency', fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.legend(loc="upper right")

# Adjust layout and show
plt.tight_layout()
plt.show()
