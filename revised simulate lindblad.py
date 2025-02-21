from qutip import *
import numpy as np
import matplotlib.pyplot as plt

#structure density_matrix = reservoir tensor three level system
#three level system defined as |e> |g> |0> basis

# Parameters (testing constants)
wr = 100
delta = 100
t = 3
g0 = kin = kappa = 2
gamma = gamma_r = gamma_phi = 0.01

wr = 4
t = 1.1
delta = np.sqrt(wr**2 - 4*t**2)
kappa = 0.04
kin = 0.038
g0 = 0.1
gamma_r = 0.03
gamma_phi = 0.06
gamma = 2.5
e, N = 1.602e-19, 14
g = g0 * 2 * t / wr

# Power range to sweep (test values)
P_values = np.linspace(0, 0.2, 10) # From 1 to 10 in 5 steps
#P_values_large = np.linspace(1,50,20)
#P_values = np.hstack((P_values, P_values_large))
# System setup (constant for all simulations)
a = destroy(N)
rho_up, rho_down = destroy(2), create(2)
rho_z = rho_up * rho_down - rho_down * rho_up

print(rho_z)



# 3-level operators
def extend(op):
    a = np.block([[op.full(), np.zeros((2, 1))], [np.zeros((1, 3))]])
    a[2,2] = 1
    return Qobj(a)

def system_hamiltonian(w, K=0.0):
    Delta_c = wr - w
    Delta_q = np.sqrt(delta ** 2 + 4 * t ** 2) - w
    f = np.sqrt(kin * n_dot)
    print('f',f)
    print('g',g)
    print('Delta_q',Delta_q)
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
tlist = np.linspace(0, 30, 300)
psi0 = tensor(fock(N, 0), fock(3, 1))
observables = [
    tensor(qeye(N), basis(3, 0) * basis(3, 0).dag()),
    tensor(qeye(N), basis(3, 2) * basis(3, 2).dag())
]

# Store results
currents = []

for P in P_values:
    # Update power-dependent parameters
    n_dot = P  # Simplified relationship for testing
    H = system_hamiltonian(wr)  # Rebuild Hamiltonian with new P

    # Run simulation
    output = mesolve(H, psi0, tlist, c_ops, observables)

    gamma_r_out = gamma * (wr + delta) / (wr)
    gamma_r_in = gamma * (wr - delta) / (wr)
    current = e * (output.expect[0] * gamma_r_out -
                   output.expect[1] * gamma_r_in)*10**9*2*np.pi

    currents.append(np.trapezoid(current, tlist)/(tlist[-1]-tlist[0]))  # Average steady-state current
    #plt.figure(figsize=(10, 6))
    #plt.plot(tlist, current, 'o-')
    #plt.xlabel('Power (P)')
    #plt.ylabel('Time averaged current (A)')
    #plt.title('Current vs Input Power cavity frequency approximately equal to g, k, gamma')
    #plt.grid(True)
    #plt.show()
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(P_values, currents, 'o-')
plt.xlabel('Power (P)')
plt.ylabel('Time averaged current (A)')
plt.title('Current vs Input Power cavity frequency approximately equal to g, k, gamma')
plt.grid(True)
plt.show()

#low drive current:
up = 1.602*10**-19*16*g**2*kin*gamma
#low_drive_current =
