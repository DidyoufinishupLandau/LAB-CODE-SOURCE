using QuantumToolbox
using PyPlot
using LinearAlgebra

wr = 3.8 * 2pi
t = 1.1 *  2pi
delta = sqrt(wr ^ 2 - 4 * t ^ 2)
kappa = 0.029 *  2pi
kin = (0.023) *  2pi #to feedline
K = 2 *  2pi
g0 = 0.10 *  2pi
gamma_r = 0.03 *  2pi
gamma_phi = 0.06 *  2pi
gamma = 2.5
dephasing = 0.004 * 2* 2pi
e, N = 1.602e-19, 300



g = g0 * 2 * t / wr

P_values = [range(0.001, 0.01, 5);range(0.01, 0.1, 5); range(0.1, 1, 5); range(1, 100, 20)] #fw # From 1 to 10 in 5 steps
#P_values_large = np.linspace(1,50,20)
#P_values = np.hstack((P_values, P_values_large))
# System setup (constant for all simulations)
a = destroy(N)
rho_up, rho_down =  destroy(2), create(2)
rho_z = rho_up * rho_down - rho_down * rho_up

function extentions(op)
    # Convert operator to regular matrix first
    op_matrix = Matrix(op)
    extended = zeros(ComplexF64, 3, 3)
    extended[1:size(op_matrix,1), 1:size(op_matrix,2)] .= op_matrix

    # Convert to GPU array and wrap in quantum object
    return Qobj(extended)
end


rho_up = extentions(rho_up) # rho +
rho_down = extentions(rho_down) # rho -
rho_z = extentions(rho_z)
rho_0e = basis(3, 2) * dag(basis(3, 0))
rho_g0 = basis(3, 1) * dag(basis(3, 2))

function system_hamiltonian(w, n_dot)
    Delta_c = wr - w
    Delta_q = sqrt(delta ^ 2 + 4 * t ^ 2) - w
    f = sqrt(kin * n_dot)
    #print((tensor(a, rho_up)))
    #print(tensor(a.dag(), rho_down))
    h0 = Delta_c * tensor(dag(a) * a, qeye(3))
    h1 = Delta_q * tensor(qeye(N), rho_z) / 2
    kerr_h = K/2 * tensor(dag(a) * dag(a) * a * a, qeye(3))
    coupling_h = g * (tensor(a, rho_up) + tensor(dag(a), rho_down))
    drive_h = f * (tensor(a, qeye(3)) + tensor(dag(a), qeye(3)))

    return h0 + h1 + kerr_h + coupling_h + drive_h

end


observables = [
    tensor(qeye(N), basis(3, 0) * dag(basis(3, 0))),
    tensor(qeye(N), basis(3, 2) * dag(basis(3, 2))),
    tensor(dag(a) * a, qeye(3))

]
println(basis(3, 0) * dag(basis(3, 0)))

#print(rho_up)
# Collapse operators (constant for all simulations)
c_ops = [
    sqrt(gamma) * tensor(qeye(N), rho_0e),
    sqrt(2 * gamma) * tensor(qeye(N), rho_g0),
    sqrt(gamma_r) * tensor(qeye(N), rho_down),
    sqrt(gamma_phi) * tensor(qeye(N), rho_z),
    sqrt(kappa) * tensor(a, qeye(3)),
    sqrt(dephasing) * tensor(dag(a)*a, qeye(3))
]

#construct the projection operator for the usage of the
currents = []
efficiencies = []
cavity_photon_number = []
saturation_current = []
number_of_photon_fit = []


for P in P_values
    # Update power-dependent parameters
    n_dot = P * 1e-15 / (1.055e-34 * wr * 1e18)
    push!(cavity_photon_number, n_dot)
    H = system_hamiltonian(wr, n_dot)  # Rebuild Hamiltonian with new P
    #Run simulation
    output = steadystate(H, c_ops; solver = SteadyStateLinearSolver())
    #output = steadystate(H, c_ops)
    expect_e = expect(output, observables[1])
    expect_0 = expect(output,observables[2])
    expect_photon = expect(output, observables[3])

    gamma_l_in = gamma * (wr + delta) / (wr)
    gamma_r_in = gamma * (wr - delta) / (wr)
    gamma_r_out = gamma_l_in/2
    gamma_l_out = gamma_r_in/2

    current = e * (expect_e * gamma_r_out -
                   expect_0 * gamma_r_in)*10^9

    push!(currents, current) #steady state current, roughtly after 30 ps
    push!(number_of_photon_fit, expect_photon)
    push!(efficiencies, current/(1.602*10^-19*n_dot*10^9))
    pf = gamma_l_in*gamma_r_out/(2*gamma^2)-gamma_r_in*gamma_l_out/(2*gamma^2)
    N_saturation = 1.602 * 10 ^ -19 * 2 / 5 * gamma * pf * 10^9
    push!(saturation_current, N_saturation)
end

figure(figsize=(12, 6))

# Subplot 1: Current vs Cavity Photon Number
subplot(1, 2, 1)
plot(P_values, currents, "o-", color="blue")
xlabel("Power (fW", fontsize=12)
ylabel("Current (nA)", fontsize=12)
title("Steady-State Current")
grid(true, linestyle="--", alpha=0.7)

# Subplot 2: Efficiency vs Power (log-log scale)
subplot(1, 2, 2)
loglog(P_values, efficiencies, "s-", color="red", markersize=6)
xlabel("Power (fW)", fontsize=12)
ylabel("Efficiency", fontsize=12)
title("Conversion Efficiency")
grid(true, which="both", linestyle="--", alpha=0.7)

figure()
plot(P_values, number_of_photon_fit, "o-", label="number of photon in cavity")
xlabel("Power (fW)", fontsize=12)
ylabel("photon number (nA)", fontsize=12)
title("Current vs Power")
legend()
grid(true)
# Add some spacing between subplots

tight_layout(pad=3.0)

# Optional: Plot saturation current comparison
figure()
plot(P_values, currents, "o-", label="Calculated Current")
plot(P_values, saturation_current, "--", label="Saturation Limit")
xlabel("Power (fW)", fontsize=12)
ylabel("Current (nA)", fontsize=12)
title("Current vs Power")
legend()
grid(true)

# Show all plots
show()
