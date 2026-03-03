Here is a comprehensive note on the Mean-Field Theory analysis for the general 3-State Potts model with nearest-neighbor interactions, formulated using the minimal 5-parameter set.

---

# Mean-Field Theory for the Generalized 3-State Potts Model

## 1. Model Definition

We consider the most general form of a 3-State Potts model with nearest-neighbor interactions. To eliminate redundancies caused by the constraint $\sum_{\alpha=1}^3 P^\alpha = 1$, we define the energy relative to state 3 (the "background" state).

**System Parameters (5 Independent Parameters):**

*   **Coupling Matrix $\mathbf{J}$** ($2 \times 2$ symmetric): Describes interaction energies between states 1 and 2.
    *   $J_{11}$: Interaction strength for $1-1$ pair.
    *   $J_{22}$: Interaction strength for $2-2$ pair.
    *   $J_{12}$: Interaction strength for $1-2$ pair.
*   **External Field $\mathbf{h}$** ($2 \times 1$ vector):
    *   $h_1, h_2$: Chemical potential favoring state 1 and 2 respectively (relative to state 3).

**Hamiltonian:**
$$
H = -\sum_{\langle i,j \rangle} \begin{pmatrix} P_i^1 & P_i^2 \end{pmatrix} 
\begin{pmatrix} J_{11} & J_{12} \\ J_{12} & J_{22} \end{pmatrix} 
\begin{pmatrix} P_j^1 \\ P_j^2 \end{pmatrix} 
- \sum_i \left( h_1 P_i^1 + h_2 P_i^2 \right)
$$

Where $P_i^\alpha$ is the projection operator for state $\alpha$ at site $i$.

---

## 2. Mean-Field Derivation

### 2.1 The Ansatz

We assume the density matrix factorizes: $\rho_{total} = \bigotimes_i \rho_i$.
We define the order parameters as the thermal expectations of the projection operators:
$$ \mathbf{m} = \begin{pmatrix} m_1 \\ m_2 \end{pmatrix} = \begin{pmatrix} \langle P^1 \rangle \\ \langle P^2 \rangle \end{pmatrix} $$
*Constraint:* $m_1, m_2 \ge 0$ and $m_1 + m_2 \le 1$. The population of state 3 is implicitly $m_3 = 1 - m_1 - m_2$.

### 2.2 Effective Hamiltonian

Using the fluctuation expansion $P_i^\alpha P_j^\beta \approx m_\alpha P_j^\beta + m_\beta P_i^\alpha - m_\alpha m_\beta$, we derive the single-site effective Hamiltonian $H_{eff}$.

The effective field acting on state $\alpha$ ($\alpha \in \{1,2\}$) is:
$$ \mathbf{B}_{eff} = \mathbf{h} + z \mathbf{J} \mathbf{m} $$
$$
\begin{cases}
B_1^{eff} = h_1 + z(J_{11}m_1 + J_{12}m_2) \\
B_2^{eff} = h_2 + z(J_{12}m_1 + J_{22}m_2)
\end{cases}
$$
*$z$ is the coordination number of the lattice.*

The single-site effective Hamiltonian is diagonal in the basis $\{|1\rangle, |2\rangle, |3\rangle\}$:
$$
H_{eff} = \text{diag}(-B_1^{eff}, -B_2^{eff}, 0)
$$
*(Note: State 3 has zero energy in this reference frame).*

### 2.3 Free Energy Density

The mean-field free energy per site, $f(\mathbf{m})$, is constructed from the internal energy correction and the single-site partition function.

$$
f(m_1, m_2) = \frac{z}{2} \left( J_{11}m_1^2 + J_{22}m_2^2 + 2J_{12}m_1 m_2 \right) - k_B T \ln \mathcal{Z}_{1}
$$

Where the single-site partition function $\mathcal{Z}_1$ is:
$$ \mathcal{Z}_1 = e^{\beta B_1^{eff}} + e^{\beta B_2^{eff}} + 1 $$

### 2.4 Self-Consistency Equations

Minimizing free energy with respect to $\mathbf{m}$ ($\nabla f = 0$) yields:
$$
m_1 = \frac{e^{\beta B_1^{eff}}}{\mathcal{Z}_1}, \quad m_2 = \frac{e^{\beta B_2^{eff}}}{\mathcal{Z}_1}
$$

---

## 3. Numerical Workflow

To find all stable, metastable, and unstable solutions, we solve the vector root-finding problem in the 2D simplex.

### Step 1: Initialization

Define the inputs: Temperature $T$, coordination number $z$, coupling matrix $\mathbf{J}$, and field $\mathbf{h}$.

### Step 2: Discretize the Search Space

Since the solution space is constrained ($m_1+m_2 \le 1$), we generate a grid of initial guesses $\mathbf{m}^{(0)}$ strictly within the triangle:

*   Loop $m_1$ from $0$ to $1$.
*   Loop $m_2$ from $0$ to $1 - m_1$.

### Step 3: Newton-Raphson Solver

For each initial guess, solve for the root of the residual function $\mathbf{G}(\mathbf{m}) = \mathbf{m} - \langle \mathbf{P} \rangle = 0$.

**Jacobian Calculation ($2 \times 2$):**
$$ \mathcal{J}_{ij} = \delta_{ij} - \frac{\partial \langle P^i \rangle}{\partial m_j} $$
Using $\beta = 1/k_B T$:
$$ \frac{\partial \langle P^i \rangle}{\partial m_j} = \beta z \sum_{k=1}^2 \left( \langle P^i P^k \rangle_c J_{kj} \right) $$
Where the connected correlation is $\langle P^i P^k \rangle_c = \delta_{ik}m_i - m_i m_k$.

### Step 4: Stability Classification (Hessian Analysis)

For every unique solution $\mathbf{m}^*$, compute the Hessian of the Free Energy density $\mathcal{H}_{ij} = \frac{\partial^2 f}{\partial m_i \partial m_j}$.

Calculate eigenvalues $\lambda_1, \lambda_2$ of $\mathcal{H}$:

1.  **Stable (Global Min):** $\lambda_{1,2} > 0$ and $f(\mathbf{m}^*)$ is minimal among all solutions.
2.  **Metastable (Local Min):** $\lambda_{1,2} > 0$ but $f(\mathbf{m}^*)$ is not global minimal.
3.  **Unstable (Saddle/Max):** At least one $\lambda < 0$.

---

## 4. Algorithm Implementation (Julia)

```julia
using LinearAlgebra
using ForwardDiff
using Plots
using LaTeXStrings

# ==========================================
# 1. Model definition and physical functions
# ==========================================

struct PottsParams
    z::Float64          # Coordination number
    J::Matrix{Float64}  # 2x2 interaction matrix [[J11, J12], [J12, J22]]
    h::Vector{Float64}  # 2D external field vector [h1, h2]
end

# Compute the single-site effective field B_eff
function get_effective_field(m::Vector{T}, p::PottsParams) where T
    # B = h + z * J * m
    # m is [m1, m2]
    return p.h + p.z * (p.J * m)
end

# Compute free energy density f(m1, m2)
# Input m is a vector [m1, m2]
function free_energy_density(m::Vector{T}, p::PottsParams, beta::Real) where T
    m1, m2 = m[1], m[2]
    m3 = 1.0 - m1 - m2
    
    # 1. Interaction correction term (Mean-Field Correction)
    # E_corr = (z/2) * (J11*m1^2 + J22*m2^2 + 2*J12*m1*m2)
    # Here we write it in matrix form
    E_corr = (p.z / 2) * dot(m, p.J * m)

    # 2. Compute the single-site partition function Z_1
    B_eff = get_effective_field(m, p)
    
    # Energy levels: E_1 = -B1, E_2 = -B2, E_3 = 0
    # To avoid numerical overflow, use the LogSumExp trick
    energies = [-B_eff[1], -B_eff[2], 0.0]
    min_E = minimum(energies)
    
    # Z = sum(exp(-beta * (E_i - min_E))) * exp(-beta * min_E)
    # lnZ = -beta * min_E + ln(sum(...))
    args = -beta .* (energies .- min_E)
    log_sum = log(sum(exp.(a) for a in args))
    log_Z = -beta * min_E + log_sum
    
    # F = E_corr - T * lnZ
    return E_corr - (1/beta) * log_Z
end

# Compute single-site thermodynamic quantities (entropy, internal energy, order parameter expectation)
function compute_observables(m_fixed::Vector{Float64}, p::PottsParams, beta::Float64)
    # For a given m (assumed to be a fixed point), compute the true distribution p_i
    B_eff = get_effective_field(m_fixed, p)
    energies = [-B_eff[1], -B_eff[2], 0.0]
    
    # Boltzmann weights
    weights = exp.(-beta .* energies)
    Z = sum(weights)
    probs = weights ./ Z  # [p1, p2, p3]
    
    # 1. Von Neumann entropy S = -sum p ln p
    # In mean-field, spatial entanglement is zero, but this represents local mixedness
    S = -sum(x -> x > 1e-10 ? x * log(x) : 0.0, probs)
    
    # 2. Free energy
    F = free_energy_density(m_fixed, p, beta)
    
    # 3. Internal energy U = F + T S
    T_val = 1.0 / beta
    U = F + T_val * S
    
    return (m=m_fixed, p3=probs[3], S=S, F=F, U=U)
end

# ==========================================
# 2. Numerical solver
# ==========================================

function solve_potts(p::PottsParams, T_range::Vector{Float64})
    results = []

    # Define residual function G(m) = gradient(FreeEnergy)
    # Stable points satisfy dF/dm = 0
    function gradient_F(m, beta)
        ForwardDiff.gradient(x -> free_energy_density(x, p, beta), m)
    end

    # Define Hessian for stability analysis
    function hessian_F(m, beta)
        ForwardDiff.hessian(x -> free_energy_density(x, p, beta), m)
    end

    println("Starting simulation over $(length(T_range)) temperature points...")

    for T_val in T_range
        beta = 1.0 / T_val
        
        # 2.1 Generate multiple initial guesses (Grid search in simplex)
        initial_guesses = Vector{Vector{Float64}}()
        steps = 7
        for i in 0:steps
            for j in 0:(steps-i)
                x = 0.01 + 0.98 * (i/steps)
                y = 0.01 + 0.98 * (j/steps)
                if x+y < 0.99
                    push!(initial_guesses, [x, y])
                end
            end
        end
        
        # Additionally add guesses near the corners to capture pure states
        push!(initial_guesses, [0.98, 0.01], [0.01, 0.98], [0.01, 0.01])

        found_roots = []
        
        # 2.2 Newton iteration
        for m0 in initial_guesses
            m_curr = copy(m0)
            converged = false
            for iter in 1:20
                grad = gradient_F(m_curr, beta)
                hess = hessian_F(m_curr, beta)
                
                if norm(grad) < 1e-8
                    converged = true
                    break
                end
                
                # Newton step: m_new = m - H^-1 * g
                try
                    delta = hess \ grad
                    # Simple backtracking line search to prevent leaving the simplex
                    step_size = 1.0
                    while step_size > 1e-4
                        m_next = m_curr - step_size * delta
                        if m_next[1] > 0 && m_next[2] > 0 && sum(m_next) < 1
                             # Check whether residual decreases
                             if norm(gradient_F(m_next, beta)) < norm(grad)
                                 m_curr = m_next
                                 break
                             end
                        end
                        step_size *= 0.5
                    end
                catch
                    break # Hessian is singular
                end
            end
            
            if converged
                push!(found_roots, m_curr)
            end
        end

        # 2.3 Remove duplicates
        unique_roots = []
        for r in found_roots
            is_duplicate = false
            for u in unique_roots
                if norm(r - u) < 1e-4
                    is_duplicate = true; break;
                end
            end
            if !is_duplicate
                push!(unique_roots, r)
            end
        end

        # 2.4 Classification and data recording
        
        # First find the global minimum of free energy
        min_F = Inf
        temp_data = []
        
        for m_sol in unique_roots
            obs = compute_observables(m_sol, p, beta)
            hess = hessian_F(m_sol, beta)
            evals = eigvals(hess)
            
            # Classification
            # Stable: Hessian positive definite (all eigenvalues > 0)
            # Unstable: at least one eigenvalue < 0
            is_stable_local = all(e -> e > 1e-6, evals)
            
            status = :unstable
            if is_stable_local
                status = :metastable # provisional
            end
            
            if obs.F < min_F
                min_F = obs.F
            end
            
            push!(temp_data, (obs..., status=status, T=T_val))
        end
        
        # Correct global stable state
        for i in 1:length(temp_data)
            d = temp_data[i]
            if d.status == :metastable && abs(d.F - min_F) < 1e-6
                temp_data[i] = merge(d, (status=:stable,))
            end
            push!(results, temp_data[i])
        end
    end
    
    return results
end

# ==========================================
# 3. Run simulation and plotting
# ==========================================

# Set parameters: standard 3-State Potts (ferromagnetic)
# J11=J22=1, J12=0 (favor same color). h=0.
params = PottsParams(
    2.0,                 # Coordination number z
    [1.0 0.2; 0.2 0.95], # J matrix
    [-0.05, 0.]          # External field
)

# Temperature range
T_range = collect(0.02:0.001:3)

# Run solver
data = solve_potts(params, T_range)

# ==========================================
# 4. Data processing and visualization
# ==========================================

# Extract data
Ts = [d.T for d in data]
Us = [d.U for d in data]
Fs = [d.F for d in data]
Ss = [d.S for d in data]
# Order parameter (magnetization) M = sqrt(m1^2 + m2^2 + m3^2)
# Or simply take m1 (if symmetry breaking selects state 1)
# Here we plot m1
M1s = [d.m[1] for d in data]
Statuses = [d.status for d in data]

# Separate data by stability for plotting
function get_series(status_sym)
    mask = Statuses .== status_sym
    return Ts[mask], Us[mask], Fs[mask], Ss[mask], M1s[mask]
end

t_s, u_s, f_s, s_s, m_s = get_series(:stable)
t_m, u_m, f_m, s_m, m_m = get_series(:metastable)
t_u, u_u, f_u, s_u, m_u = get_series(:unstable)

# --- Plot 1: Physical quantities vs Temperature (T) ---

p1 = plot(layout=(3,1), size=(600, 800), legend=:right)

# 1. Order parameter m1 vs T
scatter!(p1[1], t_u, m_u, label="Unstable", markersize=2, color=:red, alpha=0.5)
scatter!(p1[1], t_m, m_m, label="Metastable", markersize=3, color=:blue, marker=:square)
scatter!(p1[1], t_s, m_s, label="Stable", markersize=4, color=:black)
ylabel!(p1[1], L"Order Parameter $m_1$")

# 2. Entropy S vs T
scatter!(p1[2], t_u, s_u, label="", markersize=2, color=:red, alpha=0.5)
scatter!(p1[2], t_m, s_m, label="", markersize=3, color=:blue, marker=:square)
scatter!(p1[2], t_s, s_s, label="", markersize=4, color=:black)
ylabel!(p1[2], L"Entropy $S$")

# 3. Free energy F vs T
scatter!(p1[3], t_u, f_u, label="", markersize=2, color=:red, alpha=0.5)
scatter!(p1[3], t_m, f_m, label="", markersize=3, color=:blue, marker=:square)
scatter!(p1[3], t_s, f_s, label="", markersize=4, color=:black)
ylabel!(p1[3], L"Free Energy $F$")
xlabel!(p1[3], L"Temperature $T$")

display(p1)

# --- Plot 2: Physical quantities vs Internal Energy (U) ---
# Often used to observe phase transition "back-bending" or energy gaps

p2 = plot(layout=(2,1), size=(600, 600), legend=:topright)

# 1. Entropy S vs Internal Energy U (related to caloric curve)
# Theoretically dS/dU = 1/T, which is a fundamental thermodynamic relation
scatter!(p2[1], u_u, s_u, label="Unstable", markersize=2, color=:red, alpha=0.5)
scatter!(p2[1], u_m, s_m, label="Metastable", markersize=3, color=:blue)
scatter!(p2[1], u_s, s_s, label="Stable", markersize=4, color=:black)
ylabel!(p2[1], L"Entropy $S$")
xlabel!(p2[1], L"Energy $E$")

# 2. Free energy F vs Internal Energy E
scatter!(p2[2], u_u, f_u, label="Unstable", markersize=2, color=:red, alpha=0.5)
scatter!(p2[2], u_m, f_m, label="Metastable", markersize=3, color=:blue)
scatter!(p2[2], u_s, f_s, label="Stable", markersize=4, color=:black)
ylabel!(p2[2], L"Free Energy $F$")
xlabel!(p2[2], L"Energy $E$")

display(p2)

```

