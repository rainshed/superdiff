using LinearAlgebra
using ForwardDiff
using Plots
using LaTeXStrings

# ==========================================
# 1. Model definition and physical functions
# ==========================================

struct PottsParams
    z::Float64          # Coordination number
    J::Matrix{Float64}  # 2x2 interaction matrix
    h::Vector{Float64}  # 2D external field vector [h1, h2]
end

# Compute the single-site effective field B_eff (for states 1 and 2)
function get_effective_field(m::Vector{T}, p::PottsParams) where T
    # B = h + z * J * m
    return p.h + p.z * (p.J * m)
end

# Compute free energy density f(m1, m2)
function free_energy_density(m::Vector{T}, p::PottsParams, beta::Real) where T
    # 1. Interaction correction term
    # E_corr = (z/2) * m' * J * m
    E_corr = (p.z / 2) * dot(m, p.J * m)

    # 2. Compute the single-site partition function
    # Energies: E1 = -B1, E2 = -B2, E3 = 0 (Fixed reference state as requested)
    B_eff = get_effective_field(m, p)
    
    # Numerical stability trick: Shift energies by a constant value
    # Note: For ForwardDiff, we treat the shift as a constant value to avoid 
    # derivative discontinuities at the switch points of minimum().
    val_B = ForwardDiff.value.(B_eff)
    min_E_val = minimum([-val_B[1], -val_B[2], 0.0])
    
    # E_i = -B_eff (for 1,2) and 0 (for 3)
    # Argument for exp: -beta * (E_i - min_E_val)
    # This simplifies to: beta * (B_eff + min_E_val) roughly
    
    term1 = exp(beta * (B_eff[1] + min_E_val))
    term2 = exp(beta * (B_eff[2] + min_E_val))
    term3 = exp(beta * (0.0       + min_E_val))
    
    Z_shifted = term1 + term2 + term3
    log_Z = -beta * min_E_val + log(Z_shifted)
    
    return E_corr - (1/beta) * log_Z
end

# Compute observables for a fixed point solution
function compute_observables(m_fixed::Vector{Float64}, p::PottsParams, beta::Float64)
    B_eff = get_effective_field(m_fixed, p)
    energies = [-B_eff[1], -B_eff[2], 0.0]
    
    # Calculate probabilities explicitly
    min_E = minimum(energies)
    weights = exp.(-beta .* (energies .- min_E))
    Z = sum(weights)
    probs = weights ./ Z
    
    # Entropy
    S = -sum(x -> x > 1e-12 ? x * log(x) : 0.0, probs)
    
    # Free Energy
    F = free_energy_density(m_fixed, p, beta)
    
    # Internal Energy U = F + T S
    T_val = 1.0 / beta
    U = F + T_val * S
    
    return (m=m_fixed, p3=probs[3], S=S, F=F, U=U)
end

# ==========================================
# 2. Numerical solver
# ==========================================

function solve_potts(p::PottsParams, T_range::Vector{Float64})
    # Define the type for results to ensure Type Stability
    ResultType = NamedTuple{(:m, :p3, :S, :F, :U, :status, :T), 
                            Tuple{Vector{Float64}, Float64, Float64, Float64, Float64, Symbol, Float64}}
    results = ResultType[]

    # Helper for Gradient and Hessian
    function gradient_F(m, beta)
        ForwardDiff.gradient(x -> free_energy_density(x, p, beta), m)
    end

    function hessian_F(m, beta)
        ForwardDiff.hessian(x -> free_energy_density(x, p, beta), m)
    end

    function self_consistency_map(m, beta)
        B = get_effective_field(m, p)
        bmax = max(B[1], B[2], 0.0)
        w1 = exp(beta * (B[1] - bmax))
        w2 = exp(beta * (B[2] - bmax))
        w3 = exp(beta * (0.0  - bmax))
        Z = w1 + w2 + w3
        return [w1 / Z, w2 / Z]
    end

    function in_simplex(m; tol=1e-10)
        return m[1] >= -tol && m[2] >= -tol && sum(m) <= 1.0 + tol
    end

    function residual_G(m, beta)
        return m .- self_consistency_map(m, beta)
    end

    function jacobian_G(m, beta)
        pvec = self_consistency_map(m, beta)
        c11 = pvec[1] * (1.0 - pvec[1])
        c22 = pvec[2] * (1.0 - pvec[2])
        c12 = -pvec[1] * pvec[2]
        C = [c11 c12; c12 c22]
        return Matrix{Float64}(I, 2, 2) - beta * p.z * (C * p.J)
    end

    println("Starting simulation over $(length(T_range)) temperature points...")

    for T_val in T_range
        beta = 1.0 / T_val
        
        # 2.1 Generate initial guesses inside the simplex (m1+m2 <= 1, m_i >= 0)
        initial_guesses = Vector{Vector{Float64}}()
        steps = 8
        # Grid over the simplex, including boundaries.
        for i in 0:steps, j in 0:(steps - i)
            x, y = i / steps, j / steps
            push!(initial_guesses, [x, y])
        end
        # Add a few interior points to help Newton avoid boundary singularities.
        push!(initial_guesses, [0.90, 0.05], [0.05, 0.90], [0.05, 0.05])

        found_roots = Vector{Float64}[]
        
        # 2.2 Newton iteration
        for m0 in initial_guesses
            m_curr = copy(m0)
            converged = false
            current_F = free_energy_density(m_curr, p, beta)

            for iter in 1:30
                grad = gradient_F(m_curr, beta)
                
                if norm(grad) < 1e-8
                    converged = true
                    break
                end

                hess = hessian_F(m_curr, beta)
                
                # Newton step: delta = H^-1 * g
                delta = try
                    hess \ grad
                catch
                    break # Singular Hessian
                end
                
                # Backtracking Line Search
                # CRITICAL FIX: Check if Free Energy decreases, not Gradient norm
                step_size = 1.0
                accepted = false
                while step_size > 1e-4
                    m_next = m_curr - step_size * delta
                    
                    # Check constraints (stay inside simplex)
                    if m_next[1] >= -1e-10 && m_next[2] >= -1e-10 && sum(m_next) <= 1.0 + 1e-10
                         next_F = free_energy_density(m_next, p, beta)
                         # Armijo-like condition or simple decrease
                         if next_F < current_F
                             m_curr = m_next
                             current_F = next_F
                             accepted = true
                             break
                         end
                    end
                    step_size *= 0.5
                end
                
                if !accepted
                    # If Newton direction fails, try simple Gradient Descent step
                    m_next = m_curr - 0.01 * grad
                    if m_next[1] >= -1e-10 && m_next[2] >= -1e-10 && sum(m_next) <= 1.0 + 1e-10
                        m_curr = m_next
                        current_F = free_energy_density(m_next, p, beta)
                    else
                        break
                    end
                end
            end
            
            if !converged
                # Fallback: damped fixed-point iteration on self-consistency equations.
                m_fp = copy(m_curr)
                for _ in 1:2000
                    rhs = self_consistency_map(m_fp, beta)
                    if norm(rhs - m_fp) < 1e-10
                        m_curr = rhs
                        converged = true
                        break
                    end
                    m_next = 0.9 .* m_fp .+ 0.1 .* rhs
                    if norm(m_next - m_fp) < 1e-12
                        m_curr = m_next
                        converged = true
                        break
                    end
                    m_fp = m_next
                end
            end

            if converged
                push!(found_roots, m_curr)
            end
        end

        # 2.3 Stationary-point search via self-consistency residual Newton.
        # This branch is needed to recover saddle/max solutions (unstable).
        for m0 in initial_guesses
            m_curr = copy(m0)
            converged = false

            for _ in 1:60
                g = residual_G(m_curr, beta)
                gnorm = norm(g)
                if gnorm < 1e-10
                    converged = true
                    break
                end

                Jg = try
                    jacobian_G(m_curr, beta)
                catch
                    break
                end

                delta = try
                    Jg \ g
                catch
                    break
                end

                step_size = 1.0
                accepted = false
                while step_size > 1e-6
                    m_next = m_curr - step_size * delta
                    if in_simplex(m_next)
                        g_next = residual_G(m_next, beta)
                        if norm(g_next) < gnorm
                            m_curr = m_next
                            accepted = true
                            break
                        end
                    end
                    step_size *= 0.5
                end

                if !accepted
                    break
                end
            end

            if converged
                push!(found_roots, m_curr)
            end
        end

        # 2.4 Remove duplicates
        unique_roots = Vector{Float64}[]
        for r in found_roots
            is_new = true
            for u in unique_roots
                if norm(r - u) < 1e-4
                    is_new = false
                    break
                end
            end
            if is_new
                push!(unique_roots, r)
            end
        end

        # 2.5 Classification
        if isempty(unique_roots)
            # Guaranteed fallback from the simplex center.
            m_fp = [1.0 / 3.0, 1.0 / 3.0]
            for _ in 1:3000
                rhs = self_consistency_map(m_fp, beta)
                if norm(rhs - m_fp) < 1e-10
                    m_fp = rhs
                    break
                end
                m_next = 0.9 .* m_fp .+ 0.1 .* rhs
                if norm(m_next - m_fp) < 1e-12
                    m_fp = m_next
                    break
                end
                m_fp = m_next
            end
            if norm(self_consistency_map(m_fp, beta) - m_fp) < 1e-7
                push!(unique_roots, m_fp)
            else
                println("Warning: No roots found at T=$T_val")
                continue
            end
        end

        # Calculate F for all roots to find global minimum
        temp_storage = []
        min_F_global = Inf
        
        for m_sol in unique_roots
            obs = compute_observables(m_sol, p, beta)
            if obs.F < min_F_global
                min_F_global = obs.F
            end
            
            hess = hessian_F(m_sol, beta)
            evals = eigvals(hess)
            # Semi-definite minima can appear at symmetry-protected marginal points.
            is_stable_local = all(e -> e > -1e-8, evals)
            
            status = is_stable_local ? :metastable : :unstable
            push!(temp_storage, (obs..., status=status, T=T_val))
        end
        
        # Mark the global stable state
        for d in temp_storage
            final_status = d.status
            if d.status == :metastable && abs(d.F - min_F_global) < 1e-6
                final_status = :stable
            end
            
            # Construct NamedTuple manually to match ResultType
            entry = (m=d.m, p3=d.p3, S=d.S, F=d.F, U=d.U, status=final_status, T=d.T)
            push!(results, entry)
        end
    end
    
    return results
end

# ==========================================
# 3. Run simulation
# ==========================================

# Use the parameter set you provided (overwriting the first one)
# This matrix has eigenvalues 0 and -2, combined with the positive quadratic form
# in free energy, it drives phase transitions.
# 在 Code A 中修改参数以匹配 Code B
params = PottsParams(
    2.0, 
    [1.0 -1.0; -1.0 1.0],  # 注意这里的符号
    [-0.95, -0.95]         # 注意这里的符号
)

T_range = collect(0.05:0.02:3.0) # Slightly coarser step for speed, adjust as needed
data = solve_potts(params, T_range)

# ==========================================
# 4. Plotting
# ==========================================

# Helper to extract arrays
function get_series(data_vec, status_sym)
    mask = [d.status == status_sym for d in data_vec]
    subset = data_vec[mask]
    return (
        [d.U for d in subset],
        [d.S for d in subset]
    )
end

u_s, s_s = get_series(data, :stable)
u_m, s_m = get_series(data, :metastable)
u_u, s_u = get_series(data, :unstable)

p = plot()
scatter!(p,u_u, s_u, label="Unstable", markersize=2, color=:red, alpha=0.5, legend=:topleft)
scatter!(p, u_m, s_m, label="Metastable", markersize=3, color=:blue)
scatter!(p, u_s, s_s, label="Stable", markersize=4, color=:black)

ylabel!(p, L"Entropy $S$")
xlabel!(p, L"Internal Energy $U$")
title!(p, "E-S Diagram for Anisotropic Potts Model")

# Explicitly display the plot
display(p)
