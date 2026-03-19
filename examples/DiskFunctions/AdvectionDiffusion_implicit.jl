using ProjectedSphericalHarmonics
using FFTW
using LinearAlgebra
using KrylovKit
using CairoMakie

# ─── Implicit advection-diffusion on the unit disk ─────────────────────────────
#
# PDE:  ∂ₜu + 2(a ∂ζu + ā ∂ζ̄u) = Δu    on D,  u|∂D = 0
#
# Backward Euler (fully implicit in both diffusion and advection):
#
#   (α I − Δ + 2a ∂ζ + 2ā ∂ζ̄) u^{n+1} = α u^n,   α = 1/dt
#
# Solved via GMRES in PSH coefficient space with the decomposition
# u = Δ⁻¹μ − h_corr(μ), where h_corr is the harmonic that corrects the
# aliasing boundary error of the finite-truncation Δ⁻¹μ.
#
# The GMRES operator for density μ is:
#
#   Ã_α(μ) = −μ + α v + 2a ∂ζv + 2ā ∂ζ̄v,   v = Δ⁻¹μ − h_corr(μ)
#
# Key advantages over a naive grid-based scheme:
#
#   1. From step n > 0 the RHS α u^n is assembled as α * û_tri directly from
#      the stored TriangularCoeffArray — no nodal-to-coefficient re-transform.
#
#   2. Derivatives ∂ζv, ∂ζ̄v inside each GMRES iteration are computed exactly
#      by the sparse operators ∂Ĝ∂ζ / ∂Ĝ∂ζ̄ (no finite differences).
#
#   3. After each step, ∂ζu^n and ∂ζ̄u^n are available exactly from the stored
#      density μ^n via the same sparse operators — no ill-conditioned grid
#      differentiation needed.
#
#   4. The GMRES is warm-started with the density μ^n from the previous step,
#      which cuts iterations for small dt.


# ─── GMRES operator  ─────────────────────────────────────────────────────────

"""
    _apply_A_timedep(μ_tri, a, α, D)

GMRES operator for one backward-Euler step of the implicit advection-diffusion.

Substituting u = v = Δ⁻¹μ − h_corr(μ) into (αI − Δ + 2a ∂ζ + 2ā ∂ζ̄)u = αu^n
gives  Ã_α(μ) = −μ + α v + 2a ∂ζv + 2ā ∂ζ̄v.

# Arguments
- `μ_tri` : density `TriangularCoeffArray` (GMRES iterate)
- `a`     : complex advection velocity
- `α`     : 1/dt
- `D`     : disk discretization

# Returns
- `TriangularCoeffArray` of the same structure as `μ_tri`
"""
function _apply_A_timedep(μ_tri::TriangularCoeffArray, a, α, D)
    lmax  = D.Mr
    Mspan = μ_tri.Mspan
    N_m   = length(Mspan)

    # Build v = Δ⁻¹μ and its first derivatives via the exact sparse operators
    df = DiskFunction(μ_tri, D; derivatives = [(1,0), (0,1)])

    # Aliasing-correction harmonic: h_corr with BC = trace(Δ⁻¹μ)
    trace_hat = zeros(ComplexF64, N_m)
    for (i, m) in enumerate(Mspan)
        trace_hat[i] = trace(mode_coefficients(df._coeffs[1], m), lmax, m)
    end
    û_h    = [trace_hat[i] / ylm(abs(Mspan[i]), Mspan[i], 1.0) for i in 1:N_m]
    h_corr = HarmonicFunction(û_h, D; from_coefficients = true)

    # sub! corrects all populated slots for the aliasing harmonic:
    #   _coeffs[1] ← Δ⁻¹μ − h_corr          (= v,    zero-BC solution)
    #   _coeffs[2] ← ∂ζΔ⁻¹μ − ∂ζh_corr      (= ∂ζv,  exact derivative via ∂Ĝ∂ζ)
    #   _coeffs[3] ← ∂ζ̄Δ⁻¹μ − ∂ζ̄h_corr     (= ∂ζ̄v, exact derivative via ∂Ĝ∂ζ̄)
    sub!(df, h_corr)

    # Ã_α(μ) = −μ + α v + 2a ∂ζv + 2ā ∂ζ̄v
    return -μ_tri + α * df._coeffs[1] + 2*a * df._coeffs[2] + 2*conj(a) * df._coeffs[3]
end


# ─── Solution recovery ────────────────────────────────────────────────────────

"""
    _recover_u(μ_tri, D)

Given the GMRES density `μ_tri`, return u = Δ⁻¹μ − h_corr(μ) as a
`TriangularCoeffArray`.  This is stored as û_tri for the next time step.

# Arguments
- `μ_tri` : density `TriangularCoeffArray`
- `D`     : disk discretization

# Returns
- `TriangularCoeffArray` of u
"""
function _recover_u(μ_tri::TriangularCoeffArray, D)
    lmax  = D.Mr
    Mspan = μ_tri.Mspan
    N_m   = length(Mspan)

    df = DiskFunction(μ_tri, D)

    trace_hat = zeros(ComplexF64, N_m)
    for (i, m) in enumerate(Mspan)
        trace_hat[i] = trace(mode_coefficients(df._coeffs[1], m), lmax, m)
    end
    û_h    = [trace_hat[i] / ylm(abs(Mspan[i]), Mspan[i], 1.0) for i in 1:N_m]
    h_corr = HarmonicFunction(û_h, D; from_coefficients = true)
    sub!(df, h_corr)

    return copy(df._coeffs[1])
end


# ─── One backward-Euler step  ─────────────────────────────────────────────────

"""
    advance_step(û_tri, μ_tri_prev, a, dt, D; tol, itmax)

Advance one backward-Euler step of the implicit advection-diffusion equation.

From step n > 0 the RHS is formed as α û_tri (exact, no nodal conversion).
The GMRES is warm-started from the previous density `μ_tri_prev`.

# Arguments
- `û_tri`      : `TriangularCoeffArray` of u^n
- `μ_tri_prev` : density from the previous GMRES step (warm start); `nothing` on the first call
- `a`          : complex advection velocity
- `dt`         : time step
- `D`          : disk discretization
- `tol`        : GMRES tolerance (default 1e-10)
- `itmax`      : maximum GMRES iterations (default 200)

# Returns
- `û_tri_new`  : `TriangularCoeffArray` of u^{n+1}
- `μ_tri_new`  : density for warm-starting the next step and for exact derivatives
"""
function advance_step(û_tri::TriangularCoeffArray, μ_tri_prev, a, dt, D;
                      tol = 1e-10, itmax = 200)
    α = 1 / dt

    # RHS: assembled directly from the stored TriangularCoeffArray (key advantage 1)
    rhs_tri = α * û_tri

    x0 = μ_tri_prev !== nothing ? μ_tri_prev : zero(rhs_tri)

    μ_tri_new, info = linsolve(
        μ -> _apply_A_timedep(μ, a, α, D), rhs_tri, x0;
        tol = tol, maxiter = itmax, isposdef = false
    )
    info.converged == 0 && @warn "GMRES did not converge: $(info)"

    û_tri_new = _recover_u(μ_tri_new, D)
    return û_tri_new, μ_tri_new
end


# ─── Top-level solver  ────────────────────────────────────────────────────────

"""
    solve_advection_diffusion_timedep(u0, a, dt, nsteps, D; tol, itmax)

Time-integrate  ∂ₜu + 2(a ∂ζu + ā ∂ζ̄u) = Δu  on D with zero Dirichlet BC
using backward Euler.

# Arguments
- `u0`     : initial condition (Nr × Nθ nodal matrix)
- `a`      : complex advection velocity
- `dt`     : time step
- `nsteps` : number of time steps
- `D`      : disk discretization
- `tol`    : GMRES tolerance per step
- `itmax`  : maximum GMRES iterations per step

# Returns
- `u`     : final solution as a nodal matrix (Nr × Nθ)
- `μ_tri` : final density (use with `exact_derivatives_from_density` for diagnostics)
"""
function solve_advection_diffusion_timedep(u0, a, dt, nsteps, D;
                                           tol = 1e-10, itmax = 200)
    û_tri = NodalToTriangularArray(u0, D)
    μ_tri = nothing

    for _ in 1:nsteps
        û_tri, μ_tri = advance_step(û_tri, μ_tri, a, dt, D; tol = tol, itmax = itmax)
    end

    u = zeros(ComplexF64, size(D.ζ))
    ipsh!(u, û_tri, D)
    return u, μ_tri
end


# ─── Exact derivatives from stored density  ──────────────────────────────────
#
# At any step n, ∂ζu^n and ∂ζ̄u^n are available exactly from μ^n (the GMRES
# density returned by advance_step).  This avoids calling the ill-conditioned
# grid-space ∂ζ(u, D) operator.
#
# Since u^n = Δ⁻¹μ^n − h_corr^n, we have
#   ∂ζu^n  = ∂ζ(Δ⁻¹μ^n) − ∂ζh_corr^n   (via ∂Ĝ∂ζ  sparse operator, exact)
#   ∂ζ̄u^n = ∂ζ̄(Δ⁻¹μ^n) − ∂ζ̄h_corr^n  (via ∂Ĝ∂ζ̄ sparse operator, exact)

"""
    exact_derivatives_from_density(μ_tri, D)

Compute ∂ζu and ∂ζ̄u exactly from the stored GMRES density μ_tri.

Uses the sparse coefficient-space operators ∂Ĝ∂ζ and ∂Ĝ∂ζ̄ (no grid-space
finite differences, no ill-conditioned ∂ζ(u, D) call).

# Arguments
- `μ_tri` : density `TriangularCoeffArray` from `advance_step`
- `D`     : disk discretization

# Returns
- `(∂ζu, ∂ζ̄u)` : nodal matrices on the disk grid
"""
function exact_derivatives_from_density(μ_tri::TriangularCoeffArray, D)
    lmax  = D.Mr
    Mspan = μ_tri.Mspan
    N_m   = length(Mspan)

    df = DiskFunction(μ_tri, D; derivatives = [(1,0), (0,1)])

    trace_hat = zeros(ComplexF64, N_m)
    for (i, m) in enumerate(Mspan)
        trace_hat[i] = trace(mode_coefficients(df._coeffs[1], m), lmax, m)
    end
    û_h    = [trace_hat[i] / ylm(abs(Mspan[i]), Mspan[i], 1.0) for i in 1:N_m]
    h_corr = HarmonicFunction(û_h, D; from_coefficients = true)
    sub!(df, h_corr)

    # df._coeffs[2] = ∂ζ(Δ⁻¹μ − h_corr) = ∂ζu  (exact)
    # df._coeffs[3] = ∂ζ̄(Δ⁻¹μ − h_corr) = ∂ζ̄u (exact)
    return evaluate(df, 1, 0, D), evaluate(df, 0, 1, D)
end


# ─────────────────────────────────────────────────────────────────────────────
# Examples
# ─────────────────────────────────────────────────────────────────────────────

Mr = 50
D  = disk(Mr)
Nr = length(D.r)
Nθ = length(D.θ)

# Initial condition: Gaussian bump with exact zero BC
#   u0 = (1 − |ζ|²) · exp(−|ζ − c|²/σ²)
c  = 0.3 + 0.2im
σ² = 0.05
u0 = @. real((1 - abs2(D.ζ)) * exp(-abs2(D.ζ - c) / σ²))


# ── Test 1: pure diffusion (a = 0) ───────────────────────────────────────────
# The bump should spread and decay under Δu; solution stays O(u0) at short times.

println("Test 1 — pure diffusion (a = 0)")
a1          = 0.0 + 0.0im
dt1         = 0.01
T1          = 0.3
ns1         = round(Int, T1 / dt1)
u1, μ1_tri  = solve_advection_diffusion_timedep(u0, a1, dt1, ns1, D)
println("  max |u(T=$(T1))| = ", maximum(abs.(real.(u1))))


# ── Test 2: strong advection-diffusion ───────────────────────────────────────

println("Test 2 — advection-diffusion (a = 1.2 + 0.3im)")
a2          = 1.2 + 0.3im
dt2         = 0.005
T2          = 0.2
ns2         = round(Int, T2 / dt2)
u2, μ2_tri  = solve_advection_diffusion_timedep(u0, a2, dt2, ns2, D)
println("  max |u(T=$(T2))| = ", maximum(abs.(real.(u2))))


# ── Validate backward Euler: first-order convergence in dt ───────────────────
# Run from u0 to T=dt_ref with a single step of size dt_ref.
# Compare against many steps of size dt_fine.

println("Test 3 — validate first-order convergence in dt")
a3      = 0.5 + 0.3im
dt_ref  = 0.1
dt_fine = dt_ref / 10
T3      = dt_ref

u3_coarse, _ = solve_advection_diffusion_timedep(u0, a3, dt_ref,  1,               D)
u3_fine,   _ = solve_advection_diffusion_timedep(u0, a3, dt_fine, round(Int, T3/dt_fine), D)
err3 = maximum(abs.(real.(u3_coarse) .- real.(u3_fine)))
println("  ‖u_coarse − u_fine‖_∞ = ", err3, "   (should be O(dt_ref) ≈ $(dt_ref))")


# ── Exact derivatives: advantage from step n − 1  ────────────────────────────
#
# μ2_tri is the density from the last step of Test 2.
# ∂ζu and ∂ζ̄u are computed exactly without any grid-space differentiation.

println("Test 4 — exact derivatives from stored density")
∂ζu_exact,  ∂ζ̄u_exact  = exact_derivatives_from_density(μ2_tri, D)

# Compare to the (generally less accurate) grid-space operator ∂ζ(u2, D)
∂ζu_grid = ∂ζ(real.(u2), D)
err_∂ζ = maximum(abs.(∂ζu_exact .- ∂ζu_grid))
println("  ‖∂ζu_exact − ∂ζu_grid‖_∞ = ", err_∂ζ)
println("  (expect small discrepancy at low resolution; vanishes as Mr → ∞)")


# ─── Visualization ────────────────────────────────────────────────────────────

R_samples = collect(0.0:0.005:1.0)
Θ_samples = [D.θ..., 2π]
X_samples = (R_samples' .* cos.(Θ_samples))'
Y_samples = (R_samples' .* sin.(Θ_samples))'

function to_grid(u, D, R_samples)
    û_tri = NodalToTriangularArray(u, D)
    û_psh = TriangularArrayToPSH(û_tri, D)
    ui = hcat(ipsh(û_psh, D, R_samples), ipsh(û_psh, D, R_samples)[:,1])
    return real.(ui)
end

u0_grid = to_grid(u0,        D, R_samples)
u1_grid = to_grid(real.(u1), D, R_samples)
u2_grid = to_grid(real.(u2), D, R_samples)

fig = Figure(size = (1050, 350))
ax0 = Axis(fig[1, 1], title = "u₀ (initial)",                      aspect = DataAspect())
ax1 = Axis(fig[1, 2], title = "Diffusion only  t = $(T1)",          aspect = DataAspect())
ax2 = Axis(fig[1, 3], title = "Advection-diffusion  t = $(T2)",     aspect = DataAspect())

cr = extrema(u0_grid)
for (ax, zs) in zip((ax0, ax1, ax2), (u0_grid, u1_grid, u2_grid))
    surface!(ax, X_samples, Y_samples, fill(0f0, size(zs));
             color = zs, colorrange = cr, shading = NoShading, colormap = :coolwarm)
end

display(fig)
