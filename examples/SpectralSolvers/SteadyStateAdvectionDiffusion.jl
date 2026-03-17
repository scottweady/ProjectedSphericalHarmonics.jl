using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra
# using Krylov
using KrylovKit


# Solver for the steady-state advection-diffusion equation on the disk with Dirichlet BC.
# The PDE in complex form:
#   4 ∂ζ∂ζ̄ u + 2 (a ∂ζ u + ā ∂ζ̄ u) = f    on D
#   u = g    on ∂D
#
# where a ∈ ℂ is the (constant) complex advection velocity.
#
# Decomposition: u = Δ⁻¹μ + u_h
# where Δ⁻¹μ has zero Dirichlet BC and u_h is harmonic with u_h|_{∂D} = g - Δ⁻¹μ|_{∂D}.
#
# Substituting (Δ(Δ⁻¹μ) = μ, Δu_h = 0):
#   μ + 2a ∂ζΔ⁻¹μ + 2ā ∂ζ̄Δ⁻¹μ + 2(a ∂ζ + ā ∂ζ̄) u_h = f
#
# Since u_h depends linearly on μ via the BC, eliminating u_h yields:
#
#   [I + 2a ∂ζΔ⁻¹ + 2ā ∂ζ̄Δ⁻¹ - 2(a ∂ζ + ā ∂ζ̄)∘SolveHarmonic∘DirichletTrace∘Δ⁻¹] μ
#       = f - 2(a ∂ζ + ā ∂ζ̄) SolveHarmonic(g)
#
# The GMRES unknown μ is a TriangularCoeffArray (PSH coefficient space), NOT a
# nodal grid of size Nr × Nθ. The per-frequency operators are applied directly,
# exploiting the frequency-shift rules:
#   ∂ζΔ⁻¹_m_sparse  : input at freq m → output at freq m-1
#   ∂ζ̄Δ⁻¹_m_sparse : input at freq m → output at freq m+1
# To fill column m of the result:
#   ∂ζΔ⁻¹ contribution: feed column(μ, m+1)  [shifts m+1 → m]
#   ∂ζ̄Δ⁻¹ contribution: feed column(μ, m-1) [shifts m-1 → m]
# The harmonic correction only populates column(result, m)[1] (l = |m| mode).




# ─── Coefficient-space linear operator ───────────────────────────────────────

"""
    _apply_A_coeff(μ_tri, a, D)

Apply the advection-diffusion operator in PSH coefficient space:
    Aμ = μ + 2a ∂ζΔ⁻¹μ + 2ā ∂ζ̄Δ⁻¹μ
           - 2(a ∂ζ + ā ∂ζ̄) SolveHarmonic(DirichletTrace(Δ⁻¹μ))

A `DiskFunction` with derivative slots (1,0) and (0,1) is built from `μ_tri`,
giving `∂ζΔ⁻¹μ` and `∂ζ̄Δ⁻¹μ` directly as `TriangularCoeffArray` objects.
Then `sub!(df, h_corr)` subtracts the harmonic correction from both slots at once.

# Arguments
- `μ_tri` : source density as a `TriangularCoeffArray`
- `a`     : complex advection velocity
- `D`     : disk discretization

# Returns
- `TriangularCoeffArray` of the same structure as `μ_tri`
"""
function _apply_A_coeff(μ_tri::TriangularCoeffArray, a, D)
    lmax  = D.Mr
    Mspan = μ_tri.Mspan
    N_m   = length(Mspan)

    # Slots: (0,0) = Δ⁻¹μ,  (1,0) = ∂ζΔ⁻¹μ,  (0,1) = ∂ζ̄Δ⁻¹μ
    df = DiskFunction(μ_tri, D; derivatives = [(1,0), (0,1)])

    # Build harmonic correction with BC = Δ⁻¹μ|_{∂D}
    trace_hat = zeros(ComplexF64, N_m)

    for (i, m) in enumerate(Mspan)
        trace_hat[i] = DirichletTrace(column(df._coeffs[1], m), lmax, m)
    end

    û_h    = [trace_hat[i] / ylm(abs(Mspan[i]), Mspan[i], 1.0) for i in 1:N_m]
    h_corr = HarmonicFunction(û_h, D; from_coefficients = true)

    # sub! modifies all populated slots:
    #   slot (1,0) ← ∂ζΔ⁻¹μ - ∂ζh_corr
    #   slot (0,1) ← ∂ζ̄Δ⁻¹μ - ∂ζ̄h_corr
    sub!(df, h_corr)

    # Aμ = μ + 2a*(∂ζΔ⁻¹μ - ∂ζh_corr) + 2ā*(∂ζ̄Δ⁻¹μ - ∂ζ̄h_corr)
    # Slot indices: (1,0) → 2,  (0,1) → 3  (see _IDX_TO_IJ in AbstractDiskFunction)
    return μ_tri + 2*a * df._coeffs[2] + 2*conj(a) * df._coeffs[3]
end


"""
    SolveAdvectionDiffusion(f, g, a, D; tol=1e-10, itmax=500, krylovdim=nothing)

Solve the steady-state advection-diffusion equation on the unit disk:
    4 ∂ζ∂ζ̄ u + 2 (a ∂ζ u + ā ∂ζ̄ u) = f    on D
    u = g    on ∂D

via the decomposition `u = Δ⁻¹μ + u_h` and GMRES in the PSH coefficient space.
The GMRES unknown is the source density `μ` as a `TriangularCoeffArray`.

`krylovdim` controls the Krylov subspace dimension before restart. Defaults to the
full coefficient-space size (unrestarted GMRES), which avoids a type-conversion issue
in KrylovKit's restart path for custom vector types.

# Arguments
- `f`          : source term, nodal values (Nr × Nθ matrix)
- `g`          : Dirichlet boundary data (vector of length Nθ = 2Mθ+1)
- `a`          : complex advection velocity (scalar)
- `D`          : disk discretization
- `tol`        : GMRES tolerance (default 1e-10)
- `itmax`      : maximum outer iterations (default 500)
- `krylovdim`  : Krylov subspace size; `nothing` uses the full coefficient-space dimension

# Returns
- `u` : solution on the disk grid (Nr × Nθ matrix)
"""
function SolveAdvectionDiffusion(f, g, a, D; tol=1e-10, itmax=500, krylovdim=nothing)

    lmax  = D.Mr
    Mspan = vec(Array(D.Mspan))
    N_m   = length(Mspan)

    # ── RHS in coefficient space ──────────────────────────────────────────────
    # rhs_tri = NodalToTriangularArray(f) - 2(a ∂ζ + ā ∂ζ̄) u_h^g
    # The harmonic correction is purely at the l=|m| mode of each column.
    f_tri = NodalToTriangularArray(f, D)
    h_g   = HarmonicFunction(g, D)

    dû_ζ  = zeros(ComplexF64, N_m)
    dû_ζ̄  = zeros(ComplexF64, N_m)
    ∂ζ_HarmonicFunction!(dû_ζ,  h_g.û, Mspan)
    ∂ζ̄_HarmonicFunction!(dû_ζ̄, h_g.û, Mspan)

    rhs_tri = copy(f_tri)
    
    for (i, m) in enumerate(Mspan)
        column(rhs_tri, m)[1] -= 2*(a * dû_ζ[i] + conj(a) * dû_ζ̄[i])
    end

    _apply_A_coeff(rhs_tri, a, D)

    # ── GMRES in coefficient space ────────────────────────────────────────────
    kdim = something(krylovdim, length(rhs_tri))
    # μ_tri, info = linsolve(μ -> _apply_A_coeff(μ, a, D), rhs_tri;
    #                        tol=tol, maxiter=itmax, krylovdim=kdim, isposdef=false)

    μ_tri, info = linsolve(μ -> _apply_A_coeff(μ, a, D), rhs_tri;
                           tol=tol, maxiter=itmax, isposdef=false)

    info.converged == 0 && @warn "GMRES did not converge: $(info)"
    display(info)

    # ── Reconstruct u = Δ⁻¹μ + u_h ──────────────────────────────────────────
    df_final = DiskFunction(μ_tri, D)

    # Corrected harmonic BC: g - Δ⁻¹μ|_{∂D}
    trace_hat = zeros(ComplexF64, N_m)

    for (i, m) in enumerate(Mspan)
        trace_hat[i] = DirichletTrace(column(df_final._coeffs[1], m), lmax, m)
    end
    bc_correction = real.(ifft(trace_hat) * N_m)
    h_final = HarmonicFunction(g .- bc_correction, D)

    # add! inserts h_final into slot (0,0): df_final now holds Δ⁻¹μ + u_h
    add!(df_final, h_final)

    return evaluate(df_final, D)
end


# ─────────────────────────────────────────────────────────────────────────────
# Validation examples
# ─────────────────────────────────────────────────────────────────────────────

Mr = 40
D  = disk(Mr)
Nr = length(D.r)
Nθ = length(D.θ)

# ── Test 1: pure Laplacian (a = 0) ───────────────────────────────────────────
# Exact solution: u = (1 - |ζ|²) / 4,  Δu = -1  ⟹  f = -1, g = 0
a1       = 0.0 + 0.0im
f1       = -ones(Nr, Nθ)
g1       = zeros(Nθ)
u_sol1   = SolveAdvectionDiffusion(f1, g1, a1, D)
u_exact1 = @. (1 - abs(D.ζ)^2) / 4
println("Test 1 — pure Laplacian (a = 0)")
println("  max |error| = ", maximum(abs.(real.(u_sol1) .- real.(u_exact1))))

# ── Test 2: advection-diffusion with non-trivial a ────────────────────────────
# Exact solution: u = 1 - |ζ|²
# ∂ζu = -ζ̄,  ∂ζ̄u = -ζ
# f = Δu + 2(a∂ζu + ā∂ζ̄u) = -4 - 2(aζ̄ + āζ),  g = 0
a2       = 0.5 + 0.3im
f2       = @. real(-4 - 2*(a2*conj(D.ζ) + conj(a2)*D.ζ))
g2       = zeros(Nθ)
u_sol2   = SolveAdvectionDiffusion(f2, g2, a2, D)
u_exact2 = @. 1 - abs(D.ζ)^2
println("Test 2 — advection-diffusion (a = $(a2))")
println("  max |error| = ", maximum(abs.(real.(u_sol2) .- real.(u_exact2))))


# ── For fancies

Mr = 400
D  = disk(Mr)
Nr = length(D.r)
Nθ = length(D.θ)


a3       = 80.0 + 0.01im 
f3 = real.(exp.(-abs2.(D.ζ .- (0.4 + 0.3*im))/0.01 ) ) +  real.(exp.(-abs2.(D.ζ .- (0.4 - 0.3*im))/0.01 ) )
g3       = zeros(Nθ)
u_sol3   = SolveAdvectionDiffusion(f3, g3, a3, D; itmax=50)




# ── Visualization manufactured solution ─────────────────────────────────────────────────────────────
R_samples = collect(0.0:0.01:1.0)
Θ_samples = [D.θ..., 2π]
X_samples = (R_samples' .* cos.(Θ_samples))'
Y_samples = (R_samples' .* sin.(Θ_samples))'

u_interp2       = ipsh(psh(real.(u_sol2),   D), D, R_samples)
u_interp_exact2 = ipsh(psh(real.(u_exact2), D), D, R_samples)
u_interp2       = hcat(u_interp2,       u_interp2[:,1])
u_interp_exact2 = hcat(u_interp_exact2, u_interp_exact2[:,1])

zs_sol   = real.(u_interp2)
zs_exact = real.(u_interp_exact2)
zs_err   = abs.(zs_sol .- zs_exact)

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Solution u",     aspect = DataAspect())
ax2 = Axis(fig[1, 2], title = "Exact solution",  aspect = DataAspect())
ax3 = Axis(fig[1, 3], title = "Pointwise error", aspect = DataAspect())

srf1 = surface!(ax1, X_samples, Y_samples, fill(0f0, size(zs_sol));   color = zs_sol,   colorrange = extrema(zs_sol),   shading = NoShading, colormap = :coolwarm)
srf2 = surface!(ax2, X_samples, Y_samples, fill(0f0, size(zs_exact)); color = zs_exact, colorrange = extrema(zs_exact), shading = NoShading, colormap = :coolwarm)
srf3 = surface!(ax3, X_samples, Y_samples, fill(0f0, size(zs_err));   color = zs_err,   colorrange = extrema(zs_err),   shading = NoShading, colormap = :inferno)

levels = Makie.get_tickvalues(Makie.LinearTicks(20), extrema(zs_sol)...)
# contour!(ax1, X_samples, Y_samples, max.(0.0, zs_sol); color = :black, levels = levels, labels = false)
# contour!(ax1, X_samples, Y_samples, min.(0.0, zs_sol); color = :black, levels = levels, labels = false, linestyle = :dash)

# Colorbar(fig[1, 1][1, 2], srf1)
# Colorbar(fig[1, 2][1, 2], srf2)
# Colorbar(fig[1, 3][1, 2], srf3)
display(fig)





# ── Visualization ─────────────────────────────────────────────────────────────
R_samples = collect(0.0:0.01:1.0)
Θ_samples = [D.θ..., 2π]
X_samples = (R_samples' .* cos.(Θ_samples))'
Y_samples = (R_samples' .* sin.(Θ_samples))'

u_interp3       = ipsh(psh(real.(u_sol3),   D), D, R_samples)
# u_interp_exact2 = ipsh(psh(real.(u_exact2), D), D, R_samples)
u_interp3       = hcat(u_interp3,       u_interp3[:,1])
# u_interp_exact2 = hcat(u_interp_exact2, u_interp_exact2[:,1])

zs_sol   = real.(u_interp3)
# zs_exact = real.(u_interp_exact2)
# zs_err   = abs.(zs_sol .- zs_exact)

fig = Figure()
ax1 = Axis(fig[1, 1], title = "Solution u",     aspect = DataAspect())
ax2 = Axis(fig[1, 2], title = "Zoom in",  aspect = DataAspect())
# ax3 = Axis(fig[1, 3], title = "Pointwise error", aspect = DataAspect())

srf1 = surface!(ax1, X_samples, Y_samples, fill(0f0, size(zs_sol));   color = zs_sol,   colorrange = extrema(zs_sol),   shading = NoShading, colormap = :coolwarm)
srf2 = surface!(ax2, X_samples, Y_samples, fill(0f0, size(zs_sol));   color = zs_sol,   colorrange = extrema(zs_sol),   shading = NoShading, colormap = :coolwarm)

xlims!(ax2, -1., -0.5)
ylims!(ax2, 0.0, 1.0)

display(fig)
# srf2 = surface!(ax2, X_samples, Y_samples, fill(0f0, size(zs_exact)); color = zs_exact, colorrange = extrema(zs_exact), shading = NoShading, colormap = :coolwarm)
# srf3 = surface!(ax3, X_samples, Y_samples, fill(0f0, size(zs_err));   color = zs_err,   colorrange = extrema(zs_err),   shading = NoShading, colormap = :inferno)

levels = Makie.get_tickvalues(Makie.LinearTicks(10), extrema(zs_sol)...)
# contour!(ax1, X_samples, Y_samples, max.(0.0, zs_sol); color = :black, levels = levels, labels = false)
contour!(ax2, X_samples, Y_samples, min.(0.0, zs_sol); color = :black, levels = levels, labels = false, linestyle = :dash)

# Colorbar(fig[1, 1][1, 2], srf1)
# Colorbar(fig[1, 2][1, 2], srf2)
# Colorbar(fig[1, 3][1, 2], srf3)
display(fig)