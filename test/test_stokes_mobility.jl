
using ProjectedSphericalHarmonics

println("Testing Stokes mobility solve...")

Mr, Mθ = 32, 16
D  = disk(Mr, Mθ)
x,y = real.(D.z), imag.(D.z)

# ── Test 1: N=1, compare against single-body solver ──────────────────────────

xcm  = [0.0 + 0.0im]
uinf(z) = (imag.(z), zero(real.(z)))    # linear shear: ux = y, uy = 0
F    = [[1.0, 0.5, 0.1]]

f_sol, U_sol = suspension_mobility_solve(xcm, zeros(length(xcm)), uinf, nothing, F, D)

# Reference via single-body solver (body at origin so z = D.z)
uinf1 = uinf(D.z)
U_ref, ω_ref = stokes_mobility_solve(uinf1, nothing, F[1], D)
err = maximum([abs(U_sol[1][1] - U_ref[1]), abs(U_sol[1][2] - U_ref[2]), abs(U_sol[1][3] - ω_ref)])
check("  N=1 direct mobility solve vs fmm solve: ", err, 1e-7)

# ── Test 2: N=2, non-zero orientations, rigid-body background flow ────────────
# Same exact solution as Test 2 (f=0, U = uinf), but with non-zero θcm.
# Tests that body-frame rotation of uinf is handled correctly.

U0, V0, ω0 = 0.7, 0.3, 0.15
uinf3(z) = (U0 .- ω0 .* imag.(z), V0 .+ ω0 .* real.(z))

xcm3 = [0.0 + 0.0im, 4.0 + 3.0im]
θcm3 = [π/4, π/3]
F3   = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

_, U3 = suspension_mobility_solve(xcm3, θcm3, uinf3, nothing, F3, D)

for i in 1:2
  Ux_exact = U0 - ω0 * imag(xcm3[i])
  Uy_exact = V0 + ω0 * real(xcm3[i])
  err_i = maximum([abs(U3[i][1] - Ux_exact), abs(U3[i][2] - Uy_exact), abs(U3[i][3] - ω0)])
  check("  N=2 oriented mobility solve $i: ", err_i, 1e-12)
end

# ── Test 3: viscosity convention ─────────────────────────────────────────────
# Unit disk, half-space (μ_bottom = 1, μ_top = 0): drag 16/3 and torque 16/3. Free space with
# μ = 1 (μ_bottom = μ_top = 1)
# recovers the classical edgewise-disk drag 32/3 μa and torque 32/3 μa³.

M1 = stokes_mobility_matrix(D)
check("  half-space disk drag (16/3): ", abs(M1[1,1] - 16/3), 1e-10)
check("  half-space disk torque (16/3): ", abs(M1[3,3] - 16/3), 1e-10)

M2 = stokes_mobility_matrix(D; μ_bottom=1.0, μ_top=1.0)
check("  free-space disk drag (32/3): ", abs(M2[1,1] - 32/3), 1e-10)
check("  free-space disk torque (32/3): ", abs(M2[3,3] - 32/3), 1e-10)

# Force-driven N-body problem: velocities scale as 1/(μ_bottom + μ_top), explicit defaults match exactly
F4 = [[1.0, 0.5, 0.1], [-0.3, 0.2, 0.0]]
_, U_default = suspension_mobility_solve(xcm3, θcm3, nothing, nothing, F4, D; eps=1e-12, gmrestol=1e-12)
_, U_mu1    = suspension_mobility_solve(xcm3, θcm3, nothing, nothing, F4, D; eps=1e-12, gmrestol=1e-12, μ_bottom=1.0, μ_top=0.0)
_, U_mu2    = suspension_mobility_solve(xcm3, θcm3, nothing, nothing, F4, D; eps=1e-12, gmrestol=1e-12, μ_bottom=1.0, μ_top=1.0)
@test U_mu1 == U_default
err_scale = maximum(abs.(vcat(U_mu2...) .- vcat(U_default...) ./ 2))
check("  N=2 mobility solve scales as 1/(μ_bottom + μ_top): ", err_scale, 1e-10)


# ── Test 4: Ewald + NUFFT vs FMM3D ───────────────────────────────────────────

using FMM3D: stfmm3d
using Random: MersenneTwister

rng  = MersenneTwister(3)
xcm5 = [complex(3.0 * (k % 4), 3.0 * (k ÷ 4)) + 0.3 * complex(rand(rng) - 0.5, rand(rng) - 0.5) for k in 0:11]
θcm5 = 2π .* rand(rng, length(xcm5))
Nz   = length(D.z)
src5 = ProjectedSphericalHarmonics.suspension_sources!(zeros(3, length(xcm5) * Nz), xcm5, θcm5, D)
stk5 = zeros(3, size(src5, 2)); stk5[1:2, :] = randn(rng, 2, size(src5, 2))

pot_fmm = stfmm3d(1e-13, src5; stoklet=stk5, ppreg=1).pot
for tol in (1e-6, 1e-10)
  E = ProjectedSphericalHarmonics.StokesEwald(src5; tol=tol)
  pot_e = ProjectedSphericalHarmonics.stokes_ewald!(similar(src5), E, stk5)
  # random-sign densities cancel, so the error relative to max |u| can exceed tol slightly
  check("  Ewald vs FMM3D Stokeslet sum (tol $tol): ", maximum(abs.(pot_e[1:2, :] .- pot_fmm[1:2, :])) / maximum(abs.(pot_fmm[1:2, :])), 10tol)
end

F5 = [[randn(rng), randn(rng), 0.1randn(rng)] for _ in xcm5]
_, U_ewald = suspension_mobility_solve(xcm5, θcm5, uinf, nothing, F5, D; eps=1e-10, gmrestol=1e-10)
_, U_fmm   = suspension_mobility_solve(xcm5, θcm5, uinf, nothing, F5, D; eps=1e-10, gmrestol=1e-10, method=:fmm)
check("  N=12 mobility solve, Ewald vs FMM: ", maximum(abs.(vcat(U_ewald...) .- vcat(U_fmm...))), 1e-8)
@test_throws ArgumentError suspension_mobility_solve(xcm5, θcm5, uinf, nothing, F5, D; method=:direct)

# ── Test 5: within-body subtraction by the precomputed body-frame matrix ─────
# Same velocities as the per-body direct sum, for both fast methods

S5 = suspension_self_matrix(D)
f5 = [(randn(rng, size(D.z)) .+ 0im, randn(rng, size(D.z)) .+ 0im) for _ in xcm5]
for method in (:ewald, :fmm)
  u_mat = suspension_velocity_eval(xcm5, θcm5, f5, D; eps=1e-10, method=method, S_self=S5)
  u_dir = suspension_velocity_eval(xcm5, θcm5, f5, D; eps=1e-10, method=method)
  err5 = maximum(maximum(abs.(u_mat[i][k] .- u_dir[i][k])) for i in eachindex(xcm5), k in 1:2) /
         maximum(maximum(abs.(u_dir[i][k])) for i in eachindex(xcm5), k in 1:2)
  check("  S_self matrix vs per-body direct subtraction ($method): ", err5, 1e-10)
end
@test_throws DimensionMismatch suspension_velocity_eval(xcm5, θcm5, f5, D; S_self=zeros(2, 2))
