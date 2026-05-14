
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
print_error("  N=1 direct mobility solve vs fmm solve: ", err)

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
  print_error("  N=2 oriented mobility solve $i: ", err_i)
end
