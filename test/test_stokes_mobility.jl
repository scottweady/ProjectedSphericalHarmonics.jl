
using ProjectedSphericalHarmonics

println("Testing stokes_mobility_fmm_solve...")

Mr, Mθ = 32, 16
D  = disk(Mr, Mθ)
xg = real.(D.z)
yg = imag.(D.z)

# ── Test 1: N=1, compare against single-body solver ──────────────────────────

xcm  = [0.0 + 0.0im]
uinf_shear(z) = (imag.(z), zero(real.(z)))    # linear shear: ux = y, uy = 0
F    = [[1.0, 0.5, 0.1]]

f_sol, U_sol = stokes_mobility_fmm_solve(xcm, uinf_shear, nothing, F, D)

# Reference via single-body solver (body at origin so z = D.z)
uinf1 = uinf_shear(D.z)
U_ref, ω_ref = stokes_mobility_solve(uinf1, (F[1][1], F[1][2]), F[1][3], D)
err = maximum([abs(U_sol[1][1] - U_ref[1]), abs(U_sol[1][2] - U_ref[2]), abs(U_sol[1][3] - ω_ref)])
print_error("  N=1 direct mobility solve vs fmm solve: ", err)

# ── Test 2: N=2, rigid-body background flow (exact known answer) ──────────────
# For uinf(z) = (U0 - ω0·y, V0 + ω0·x), the particles move exactly with (U0,V0,ω0)
# and the surface force density is zero.

U0, V0, ω0 = 1.0, 0.5, 0.2
uinf(z) = (U0 .- ω0 .* imag.(z), V0 .+ ω0 .* real.(z))

xcm2 = [0.0 + 0.0im, 5.0 + 0.0im]
F2   = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
_, U2 = stokes_mobility_fmm_solve(xcm2, uinf, nothing, F2, D)

for i in 1:2
  Ux_exact = U0 - ω0 * imag(xcm2[i])
  Uy_exact = V0 + ω0 * real(xcm2[i])
  err_i = maximum([abs(U2[i][1] - Ux_exact), abs(U2[i][2] - Uy_exact), abs(U2[i][3] - ω0)])
  print_error("  N=2 mobility solve $i: ",err_i)
end
