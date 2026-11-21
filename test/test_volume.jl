println("Testing volume potentials...")

const PSH = ProjectedSphericalHarmonics

#=
Closed forms for the density u = ρ^n e^{imθ} (n - |m| even, n ≥ |m|).

Log kernel. 𝒱[u] = v(ρ) e^{imθ} with v'' + v'/ρ - m²v/ρ² = ρ^n inside and = 0 outside, v regular at 0,
v and v' continuous at ρ = 1, and v ~ (log ρ/2π) ∫u (m = 0) or v → 0 (m ≠ 0) at infinity:
  inside:  m = 0: (ρ^{n+2} - 1)/(n+2)²;
           m ≠ 0: ρ^{n+2}/((n+2)² - m²) - ρ^{|m|}/(2|m|(n - |m| + 2))
  outside: m = 0: log(ρ)/(n+2);  m ≠ 0: -ρ^{-|m|}/(2|m|(n + |m| + 2)).
=#
function V_exact(n, m, x)
  R, a = abs(x), abs(m)
  ph = cis(m * angle(x))
  if R ≥ 1
    return (m == 0 ? log(R) / (n + 2) : -R^(-a) / (2a * (n + a + 2))) * ph
  end
  m == 0 && return (R^(n + 2) - 1) / (n + 2)^2 * ph
  return (R^(n + 2) / ((n + 2)^2 - m^2) - R^a / (2a * (n - a + 2))) * ph
end

#=
|x - y| kernel. With |ρe^{iθ} - r| = max(ρ, r) Σ_k c_k(q) e^{ikθ}, q = min/max,
c_k = (-1)^k Σ_b β_{b+|k|} β_b q^{2b+|k|}, β_b = binomial(1/2, b), the potential of u at r e^{iα} is
2π (-1)^a e^{imα} Σ_b β_{b+a} β_b S_b, a = |m|, with S_b = r^{1-2b-a}/(2b+a+n+2) for r ≥ 1 and
S_b = r^{n+3}/(2b+a+n+2) + (r^{2b+a} - r^{n+3})/(n+3-2b-a) for r < 1 (summed smallest first). The
series converges like b^{-4} at r = 1, geometrically otherwise.
=#
function T_exact(n, m, x)
  r, a = abs(x), abs(m)
  B = r == 1 ? 10^6 : clamp(ceil(Int, 50 / abs(1 - r)), 10^4, 10^6)
  β = zeros(B + a + 2); β[1] = 1.0
  for j = 1 : B + a + 1
    β[j + 1] = β[j] * (1.5 - j) / j
  end
  S = Float64[]
  for b = 0 : B
    c = β[b + a + 1] * β[b + 1]
    t = r ≥ 1 ? c * r^(1 - 2b - a) / (2b + a + n + 2) :
                c * (r^(n + 3) / (2b + a + n + 2) + (r^(2b + a) - r^(n + 3)) / (n + 3 - 2b - a))
    push!(S, t)
    abs(t) < 1e-22 && b > 50 && break
  end
  return 2π * (-1)^a * sum(reverse(S)) * cis(m * angle(x)) / 8π
end

monomial(D, n, m) = abs.(D.z).^n .* cis.(m .* angle.(D.z))

# Targets: centre, interior, near the rim, on the rim, just outside, far outside
x = ComplexF64[0, 0.3 + 0.2im, -0.7im, 0.999 * cis(2.0), 0.99999 * cis(0.3), cis(1.0), -1.0, 1.0000001 * cis(2.0),
               1.01im, 1.5, -3.0im]
x_out = filter(z -> abs(z) ≥ 1, x)
in_D(z) = abs(z) < 1

# ---------------------------------------------------------------------------------------------
# 𝒱
# ---------------------------------------------------------------------------------------------

D = disk(16)

V1 = 𝒱(1, D)
check("  𝒱(1): max error on the grid: ", maximum(abs.(V1 .- (abs2.(D.z) .- 1) / 4)), 1e-14)
err = abs.(𝒱(x, 1, D) .- [in_D(z) ? (abs2(z) - 1) / 4 : log(abs(z)) / 2 for z in x])
check("  𝒱(1): max error at targets (inside, rim, outside): ", maximum(err), 1e-14)

for (n, m, name) in ((2, 0, "|ζ|²"), (3, 3, "ζ³"), (4, -4, "ζ̄⁴"), (5, 3, "|ζ|²ζ³"), (6, 0, "|ζ|⁶"), (16, 16, "ζ¹⁶"))
  uₙ = monomial(D, n, m)
  e_grid = maximum(abs.(𝒱(uₙ, D) .- V_exact.(n, m, D.z)))
  e_tgt = maximum(abs.(𝒱(x, uₙ, D) .- V_exact.(n, m, x)))
  check("  𝒱($name): max error on the grid / at targets: $e_grid / ", e_tgt, 1e-14)
  @test e_grid ≤ 1e-14
end

# Grid evaluation and target evaluation agree
u = exp.(D.z) .+ abs2.(D.z)
check("  𝒱 at the grid points as targets vs on the grid: ", maximum(abs.(𝒱(D.z, u, D) .- 𝒱(u, D))), 1e-14)

# Smooth, non-polynomial density: u = e^ζ + e^{-|ζ|²} = Σ ζ^k/k! + Σ (-1)^k ρ^{2k}/k!
u_smooth(z) = exp(z) + exp(-abs2(z))
V_smooth(z) = sum(V_exact(k, k, z) / factorial(big(k)) + (-1)^k * V_exact(2k, 0, z) / factorial(big(k)) for k = 0 : 30)
println("  Convergence of 𝒱 for u = e^ζ + e^{-|ζ|²} (max error on the grid: spectral / direct sum; at targets):")
errs = Float64[]
for M in (4, 8, 12, 16, 24)
  DM = disk(M)
  uM = u_smooth.(DM.z)
  Vex = ComplexF64.(V_smooth.(DM.z))
  e_new = maximum(abs.(𝒱(uM, DM) .- Vex))
  e_old = maximum(abs.(PSH._laplace2d_volume_direct(uM, DM) .- Vex))
  e_tgt = maximum(abs.(𝒱(x, uM, DM) .- ComplexF64.(V_smooth.(x))))
  println("    M = $M:  $e_new  /  $e_old;  $e_tgt")
  push!(errs, max(e_new, e_tgt))
end
check("  𝒱 smooth density, M = 24: ", errs[end], 1e-14)
@test errs[3] < 1e-8

# ---------------------------------------------------------------------------------------------
# 𝒯_vol
# ---------------------------------------------------------------------------------------------

check("  𝒯_vol(1) at the centre (1/12): ", abs(𝒯_vol(0im, 1, D) - 1 / 12), 1e-15)
check("  𝒯_vol(1) on the rim (4/(9π)): ", abs(𝒯_vol(1.0 + 0im, 1, D) - 4 / 9π), 1e-15)

for (n, m, name) in ((0, 0, "1"), (2, 0, "|ζ|²"), (3, 3, "ζ³"), (5, -1, "|ζ|⁴ζ̄"), (16, 16, "ζ¹⁶"), (15, 1, "|ζ|¹⁴ζ"))
  uₙ = monomial(D, n, m)
  I = abs.(D.z) .< 0.99                     # the reference series is slow very near the rim
  e_grid = maximum(abs.(𝒯_vol(uₙ, D)[I] .- T_exact.(n, m, D.z[I])))
  e_tgt = maximum(abs.(𝒯_vol(x, uₙ, D) .- T_exact.(n, m, x)))
  check("  𝒯_vol($name): max error on the grid / at targets: $e_grid / ", e_tgt, 1e-14)
  @test e_grid ≤ 1e-14
end

# Spectral convergence for u = e^ζ (= Σ ζ^k/k!), at the targets
T_exp(z) = sum(T_exact(k, k, z) / factorial(big(k)) for k = 0 : 25)
Tx = ComplexF64.(T_exp.(x))
println("  Convergence of 𝒯_vol for u = e^ζ (max error at targets):")
errs = Float64[]
for M in (4, 8, 12, 16)
  DM = disk(M)
  push!(errs, maximum(abs.(𝒯_vol(x, exp.(DM.z), DM) .- Tx)))
  println("    M = $M:  $(errs[end])")
end
check("  𝒯_vol smooth density, M = 16: ", errs[end], 1e-14)

# ---------------------------------------------------------------------------------------------
# Ring weights (contract used for non-conformal maps)
# ---------------------------------------------------------------------------------------------

D = disk(12)
n = length(D.idx_even)
∫y = transpose(ipsh_matrix(D, vec(D.ζ))) * vec(D.dz)        # ∫_D y_j dA (the grid rule is exact)
for r in (0.0, 0.6, 1.0, 1.3)
  wlog, wabs = PSH._volume_ring_weights(D, r)
  wlog2, wabs2, Wk, Vk = PSH._volume_ring_weights(D, r; N=3)
  @test size(wlog) == size(wabs) == (n,)
  @test size(Wk) == size(Vk) == (n, 7)
  @test wlog2 == wlog
  @test maximum(abs.(wabs2 .- wabs)) ≤ 1e-14
  # k = 0 gives ∫_D y_j dA; the weights are real (reflection ζ ↦ ζ̄ fixes the target)
  e1 = maximum(abs.(Wk[:, 4] .- ∫y))
  e2 = maximum(abs.(imag.([wlog; wabs; vec(Wk); vec(Vk)])))
  check("  Ring weights at r = $r: error in ∫ y_j dA $e1, imaginary parts ", e2, 1e-14)
  @test e1 ≤ 1e-14
end
