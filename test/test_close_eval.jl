using ProjectedSphericalHarmonics
using Random
using SpecialFunctions: loggamma, logabsgamma

const PSH = ProjectedSphericalHarmonics

println("Testing close evaluation...")

Random.seed!(1)

# ── Radial factor vs BigFloat series ─────────────────────────────────────────

function radial_big(l, k, r)
  setprecision(BigFloat, 512) do
    A, B, C, z = big(l + k + 1) / 2, big(l - k + 1) / 2, big(l) + big(3) / 2, 1 / big(r)^2
    s = t = one(BigFloat)
    j = 0
    while j < 10 || abs(t) > big(10.0)^-40 * abs(s)
      t *= (A + j) * (B + j) / ((C + j) * (j + 1)) * z
      s += t
      j += 1
    end
    lg2, s2 = logabsgamma((k - l + 1) / big(2))
    return s2 * exp(loggamma(A) - lg2 - loggamma(C) - (l + 1) * log(big(r))) * s
  end
end

err = 0.0
for r in (1.005, 1.05, 2.0)
  G = PSH.exterior_radial_table(16, 100, r)
  for l in (0, 3, 16), k in (l % 2) : 2 : 100
    exact = radial_big(l, k, r)
    abs(exact) < 1e-8 && continue          # exact zeros (e.g. ℓ = 0, sin(k arcsin(1/r)) = 0)
    global err = max(err, Float64(abs(G[l + 1, k + 1] - exact) / abs(exact)))
  end
end
check("  Radial factor vs BigFloat series: ", err, 1e-10)

# ── Cross velocity vs upsampled direct sum ───────────────────────────────────

"Random real density g = f·ω with decaying coefficients; returns even PSH coefficients."
function random_density(D)
  c = zeros(ComplexF64, D.shp)
  for l in 0 : D.Mℓ, mi in eachindex(D.Mspan)
    D.even[l + 1, mi] && (c[l + 1, mi] = randn(ComplexF64) * exp(-0.25l))
  end
  return psh(real.(ipsh(c, D)), D)
end

"Direct sum of the single layer (body frame) over an (Nr, Nθ) grid on the ellipse (a, b)."
function ellipse_direct(a, b, ĉ, D, Nr, Nθ, targets)
  r, θ, ζ, dζ = PSH.diskpts(Nr, Nθ)
  # Tensor grid: sum the radial profiles of each order m, then multiply by e^{imθ}
  synth(c) = real.(sum(sum(c[l + 1, mi] .* ylm(l, m, r .+ 0im) for l in abs(m) : D.Mℓ if D.even[l + 1, mi]) .* cis.(m .* θ)
                       for (mi, m) in enumerate(D.Mspan)))
  wt = a * b .* real.(dζ) ./ sqrt.(1 .- abs2.(ζ))
  fx, fy = vec(synth(ĉ[1]) .* wt), vec(synth(ĉ[2]) .* wt)
  xs, ys = vec(a .* real.(ζ)), vec(b .* imag.(ζ))
  u = zeros(2, length(targets))
  Threads.@threads for t in eachindex(targets)
    X, Y = real(targets[t]), imag(targets[t])
    ux = uy = 0.0
    @inbounds for j in eachindex(xs)
      dx, dy = X - xs[j], Y - ys[j]
      r2 = dx^2 + dy^2
      ir = 1 / sqrt(r2); ir3 = ir / r2
      d = dx * fx[j] + dy * fy[j]
      ux += fx[j] * ir + dx * d * ir3
      uy += fy[j] * ir + dy * d * ir3
    end
    u[1, t], u[2, t] = ux / 4π, uy / 4π
  end
  return u
end

M = 16
D = disk(M)
ĉ = (random_density(D), random_density(D))
ĉvec = vcat(vec(ĉ[1])[D.idx_even], vec(ĉ[2])[D.idx_even])

for (ratio, gap) in ((1.0, 0.01), (4.0, 0.01))

  local a, b = sqrt(ratio), 1 / sqrt(ratio)
  Ω = ratio == 1 ? D : ellipse(a, b, M)
  geom = PSH.exterior_geometry(Ω)

  # Antiparallel pair, long axes vertical, separated by the b-axis: nodes of body 2 in body 1's frame
  Δx = 2b + gap
  c1, θ1, c2, θ2 = -Δx/2 + 0im, -π/2, Δx/2 + 0im, π/2
  targets = cis(-θ1) .* (c2 .+ cis(θ2) .* vec(Ω.z) .- c1)
  ξ = real.(targets) ./ a .+ im .* imag.(targets) ./ b
  near = findall(abs.(ξ) .< 1.5)

  T = PSH.exterior_map(geom, Ω, ξ[near])
  v = T * ĉvec
  u_close = vcat(transpose(real.(v[1:length(near)])), transpose(real.(v[length(near)+1:end])))

  u_ref = ellipse_direct(a, b, ĉ, D, 200, 4096, targets[near])
  u_nat = ellipse_direct(a, b, ĉ, D, M + 1, 2M + 1, targets[near])
  scale = maximum(abs, u_ref)

  check("  Cross velocity a/b = $ratio, gap = $gap (closed form): ", maximum(abs, u_close .- u_ref) / scale, 1e-9)
  println("    (native node quadrature: $(maximum(abs, u_nat .- u_ref) / scale))")

end

# ── End to end: suspension_velocity_eval on a close pair of ellipses ─────────

# Long axes vertical, side by side at gap 0.05 and offset vertically, then the whole
# configuration rotated so that neither body frame is aligned with the world frame
a, b = sqrt(4.0), 1 / sqrt(4.0)
Ω = ellipse(a, b, M)
φ = 0.4
xcm = cis(φ) .* [0.0 + 0.0im, 2b + 0.05 + 0.3im]
θcm = φ .+ [π/2, -π/2]
f̃ = [(random_density(Ω), random_density(Ω)) for _ in 1 : 2]
f_body = [map(c -> real.(ipsh(c, Ω)) ./ Ω.w, fi) for fi in f̃]

u_close  = suspension_velocity_eval(xcm, θcm, f_body, Ω; eps=1e-13)
u_native = suspension_velocity_eval(xcm, θcm, f_body, Ω; eps=1e-13, close_eval=false)

# Reference at the nodes of body 2: self term + upsampled direct sum of body 1, in body 2's frame
targets = cis(-θcm[1]) .* (xcm[2] .+ cis(θcm[2]) .* vec(Ω.z) .- xcm[1])
u12 = ellipse_direct(a, b, f̃[1], Ω, 200, 4096, targets)
u12x, u12y = PSH.rotate(u12[1, :], u12[2, :], θcm[1] - θcm[2])
u_self = 𝒮_st(f_body[2], Ω)
u_ref = (vec(real.(u_self[1])) .+ u12x, vec(real.(u_self[2])) .+ u12y)

e(u) = maximum(abs, vcat(vec(real.(u[2][1])) .- u_ref[1], vec(real.(u[2][2])) .- u_ref[2])) /
       maximum(abs, vcat(u_ref...))
check("  suspension_velocity_eval, ellipse pair at gap 0.05 (closed form): ", e(u_close), 1e-9)
println("    (native node quadrature: $(e(u_native)))")

# ── Mobility solve: closed form and native agree for well-separated bodies ───

D = disk(16)
xcm = [0.0 + 0.0im, 6.0 + 2.0im]
θcm = [0.3, -0.5]
uinf(z) = (imag.(z), zero(real.(z)))
F = [[1.0, 0.5, 0.1], [-0.3, 0.2, 0.0]]
_, U_close  = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, D; eps=1e-12, gmrestol=1e-12)
_, U_native = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, D; eps=1e-12, gmrestol=1e-12, close_eval=false)
check("  Mobility solve, separated disks, closed form vs native: ", maximum(maximum.(abs, U_close .- U_native)), 1e-12)

# At gap 1 the plan is not empty, but node quadrature is still accurate
xcm = [0.0 + 0.0im, 3.0 * cis(0.7)]
@test !isempty(PSH.close_eval_plan(xcm, θcm, D))
_, U_close  = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, D; eps=1e-12, gmrestol=1e-12)
_, U_native = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, D; eps=1e-12, gmrestol=1e-12, close_eval=false)
check("  Mobility solve, disks at gap 1, closed form vs native: ", maximum(maximum.(abs, U_close .- U_native)), 1e-12)

# ── Self operator: Fourier length chosen by decay (no aliasing for elongated ellipses) ──

a, b = sqrt(10), 1 / sqrt(10)
err = 0.0
for M in (8, 16, 24)
  local Ω = ellipse(a, b, M)
  ĝ = PSH.ellipse_A0_fourier(a, b; N0=2^15)
  for (k, (i, j)) in enumerate(((1, 1), (1, 2), (2, 2)))
    K = PSH.laplace3d_angular_matrix(ĝ[k], Ω.Lspan, Ω.Mspan, Ω.idx_even; tol=1e-15)
    global err = max(err, maximum(abs, Ω.K̂_G[i, j] - K))
  end
end
check("  Ellipse a/b = 10 self operator vs Nθ = 2^15: ", err, 1e-13)
