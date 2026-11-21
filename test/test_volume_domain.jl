println("Testing volume potentials on domains...")

const PSHv = ProjectedSphericalHarmonics

# 𝒱(1) of the ellipse x²/a² + y²/b² ≤ 1: inside, the quadratic (b x² + a y²)/(2(a + b)) + V(0) with
# V(0) = (ab/2) log((a + b)/2) - ab/4; outside, from the field 2∂V/∂z = (ab/c²)(z - √(z² - c²)), c² = a² - b²,
# continuous across the boundary
function V_ellipse(a, b, x)
  Vin(z) = (b * real(z)^2 + a * imag(z)^2) / (2(a + b)) + a * b / 2 * log((a + b) / 2) - a * b / 4
  (real(x) / a)^2 + (imag(x) / b)^2 <= 1 && return Vin(x)
  c = sqrt(a^2 - b^2)
  W(z) = (s = sqrt(z - c) * sqrt(z + c); (a * b / c^2) * (z^2 / 2 - z * s / 2 + c^2 / 2 * log(z + s)))
  return real(W(x)) - real(W(a + 0im)) + Vin(a + 0im)
end

# Reference by brute force: polar quadrature on the disk about the preimage ζ₀ (|ζ₀| < 1, away from the rim) of the
# true kernel, log|F(ζ₀ + te^{iφ}) - F(ζ₀)| or |…|, times u(F) and the Jacobian (t = Ru³ removes the singularity)
function V_brute(Ω, ζ₀, uf; kernel=:log, nφ=400, nt=60)
  ext = getfield(Ω, :ext)
  if ext isa PSHv.EllipseGeometry
    a, b = ext.a, ext.b
    F = x -> a .* real.(x) .+ im .* b .* imag.(x)
    J = x -> fill(a * b, length(x))
  elseif Ω.f isa PSHv.WinslowMap
    F = Ω.f
    J = x -> ((p, q) = PSHv._derivatives(Ω.f, x); abs2.(p) .- abs2.(q))
  else
    F = x -> Ω.f.(x)
    J = x -> (d = Ω.df.(x); abs2.(first.(d)) .- abs2.(last.(d)))
  end
  u, wu = legpts(nt, [0.0, 1.0])
  pts, wts = ComplexF64[], Float64[]
  for k = 0 : nφ - 1
    e = cis(2π * k / nφ)
    c = real(conj(ζ₀) * e)
    R = -c + sqrt(c^2 + 1 - abs2(ζ₀))
    for (ui, wi) in zip(u, wu)
      push!(pts, ζ₀ + R * ui^3 * e)
      push!(wts, 2π / nφ * R * ui^3 * 3R * ui^2 * wi)
    end
  end
  z, z₀ = F(pts), F([ζ₀])[1]
  k = kernel === :log ? log.(abs.(z .- z₀)) ./ 2π : abs.(z .- z₀)
  return sum(wts .* k .* uf.(z) .* J(pts))
end

uf(z) = exp(real(z)) * cos(imag(z)) + real(z)^2

# Ellipses (affine map, constant μ): 𝒱(1) against the closed form, on the grid and at points inside and outside; a
# smooth density against brute force, converging in M
for (a, b) in ((1.1, 0.91), (1.5, 0.67))
  local Ω = ellipse(a, b, 16)
  check("  𝒱(1) on the grid, $a × $b ellipse, M = 16: ", maximum(abs, 𝒱(1, Ω) .- V_ellipse.(a, b, Ω.z)), 1e-13)
  local xs = [0.3 + 0.2im, -0.9a + 0.1im, 0.999a + 0im, 1.6a + 0.1im, 3.0 + 2.0im]
  check("  𝒱(1) at points inside and outside, $a × $b ellipse: ", maximum(abs, 𝒱(xs, 1, Ω) .- V_ellipse.(a, b, xs)), 1e-13)
  ζs = [0.0im, 0.3 + 0.2im, -0.5 + 0.4im, 0.8im]
  ref = [V_brute(Ω, ζ, uf) for ζ in ζs]
  local errs = map((8, 16)) do M
    Ωm = ellipse(a, b, M)
    maximum(abs, 𝒱(a .* real.(ζs) .+ im .* b .* imag.(ζs), uf.(Ωm.z), Ωm) .- ref)
  end
  check("  𝒱(u) at points, $a × $b ellipse, M = 8: ", errs[1], 1e-5)
  check("  𝒱(u) at points, $a × $b ellipse, M = 16: ", errs[2], 1e-12)
  I = findall(vec(abs.(Ω.ζ)) .< 0.85)[1 : 9 : end]
  V, A = 𝒱(uf.(Ω.z), Ω), volume_matrix(Ω; kernel=:abs) * vec(uf.(Ω.z))
  check("  𝒱(u) on the grid vs brute force, $a × $b ellipse: ", maximum(abs(V[p] - V_brute(Ω, Ω.ζ[p], uf)) for p in I), 1e-12)
  check("  |x - y| kernel on the grid vs brute force, $a × $b ellipse: ",
        maximum(abs(A[p] - V_brute(Ω, Ω.ζ[p], uf; kernel=:abs)) for p in I), 1e-12)
end

# Δ𝒱[u] = u on an ellipse, by the disk's derivatives in the stretched coordinates (x = aξ, y = bη)
let a = 1.5, b = 0.67, Ω = ellipse(a, b, 24)
  D = Ω.D
  V = 𝒱(uf.(Ω.z), Ω)
  ΔV = ∂x(∂x(V, D), D) ./ a^2 .+ ∂y(∂y(V, D), D) ./ b^2
  check("  Δ𝒱[u] - u, 1.5 × 0.67 ellipse, M = 24: ", maximum(abs, ΔV .- uf.(Ω.z)), 1e-6)
end

# A disk-shaped domain (harmonic map, the identity) and the circular ellipse: 𝒱(1) = (r² - 1)/4, 𝒱(r²) = (r⁴ - 1)/16
let t = 2π .* (0 : 127) ./ 128, M = 12
  Ωh, Ωe, D = domain(cis.(t), M), ellipse(1.0, 1.0, M), disk(M)
  r² = abs2.(D.z)
  check("  𝒱(1), 𝒱(r²) on a disk-shaped domain: ",
        max(maximum(abs, 𝒱(1, Ωh) .- (r² .- 1) ./ 4), maximum(abs, 𝒱(r², Ωh) .- (r².^2 .- 1) ./ 16)), 1e-13)
  check("  𝒱 on a disk-shaped domain vs the circular ellipse: ", maximum(abs, 𝒱(uf.(D.z), Ωh) .- 𝒱(uf.(D.z), Ωe)), 1e-13)
  # Against the disk's own 𝒱 (a direct sum with the singularity subtracted, accurate to about 1e-3 here)
  check("  𝒱 on a disk-shaped domain vs 𝒱 on the disk: ", maximum(abs, 𝒱(uf.(D.z), Ωh) .- 𝒱(uf.(D.z), D)), 1e-12)
end

# Non-affine maps: a reparametrized ellipse (harmonic map) and a bean (Winslow map), against brute force at points of
# the grid and inside
let t = 2π .* (0 : 255) ./ 256
  s = t .+ 0.2 .* sin.(t)
  bean(s) = (1.2cos(s) - 0.15cos(2s)) + im * (0.8sin(s) + 0.45sin(s)^2 * (1 - cos(s)))
  ζs = [0.3 + 0.2im, -0.5 + 0.6im, 0.95im]
  for (label, γ, map, Ms, tols) in (("harmonic, reparametrized 1.5 × 0.67 ellipse", 1.5 .* cos.(s) .+ 0.67im .* sin.(s), :harmonic,
                                     (8, 16), (1e-5, 1e-11)),
                                    ("Winslow, bean", bean.(t), :winslow, (8, 16), (1e-4, 1e-8)))
    ref = nothing
    for (M, tol) in zip(Ms, tols)
      Ω = domain(γ, M; map=map)
      ref === nothing && (ref = [V_brute(Ω, ζ, uf) for ζ in ζs])
      check("  𝒱(u) at points inside, $label, M = $M: ", maximum(abs, 𝒱(Ω.f.(ζs), uf.(Ω.z), Ω) .- ref), tol)
    end
    Ω = domain(γ, Ms[end]; map=map)
    V = 𝒱(uf.(Ω.z), Ω)
    I = findall(vec(abs.(Ω.ζ)) .< 0.85)[1 : 11 : end]
    check("  𝒱(u) on the grid vs brute force, $label: ", maximum(abs(V[p] - V_brute(Ω, Ω.ζ[p], uf)) for p in I), tols[end])
    @test_throws ArgumentError 𝒱([5.0 + 0im], uf.(Ω.z), Ω)
  end
end

@test_throws ArgumentError volume_matrix(ellipse(1.5, 1.0, 4); kernel=:cubic)
