using SpecialFunctions: ellipe, ellipk

println("Testing domain operators...")

# Ellipse parameters
a, b = 1.5, 1
M = 32
m = 1 - (b/a)^2

γₑ = range(0, 2π, 128)[1:end-1] .|> θ -> a*cos(θ) + im*b*sin(θ)
for (label, Ω, tolσ, tolF) in [("angle", ellipse(a, b, M), 1e-9, 1e-11),
                               ("conformal", domain(γₑ, M; map=:conformal), 1e-3, 1e-7),
                               ("harmonic", domain(γₑ, M), 1e-9, 1e-11)]

  local x, y = real.(Ω.z), imag.(Ω.z)

  # Single layer for uniform charge
  σ = 𝒮⁻¹(1.0, Ω)
  local err = maximum(abs.(σ .* sqrt.(1 .- (x/a).^2 .- (y/b).^2) .- 2 / (b * ellipk(m))))
  check("  Max error in 𝒮⁻¹ for uniform charge ($label): ", err, tolσ)

  # Drag force on ellipse translating in x direction
  f = 𝒮_st⁻¹((1.0, 0.0), Ω)
  F = integral(f[1], Ω)
  Fex = 4π / (((2a^2 - b^2) * ellipk(m) - a^2 * ellipe(m)) / (a * (a^2 - b^2)))
  check("  Max error in drag force ($label): ", abs(F - Fex), tolF)

end

println("Testing transform matrices and the construction of conformal domains...")

const PSH = ProjectedSphericalHarmonics
relerr(A, B) = maximum(abs, A - B) / maximum(abs, B)

# psh_matrix and ipsh_matrix against the transforms, with Mℓ ≠ Mₘ
let D = disk(9, 6)
  u = exp.(D.z) .* (1 .+ abs2.(D.z))
  for parity in (:even, :odd)
    P, Q = psh_matrix(D; parity=parity), ipsh_matrix(D; parity=parity)
    û = psh(u, D; parity=parity)
    idx = findall(vec(getproperty(D, parity)))
    check("  psh_matrix vs psh ($parity): ", relerr(P * vec(u), vec(û)), 1e-13)
    check("  ipsh_matrix vs ipsh ($parity): ", relerr(Q[:, idx] * vec(û)[idx], vec(ipsh(û, D; parity=parity))), 1e-13)
  end
end

# The grid and coefficient operators that `domain` builds from the factors of the transforms, against the
# dense products Q Ŝ P and P K Q
let D = disk(9, 6)
  P, Q = psh_matrix(D), ipsh_matrix(D)
  Y, _ = PSH._psh_factors(D)
  dw = vec(D.dw)
  for parity in (:even, :odd)
    idx = findall(vec(getproperty(D, parity)))
    Ŝ = parity == :even ? PSH.laplace3d_angular_matrix(ϕ -> cos(ϕ) * sin(ϕ), D.Lspan, D.Mspan, idx) :
                          PSH.sparse(PSH.Diagonal(D.K̂_N[idx]))
    A = PSH._grid_operator(Ŝ, idx, Y, dw)
    check("  grid operator vs Q Ŝ P ($parity): ", relerr(A, Q[:, idx] * Ŝ * P[idx, :]), 1e-13)
    z = vec(D.z)
    K = A .* (1 .+ abs2.(z) .+ transpose(real.(z)))         # not of the form Q Ŝ P
    check("  coefficient operator vs P K Q ($parity): ", relerr(PSH._coefficient_operator(K, idx, Y, dw), P[idx, :] * K * Q[:, idx]), 1e-13)
  end
end

# A Domain factors each operator once; repeated solves reuse it
let θ = range(0, 2π, 129)[1:end-1]
  Ω = domain(1.5 .* cos.(θ) .+ im .* (sin.(θ) .+ 0.3 .* sin.(2θ)), 12)
  f = 𝒮_st⁻¹((1.0, 0.0), Ω)
  F = PSH.factorization(Ω, Ω.K̂_G)
  f2 = 𝒮_st⁻¹((1.0, 0.0), Ω)
  @test PSH.factorization(Ω, Ω.K̂_G) === F
  @test f2 == f
  u = 𝒮_st(f, Ω)
  check("  𝒮_st(𝒮_st⁻¹(u)) on a domain: ", max(maximum(abs, u[1] .- 1), maximum(abs, u[2])), 1e-10)
  σ = 𝒮⁻¹(1.0, Ω)
  check("  𝒮(𝒮⁻¹(1)) on a domain: ", maximum(abs, 𝒮(σ, Ω) .- 1), 1e-10)
end

# domain(γ, M; map=:conformal) builds its operators column by column without forming the grid operators, with the Stokes
# directions from the kernels e^{±2iϕ}/r. Against the dense construction: Q Ŝ P of the disk's operators
# times the smooth factors, with cos², sin cos, sin² of the direction rotated by the angle of the map
let θ = range(0, 2π, 129)[1:end-1], M = 8
  Ω = domain(1.5 .* cos.(θ) .+ im .* (sin.(θ) .+ 0.3 .* sin.(2θ)), M; map=:conformal)
  D = Ω.D
  even, odd = findall(vec(D.even)), findall(vec(D.odd))
  P, Q = psh_matrix(D), ipsh_matrix(D)
  ζ, z, z′ = vec(D.ζ), vec(Ω.z), vec(Ω.df.(D.ζ))
  m = (z .- transpose(z)) ./ (ζ .- transpose(ζ))
  m[PSH.diagind(m)] .= z′
  ρ, α = abs.(m), angle.(m)
  H = transpose(abs2.(z′)) ./ ρ
  A(Ŝ, idx) = Q[:, idx] * Ŝ * P[idx, :]
  ang(g) = A(PSH.laplace3d_angular_matrix(g, D.Lspan, D.Mspan, even), even)
  K_S = H .* A(PSH.Diagonal(D.K̂_S[even]), even)
  K_N = H ./ ρ.^2 .* A(PSH.Diagonal(D.K̂_N[odd]), odd)
  cc, sc, ss = ang(ϕ -> cos(ϕ)^2), ang(ϕ -> cos(ϕ) * sin(ϕ)), ang(ϕ -> sin(ϕ)^2)
  c, s = cos.(α), sin.(α)
  G11 = K_S .+ H .* (cc .* c.^2 .- 2 .* sc .* c .* s .+ ss .* s.^2)
  G12 = H .* (sc .* (c.^2 .- s.^2) .+ (cc .- ss) .* c .* s)
  G22 = K_S .+ H .* (cc .* s.^2 .+ 2 .* sc .* c .* s .+ ss .* c.^2)
  proj(K, idx) = P[idx, :] * K * Q[:, idx]
  check("  conformal K̂_S vs dense construction: ", relerr(Ω.K̂_S, proj(K_S, even)), 1e-12)
  check("  conformal K̂_N vs dense construction: ", relerr(Ω.K̂_N, proj(K_N, odd)), 1e-12)
  check("  conformal K̂_G vs dense construction: ",
        maximum(relerr(Ω.K̂_G[k...], proj(K, even)) for (k, K) in (((1, 1), G11), ((1, 2), G12), ((2, 2), G22))), 1e-12)
  # The grid operators, formed on access
  check("  conformal grid operators K_S, K_G vs dense construction: ",
        max(relerr(Ω.K_S, K_S), relerr(Ω.K_G[1, 1], G11), relerr(Ω.K_G[1, 2], G12), relerr(Ω.K_G[2, 2], G22)), 1e-10)
  @test Ω.K_G[2, 1] === Ω.K_G[1, 2] || Ω.K_G[2, 1] == Ω.K_G[1, 2]
end

# Harmonic maps (the default): the harmonic extension of the boundary samples
println("Testing harmonic maps...")
let a = 2.0, b = 0.5, t = 2π .* (0 : 255) ./ 256
  m = 1 - (b / a)^2
  Cex = 4π * a / ellipk(m)
  Fex = 4π / (((2a^2 - b^2) * ellipk(m) - a^2 * ellipe(m)) / (a * (a^2 - b^2)))
  caperr(Ω) = abs(integral(𝒮⁻¹(1.0, Ω), Ω) - Cex) / Cex
  dragerr(Ω) = abs(integral(𝒮_st⁻¹((1.0, 0.0), Ω)[1], Ω) - Fex) / Fex

  # γ = a cos t + i b sin t: the affine map, the operators of `ellipse`
  Ω, Ωe = domain(a .* cos.(t) .+ im .* b .* sin.(t), 16), ellipse(a, b, 16)
  check("  harmonic K̂_S vs ellipse (4:1): ", relerr(Ω.K̂_S, Ωe.K̂_S), 1e-12)
  check("  harmonic K̂_G vs ellipse (4:1): ", maximum(relerr(Ω.K̂_G[k], Ωe.K̂_G[k]) for k in eachindex(Ωe.K̂_G)), 1e-12)

  # Another parametrization: a map that is not affine, converging spectrally (the conformal map is off by ~40%)
  s = t .+ 0.2 .* sin.(2t)
  Ω = domain(a .* cos.(s) .+ im .* b .* sin.(s), 24)
  check("  harmonic capacitance, 4:1 ellipse, reparametrized, M = 24: ", caperr(Ω), 1e-11)
  check("  harmonic drag, 4:1 ellipse, reparametrized, M = 24: ", dragerr(Ω), 1e-10)
end

# Against the conformal map where that is accurate, and independent of the parametrization
let t = 2π .* (0 : 255) ./ 256, curve(s) = (1 + 0.15cos(3s)) * cis(s)
  cap(Ω) = integral(𝒮⁻¹(1.0, Ω), Ω)
  drag(Ω) = integral(𝒮_st⁻¹((1.0, 0.0), Ω)[1], Ω)
  Ωc, Ωh = domain(curve.(t), 32; map=:conformal), domain(curve.(t), 24)
  Ωh2 = domain(curve.(t .+ 0.2 .* sin.(t)), 24)
  check("  harmonic vs conformal, three-lobed capacitance: ", abs(cap(Ωh) - cap(Ωc)) / abs(cap(Ωc)), 1e-10)
  check("  harmonic vs conformal, three-lobed drag: ", abs(drag(Ωh) - drag(Ωc)) / abs(drag(Ωc)), 1e-10)
  check("  harmonic, two parametrizations, three-lobed capacitance: ", abs(cap(Ωh2) - cap(Ωh)) / abs(cap(Ωh)), 1e-10)
end

# A harmonic map that folds is an error, as is an unknown map
let t = 2π .* (0 : 127) ./ 128
  @test_throws ArgumentError domain((1 .+ 0.6 .* cos.(3t)) .* cis.(t), 12; map=:harmonic)
  @test_throws ArgumentError domain(cis.(t), 8; map=:mobius)
end

# Winslow maps: the inverse map is harmonic, one-to-one for non-convex shapes
println("Testing Winslow maps...")
let t = 2π .* (0 : 255) ./ 256
  cap(Ω) = integral(𝒮⁻¹(1.0, Ω), Ω)
  drag(Ω) = integral(𝒮_st⁻¹((1.0, 0.0), Ω)[1], Ω)
  torque(Ω) = (x = real.(Ω.z); y = imag.(Ω.z); f = 𝒮_st⁻¹((-y, x), Ω); integral(-y .* f[1] .+ x .* f[2], Ω))

  # The map: its derivatives, and its mean-value divided differences are exact
  pinwheel(s) = (r = 1 + 0.3cos(5s); r * cis(s + 0.6(r - 1)))
  F = PSH.winslowmap(pinwheel.(t))
  check("  Winslow map reproduces the boundary: ", maximum(abs.(F(cis.(t)) .- pinwheel.(t))), 1e-8)
  x, h = 0.6 + 0.3im, 1e-6
  fdx, fdy = (F(x + h) - F(x - h)) / 2h, (F(x + im * h) - F(x - im * h)) / 2h
  a, b = PSH._derivatives(F, [x])
  check("  Winslow map ∂F, ∂̄F vs finite differences: ", max(abs(a[1] - (fdx - im * fdy) / 2), abs(b[1] - (fdx + im * fdy) / 2)), 1e-8)
  ζ = vec(disk(6).ζ); P, Q = PSH._mean_value_tables(F, ζ); Fζ = F(ζ)
  check("  Winslow mean values: F(ζ) - F(η) = PΔζ + Q conj(Δζ): ",
        maximum(abs(Fζ[p] - Fζ[q] - (P[p, q] * (ζ[p] - ζ[q]) + Q[p, q] * conj(ζ[p] - ζ[q]))) for p in eachindex(ζ), q in eachindex(ζ)), 1e-13)

  # 4:1 ellipse sampled as a cos t + i b sin t: the Winslow map is affine, the operators those of `ellipse`
  Ω, Ωe = domain(2 .* cos.(t) .+ 0.5im .* sin.(t), 12; map=:winslow), ellipse(2.0, 0.5, 12)
  check("  Winslow K̂_S, K̂_G vs ellipse (4:1): ", max(relerr(Ω.K̂_S, Ωe.K̂_S), maximum(relerr(Ω.K̂_G[k], Ωe.K̂_G[k]) for k in eachindex(Ωe.K̂_G))), 1e-10)

  # The bean, against the harmonic map
  bean(s) = (1.2cos(s) - 0.15cos(2s)) + im * (0.8sin(s) + 0.45sin(s)^2 * (1 - cos(s)))
  Ωw, Ωh = domain(bean.(t), 16; map=:winslow), domain(bean.(t), 24)
  check("  Winslow vs harmonic, bean capacitance: ", abs(cap(Ωw) - cap(Ωh)) / abs(cap(Ωh)), 1e-8)
  check("  Winslow vs harmonic, bean drag: ", abs(drag(Ωw) - drag(Ωh)) / abs(drag(Ωh)), 1e-8)

  # A chiral pinwheel, on which the harmonic map folds: against M = 32 (capacitance, drag, torque). The default map
  # (:auto) falls back to the Winslow map there, and keeps the harmonic map where it does not fold
  @test_throws ArgumentError domain(pinwheel.(t), 16; map=:harmonic)
  Ωa = @test_logs (:info, r"Winslow") domain(pinwheel.(t), 12)
  @test getfield(Ωa, :K_S).map isa PSH._TabulatedDD
  @test Ωa.K̂_S ≈ domain(pinwheel.(t), 12; map=:winslow).K̂_S
  @test getfield(@test_logs(domain(bean.(t), 12)), :K_S).map isa PSH._HarmonicMap
  Ω = domain(pinwheel.(t), 24; map=:winslow)
  ref = (8.7611126079, 5.8769088666, 7.9022226086)
  check("  Winslow pinwheel, M = 24 vs M = 32: ",
        maximum(abs.((cap(Ω), drag(Ω), torque(Ω)) .- ref) ./ ref), 1e-5)
end
