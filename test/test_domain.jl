using SpecialFunctions: ellipe, ellipk

println("Testing domain operators...")

# Ellipse parameters
a, b = 1.5, 1
M = 32
m = 1 - (b/a)^2

for (label, Ω, tolσ, tolF) in [("angle", ellipse(a, b, M), 1e-9, 1e-11), ("conformal", domain(range(0, 2π, 128)[1:end-1] .|> θ -> a*cos(θ) + im*b*sin(θ), M), 1e-3, 1e-7)]

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
  check("  𝒮_st(𝒮_st⁻¹(u)) on a conformal domain: ", max(maximum(abs, u[1] .- 1), maximum(abs, u[2])), 1e-10)
  σ = 𝒮⁻¹(1.0, Ω)
  check("  𝒮(𝒮⁻¹(1)) on a conformal domain: ", maximum(abs, 𝒮(σ, Ω) .- 1), 1e-10)
end
