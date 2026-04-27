
using LinearAlgebra

struct Domain
  D
  f
  df
  z
  dz
  K_S
  K_N
  K_G
  K̂_S
  K̂_N
  K̂_G
end

"""
    domain(γ, Mℓ, Mₘ)

Discretization of the image of the unit disk under a conformal map.

# Arguments
- `γ` : boundary of the domain
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Domain containing discretization information
"""
function domain(γ, Mℓ::Int, Mₘ::Int)

  f, df = conformalmap(γ)
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  z, z′ = f.(ζ), df.(ζ)  
  dz = abs2.(z′) .* dζ

  ζvec, zvec, z′vec = vec(ζ), vec(z), vec(z′)
  Jvec = ComplexF64.(abs2.(z′vec))

  M = (zvec .- transpose(zvec)) ./ (ζvec .- transpose(ζvec))
  M[diagind(M)] .= z′vec

  ρ, α = ComplexF64.(abs.(M)), angle.(M)
  ϕ = angle.(ζvec .- transpose(ζvec))

  K_S = ρ.^(-1) .* transpose(Jvec)
  K_N = ρ.^(-3) .* transpose(Jvec)

  even, odd = findall(vec(D.even)), findall(vec(D.odd))

  P, Q = psh_matrix(D), ipsh_matrix(D)
  K_S .*= Q[:,even] * (D.Ŝ[even] .* P[even, :])
  K_N .*= Q[:,odd] * (D.N̂[odd] .* P[odd, :])

  Ŝcc = laplace3d_angular_matrix(ϕ -> cos(ϕ)^2, D.Lspan, D.Mspan, even)
  Ŝsc = laplace3d_angular_matrix(ϕ -> cos(ϕ) * sin(ϕ), D.Lspan, D.Mspan, even)
  Ŝss = laplace3d_angular_matrix(ϕ -> sin(ϕ)^2, D.Lspan, D.Mspan, even)

  K_cc = (Q[:,even] * Ŝcc * P[even, :]) .* ρ.^(-1) .* transpose(Jvec)
  K_sc = (Q[:,even] * Ŝsc * P[even, :]) .* ρ.^(-1) .* transpose(Jvec)
  K_ss = (Q[:,even] * Ŝss * P[even, :]) .* ρ.^(-1) .* transpose(Jvec)

  K̂_S = P[even,:] * K_S * Q[:,even]
  K̂_N = P[odd,:] * K_N * Q[:,odd]
  
  K_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K_G[1,1] = @. K_S + ((K_cc * cos(α)^2 - 2 * K_sc * cos(α) * sin(α) + K_ss * sin(α)^2))
  K_G[1,2] = @. (K_sc * (cos(α)^2 - sin(α)^2) + (K_cc - K_ss) * cos(α) * sin(α))
  K_G[2,1] = K_G[1,2]
  K_G[2,2] = @. K_S + ((K_cc * sin(α)^2 + 2 * K_sc * cos(α) * sin(α) + K_ss * cos(α)^2))


  K̂_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K̂_G[1,1] = P[even,:] * K_G[1, 1] * Q[:,even]
  K̂_G[1,2] = P[even,:] * K_G[1, 2] * Q[:,even]
  K̂_G[2,1] = K̂_G[1,2]
  K̂_G[2,2] = P[even,:] * K_G[2, 2] * Q[:,even]

  return Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G)
  
end

domain(γ, M::Int) = domain(γ, M, M)

const _domain_fields = fieldnames(Domain)
Base.getproperty(Ω::Domain, s::Symbol) =
  s ∈ _domain_fields ? getfield(Ω, s) : getproperty(getfield(Ω, :D), s)