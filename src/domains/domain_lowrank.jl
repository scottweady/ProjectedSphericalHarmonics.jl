
using LinearAlgebra

"""
    domain_lowrank(γ, Mℓ, Mₘ)

Discretization of the image of the unit disk under a conformal map.

# Arguments
- `γ` : boundary of the domain
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Struct containing discretization information
"""
function domain_lowrank(γ, Mℓ::Int, Mₘ::Int)

  f, df = conformalmap(γ)

  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  ζvec = vec(ζ)
  z, dz = f.(ζ), abs2.(df.(ζ)) .* dζ

  M = (x, y) -> x == y ? df(x) : (f(x) - f(y)) / (x - y)

  K_S_fun = (x, y) -> abs.(M(x, y))^(-1) * abs2(df(y))
  K_N_fun = (x, y) -> abs.(M(x, y))^(-3) * abs2(df(y))

  K_S = aca(K_S_fun.(ζvec, transpose(ζvec)))
  K_N = aca(K_N_fun.(ζvec, transpose(ζvec)))

  c2_fun = (x, y) -> real(M(x, y))^2 / abs(M(x, y))^3 * abs2(df(y))
  s2_fun = (x, y) -> imag(M(x, y))^2 / abs(M(x, y))^3 * abs2(df(y))
  cs_fun = (x, y) -> real(M(x, y)) * imag(M(x, y)) / abs(M(x, y))^3 * abs2(df(y))

  K_G_fun = Array{Function}(undef, 2, 2, 2, 2)
  processed = Array{Bool}(undef, 2, 2, 2, 2)
  
  K_G_fun[1,1,1,1] = K_G_fun[1,1,2,2] = K_G_fun[2,2,1,1] = K_G_fun[2,2,2,2] = (x, y) -> c2_fun(x, y)
  processed[1,1,1,1] = processed[1,1,2,2] = processed[2,2,1,1] = processed[2,2,2,2] = false

  K_G_fun[1,2,2,1] = K_G_fun[2,1,1,2] = (x, y) -> s2_fun(x, y)
  processed[1,2,2,1] = processed[2,1,1,2] = false

  K_G_fun[1,2,1,2] = K_G_fun[2,1,2,1] = (x, y) -> -s2_fun(x, y)
  processed[1,2,1,2] = processed[2,1,2,1] = false

  K_G_fun[1,1,1,2] = K_G_fun[2,2,1,2] = K_G_fun[2,1,1,1] = K_G_fun[2,1,2,2] = (x, y) -> cs_fun(x, y)
  processed[1,1,1,2] = processed[2,2,1,2] = processed[2,1,1,1] = processed[2,1,2,2] = false

  K_G_fun[1,1,2,1] = K_G_fun[2,2,2,1] = K_G_fun[1,2,1,1] = K_G_fun[1,2,2,2] = (x, y) -> -cs_fun(x, y)
  processed[1,1,2,1] = processed[2,2,2,1] = processed[1,2,1,1] = processed[1,2,2,2] = false

  K_G = Array{Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(undef, 2, 2, 2, 2)
  
  for i = 1:2, j = 1:2, k = 1:2, l = 1:2
    if !processed[i,j,k,l]
      K_G[i,j,k,l] = aca(K_G_fun[i,j,k,l], ζ)
      processed[i,j,k,l] = true
    end
  end

  function apply_aca!(μ̂, Ω, K, Λ; parity=:even)

    # Temporary storage for ACA application
    tmp1 = zeros(ComplexF64, size(μ̂))
    tmp2 = zeros(ComplexF64, size(μ̂))

    ipsh_vec!(μ̂, Ω.D, parity=parity)

    for r = 1 : size(K[2], 2)
      tmp1 .= K[2][:, r] .* μ̂
      psh_vec!(tmp1, Ω.D, parity=parity)
      tmp1 .= Λ * tmp1
      ipsh_vec!(tmp1, Ω.D, parity=parity)
      tmp1 .*= K[1][:, r]
      tmp2 .+= tmp1
    end

    psh_vec!(tmp2, Ω.D, parity=parity)
    μ̂ .= tmp2

  end

  function Ŝ!(μ̂, Ω)
    shp = size(μ̂)
    μ̂ = vec(μ̂)
    apply_aca!(μ̂, Ω, Ω.K_S, Diagonal(vec(Ω.D.Ŝ)), parity=:even)
    μ̂ = reshape(μ̂, shp)
  end

  function Ĝ(μ̂, Ω)
    f̂ = (Ω.D.Ĝ[1,1] * μ̂[1] + Ω.D.Ĝ[1,2] * μ̂[2], Ω.D.Ĝ[2,1] * μ̂[1] + Ω.D.Ĝ[2,2] * μ̂[2])
    for i = 1:2, j = 1:2, k = 1:2, l = 1:2
      tmp .+= apply_aca(μ̂, Ω, Ω.K_G[i,j,k,l], Ω.D.Ĝ[j,k], parity=:even)
      f̂[i] .= psh_vec(tmp, Ω.D, parity=:even)
    end
    return f̂
  end
  
  return Domain(D, f, df, z, dz, K_S, K_N, K_G, [], [], [])
  
end

domain_lowrank(γ, M::Int) = domain_lowrank(γ, M, M)