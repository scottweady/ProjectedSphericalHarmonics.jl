
"""
    ellipse(Mℓ, Mₘ)

Discretization of an ellipse under affine transformation of the unit disk.

# Arguments
- `a` : semi-major axis
- `b` : semi-minor axis
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Domain containing discretization information
"""
function ellipse(a, b, Mℓ::Int, Mₘ::Int)

  # Polar coordinate discretization of the ellipse
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  f(z) = a * real(z) + im * b * imag(z)
  df(z) = inf
  z, dz = f.(ζ), a * b * dζ

  # Azimuthal and radial mode numbers
  Lspan = 0 : Mℓ
  Mspan = transpose([0 : Mₘ; -Mₘ : -1])

  # Short-hand names for convenience
  L, M = Lspan, Mspan

  # Even and odd boolean
  even = (L .+ M) .% 2 .== 0 .&& abs.(M) .<= L
  idx_even = findall(vec(even))
  
  # Angular representation of singularity swap
  ρ(θ) = sqrt(a^2 * cos(θ)^2 + b^2 * sin(θ)^2)
  g(θ) = a * b / ρ(θ)

  # Single layer operator in coefficient space
  K̂_S = laplace3d_angular_matrix(g, L, M, idx_even)
  
  # Stokes operator in coefficient space
  g11(θ) = (1 / ρ(θ) + a^2 * cos(θ)^2 / ρ(θ)^3) * a * b
  g12(θ) = (a * b * sin(θ) * cos(θ) / ρ(θ)^3) * a * b
  g22(θ) = (1 / ρ(θ) + b^2 * sin(θ)^2 / ρ(θ)^3) * a * b

  K̂_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K̂_G[1, 1] = laplace3d_angular_matrix(g11, L, M, idx_even)
  K̂_G[1, 2] = laplace3d_angular_matrix(g12, L, M, idx_even)
  K̂_G[2, 1] = K̂_G[1, 2]
  K̂_G[2, 2] = laplace3d_angular_matrix(g22, L, M, idx_even)

  # Return domain struct
  return Domain(D, f, df, z, dz, [], [], [], K̂_S, [], K̂_G)

end

# Convenient method for equal radial and azimuthal discretization
ellipse(a, b, M::Int) = ellipse(a, b, M, M)
