
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
function ellipse(a, b, Mℓ::Int, Mₘ::Int; tol=1e-15)

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
  K̂_S = laplace3d_angular_matrix(g, L, M, idx_even; tol=tol)
  
  # Stokes operator in coefficient space. The angular kernels are the entries of
  # A₀(θ) = ab (I + γ̂γ̂)/|γ₀|, γ₀ = (a cos θ, b sin θ), whose Fourier coefficients
  # are shared with the close evaluation of the single layer off the ellipse
  ĝ11, ĝ12, ĝ22 = ellipse_A0_fourier(a, b; N0=nextpow(2, 4Mₘ + 4))

  K̂_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K̂_G[1, 1] = laplace3d_angular_matrix(ĝ11, L, M, idx_even; tol=tol)
  K̂_G[1, 2] = laplace3d_angular_matrix(ĝ12, L, M, idx_even; tol=tol)
  K̂_G[2, 1] = K̂_G[1, 2]
  K̂_G[2, 2] = laplace3d_angular_matrix(ĝ22, L, M, idx_even; tol=tol)

  ext = EllipseGeometry(a, b, (ĝ11, ĝ12, ĝ22))

  # Return domain struct
  return Domain(D, f, df, z, dz, [], [], [], K̂_S, [], K̂_G, ext)

end

# Convenient method for equal radial and azimuthal discretization
ellipse(a, b, M::Int; tol=1e-15) = ellipse(a, b, M, M; tol=tol)

"""
    ellipse_A0_fourier(a, b; N0=64, tol=1e-15)

Fourier coefficients (FFT order) of the entries (11, 12, 22) of
A₀(θ) = ab (I + γ̂γ̂)/|γ₀|, γ₀ = (a cos θ, b sin θ), with the number of points chosen
by their decay (see `fourier_coeffs`).
"""
function ellipse_A0_fourier(a, b; N0=64, tol=1e-15)
  ρ(θ) = sqrt(a^2 * cos(θ)^2 + b^2 * sin(θ)^2)
  g11(θ) = (1 / ρ(θ) + a^2 * cos(θ)^2 / ρ(θ)^3) * a * b
  g12(θ) = (a * b * sin(θ) * cos(θ) / ρ(θ)^3) * a * b
  g22(θ) = (1 / ρ(θ) + b^2 * sin(θ)^2 / ρ(θ)^3) * a * b
  return fourier_coeffs((g11, g12, g22); N0=N0, tol=tol)
end

"""
    EllipseGeometry(a, b)

Geometry needed to evaluate the Stokes single layer of an ellipse off the ellipse:
half axes `a`, `b` and the Fourier coefficients Â_n of A₀ (see `ellipse_A0_fourier`) for
even n = -nmax, …, nmax, truncated where they fall below `tol` relative to Â_0.
"""
struct EllipseGeometry
  a::Float64
  b::Float64
  ns::Vector{Int}
  Â::NTuple{3, Vector{ComplexF64}}    # entries (11, 12, 22) of Â_n, n ∈ ns
end

function EllipseGeometry(a, b, ĝ::NTuple{3}; tol=1e-15)
  N = length(ĝ[1])
  idx(n) = n >= 0 ? n + 1 : n + N + 1
  scale = maximum(abs(gk[1]) for gk in ĝ)
  nmax = maximum(n for n in 0:2:N÷2-1 if maximum(abs(gk[idx(n)]) + abs(gk[idx(-n)]) for gk in ĝ) > tol * scale)
  ns = collect(-nmax:2:nmax)
  return EllipseGeometry(a, b, ns, map(gk -> gk[idx.(ns)], ĝ))
end

EllipseGeometry(a, b; tol=1e-15) = EllipseGeometry(a, b, ellipse_A0_fourier(a, b; tol=tol); tol=tol)
