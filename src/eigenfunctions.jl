
using AssociatedLegendrePolynomials: Plm, λlm as Plm_norm
using SpecialFunctions: loggamma

"""
    ylm(l, m, ζ)

Projected spherical harmonic of degree `l` and order `m` evaluated at points `ζ`.

# Arguments
- `l` : radial degree
- `m` : azimuthal order
- `ζ` : points on the disk

# Returns
- length(ζ) vector of function values
"""
function ylm(l::Int, m::Int, ζ)

  if abs(m) > l
    error("Invalid mode: |m| must be <= l.")
  end

  r, θ = abs.(ζ), angle.(ζ)

  if any(r .> 1)
    error("Invalid input: |ζ| must be <= 1.")
  end

  return sqrt(2) * ϕlm(l, m) * Plm_norm(l, abs(m), sqrt.(1 .- r.^2)) .* exp.(im * m * θ)
end

"""
    ∂ylm∂n(l, m)

Normal derivative of projected spherical harmonics of degree `l` and order `m` evaluated at `θ=0'.

# Arguments
- `l` : radial degree
- `m` : azimuthal order

# Returns
- scalar function value
"""
function ∂ylm∂n(l::Int, m::Int)

  if mod(m + l, 2) == 0

    lpm = l + abs(m)
    lmm = l - abs(m)
    tmp = 0.5 * (loggamma(lpm + 1) + loggamma(lmm + 1)) - (loggamma(lpm/2 + 1) + loggamma(lmm/2 + 1)) - l * log(2)

    return (-1)^(Int(lpm/2)) * (l + lpm * lmm) * sqrt((2 * l + 1) / 2π) * exp(tmp)

  end

  return 0.0

end

"""
    Nlm(l, m)

    Normalization factor for projected spherical harmonics

# Arguments
- `l` : degree
- `m` : order

# Returns
- scalar normalization factor
"""
function Nlm(l::Int, m::Int)

  if abs(m) > l
      return 0.0
  end

  return exp(0.5 * log((2l + 1) / (2π)) + 0.5 * (loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1)))

end

"""
    Nlm(l, m, p, q)

    Ratio of normalization factors for projected spherical harmonics

# Arguments
- `l` : degree of numerator
- `m` : order of numerator
- `p` : degree of denominator
- `q` : order of denominator

# Returns
- scalar ratio of normalization factors
"""
function Nlm(l::Int, m::Int, p::Int, q::Int)
  if l < abs(m) || p < abs(q)
      return 0.0
  end
  val = (0.5 * im * m + 0.5 * log((2l + 1) / (2π)) + 0.5 * (loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1))) - 
        (0.5 * im * q + 0.5 * log((2p + 1) / (2π)) + 0.5 * (loggamma(p - abs(q) + 1) - loggamma(p + abs(q) + 1)))
  return exp(val)
end

""" 

    ϕlm(l, m)

    Phase factor for projected spherical harmonics

# Arguments
- `l` : degree
- `m` : order

# Returns
- scalar phase factor
"""
function ϕlm(l::Int, m::Int)
  return m ≥ 0 ? 1.0 : (-1)^m
end

"""
    λlm(l, m)

    Generalized eigenvalues of projected spherical harmonics

# Arguments
- `l` : degree
- `m` : order

# Returns
- scalar eigenvalue
"""
function λlm(l::Int, m::Int)

  if abs(m) > l
      return 0.0
  end

  return exp((loggamma((l + m + 1) / 2) + loggamma((l - m + 1) / 2)) - (loggamma((l + m + 2) / 2) + loggamma((l - m + 2) / 2)))

end

"""
    ∂ylm∂r(M, r, θ)

Radial derivative of projected spherical harmonics up to order `M` evaluated on product grid `r * exp.(im * θ)`.

# Arguments
- `M` : maximum degree and order
- `r` : radial coordinates
- `θ` : angular coordinates

# Returns
- length(r) * length(θ) by (M + 1) * (2M + 1) matrix of function values

Warning: Ill-conditioned, use with caution.
"""
function ∂ylm∂r(Mr::Int, Mθ::Int, r, θ)

  # Get polar coordinates
  r, θ = reshape(vec(r), :, 1), reshape(vec(θ), 1, :)

  # Prepare mode indices
  Lspan = 0 : Mr
  Mspan = [0 : Mθ; -Mθ : -1]
  Lspan = reshape(Lspan, 1, 1, length(Lspan), 1)
  Mspan = reshape(Mspan, 1, 1, 1, length(Mspan))

  # Define argument
  x = sqrt.(1 .- r.^2)

  # Compute associated Legendre polynomials
  P = Plm(0 : Mr + 1, 0 : Mθ, x)
  Pℓ, Pℓ₊₁ = P[:, :, 1 : end - 1, :], P[:, :, 2 : end, :]
  
  # Compute derivative using recurrence relation
  dPdr = Array{Float64}(undef, length(r), 1, Mr + 1, Mθ + 1)
  for (nl, l) in enumerate(0 : Mr)
      for (nm, m) in enumerate(0 : Mθ) #positive modes only
          if abs(m) > l || l + m % 2 == 1
            continue
          end
          dPdr[:, :, nl, nm] = 1 ./ (x .* r) .* (-(l + 1) * x .* Pℓ[:, :, nl, nm] .+ (l - m + 1) * Pℓ₊₁[:, :, nl, nm])
      end
  end

  # Compute radial derivative of projected spherical harmonics
  ∂Y∂r = ϕlm.(Lspan, Mspan) .* Nlm.(Lspan, Mspan) .* dPdr[:, :, :, abs.(vec(Mspan)) .+ 1] .* exp.(im * Mspan .* θ)
  ∂Y∂r = reshape(∂Y∂r, length(r) * length(θ), length(Lspan) * length(Mspan))

  return ∂Y∂r

end
