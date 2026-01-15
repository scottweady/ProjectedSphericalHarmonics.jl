
using AssociatedLegendrePolynomials: Plm, λlm as Plm_norm
using SpecialFunctions

export ylm, Plm, λlm, Nlm, ∂ylm∂n

"""
    ylm(l, m, ζ)

Projected spherical harmonics of degree `l` and order `m` evaluated at points `ζ`.

# Arguments
- `l` : degree
- `m` : order
- `ζ` : points on the disk

# Returns
- projected spherical harmonic evaluated on the disk
"""
function ylm(l::Int, m::Int, ζ)

  # Get polar coordinates
  r, θ = abs.(ζ), angle.(ζ)

  # Return
  return sqrt(2) * ϕlm(l, m) * Plm_norm(l, abs(m), sqrt.(1 .- r.^2)) .* exp.(im * m * θ)

end

"""
    ylm(M, ζ)

Projected spherical harmonics evaluated at points `ζ` for degree `l, m <= M`.

# Arguments
- `M` : maximum degree and order
- `ζ` : points on the disk

# Returns
- projected spherical harmonics evaluated on the disk
"""
function ylm(M::Int, ζ)

  # Prepare mode indices
  Lspan = reshape(0 : M, 1, M + 1, 1)
  Mspan = reshape(-M : M, 1, 1, 2 * M + 1)
  
  # Get polar coordinates
  r, θ = abs.(ζ), angle.(ζ)

  # Compute associated Legendre polynomials
  P = Plm_norm(0 : M, 0 : M, sqrt.(1 .- r.^2))[:, 1, :, :]
  P = cat(P[:, :, (M + 1) : -1 : 2], P[:, :, 1 : (M + 1)]; dims=3) 

  # Return
  return sqrt(2) * ϕlm.(Lspan, Mspan) .* P .* exp.(im * Mspan .* θ)

end

"""
    ∂ylm∂r(M, ζ)

Radial derivative of projected spherical harmonics of degree `l` and order `m` evaluated at points `ζ` for `l, m <= M`.

# Arguments
- `M` : maximum degree and order
- `ζ` : points on the disk

# Returns
- radial derivative of projected spherical harmonics evaluated on the disk
"""
function ∂ylm∂r(M::Int, ζ)

    Lspan = reshape(0 : M, 1, M + 1, 1)
    Mspan = reshape(-M : M, 1, 1, 2 * M + 1)

    # Get polar coordinates
    r, θ = abs.(ζ), angle.(ζ)
    x = sqrt.(1 .- r.^2)

    # Compute associated Legendre polynomials
    P₀ = Plm(0 : M, 0 : M, x)[:, 1, :, :]
    P₁ = Plm(0 : (M + 1), 0 : M, x)[:, 1, 2 : end, :]

    # Compute derivative using recurrence relation
    dPdr = Array{Float64}(undef, length(ζ), M + 1, M + 1)

    for l = 0 : M
        for m = 0 : l
            nl, nm = l + 1, m + 1
            dPdr[:, nl, nm] = 1 ./ (x .* r) .* (-(l + 1) * x .* P₀[:, nl, nm] .+ (l - m + 1) * P₁[:, nl, nm])
        end
    end

    dPdr = ϕlm.(Lspan, Mspan) .* cat(dPdr[:, :, (M + 1) : -1 : 2], dPdr[:, :, 1 : (M + 1)]; dims=3)    
    
    return Nlm.(Lspan, Mspan) .* dPdr .* exp.(im * Mspan .* θ)

end

"""
    ∂ylm∂n(l, m, ζ)

Normal derivative of projected spherical harmonics of degree `l` and order `m` evaluated at points `ζ`.

# Arguments
- `l` : degree
- `m` : order
- `ζ` : points on the boundary of the disk

# Returns
- normal derivative of projected spherical harmonics evaluated on the boundary
"""
function ∂ylm∂n(l::Int, m::Int, ζ)

  # Check if points are on the boundary
  r = abs.(ζ)
  if any(abs.(r .- 1) .> 1e-12)
    error("Points ζ must be on the boundary of the disk (|ζ| = 1).")
  end

  θ = angle.(ζ)

  if mod(m + l, 2) == 0

    lpm = l + abs(m)
    lmm = l - abs(m)
    tmp = 0.5 * (loggamma(lpm + 1) + loggamma(lmm + 1)) - (loggamma(lpm/2 + 1) + loggamma(lmm/2 + 1)) - l * log(2)

    return (-1)^(Int(lpm/2)) * (l + lpm * lmm) * sqrt((2 * l + 1) / 2π) * exp(tmp) .* exp.(im * m * θ)

  end

  return Inf

end

"""
    ∂ylm∂n(M, ζ)

Normal derivative of projected spherical harmonics of degree `l` and order `m` evaluated at points `ζ` for `l, m <= M`.

# Arguments
- `M` : maximum degree and order
- `ζ` : points on the boundary of the disk

# Returns
- normal derivative of projected spherical harmonics evaluated on the boundary
"""
function ∂ylm∂n(M::Int, ζ)

  ∂Y∂n = Array{ComplexF64}(undef, length(ζ), M + 1, 2 * M + 1)

  for m = -M : M
    nm = (M + 1) + m
    for l = max(abs(m), 0) : M
      nl = l + 1
      ∂Y∂n[:, nl, nm] .= ∂ylm∂n(l, m, ζ)
    end
  end

  return ∂Y∂n

end

"""
    λlm(l, m)

    Generalized eigenvalues of projected spherical harmonics

# Arguments
- `l` : degree
- `m` : order

# Returns
- eigenvalue
"""
function λlm(l::Int, m::Int)
  return exp((loggamma((l + m + 1) / 2) + loggamma((l - m + 1) / 2)) - (loggamma((l + m + 2) / 2) + loggamma((l - m + 2) / 2)))
end

"""
    Nlm(l, m)

    Normalization factor for projected spherical harmonics

"""
function Nlm(l::Int, m::Int)
  val = 0.5 * log((2l + 1) / (2π)) + 0.5 * (loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1))
  return exp(val)
end

"""
    Nlm(l, m, p, q)

    Ratio of normalization factors for projected spherical harmonics

"""
function Nlm(l::Int, m::Int, p::Int, q::Int)
  val = (0.5 * im * m + 0.5 * log((2l + 1) / (2π)) + 0.5 * (loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1))) - 
        (0.5 * im * q + 0.5 * log((2p + 1) / (2π)) + 0.5 * (loggamma(p - abs(q) + 1) - loggamma(p + abs(q) + 1)))
  return exp(val)
end

""" 

    ϕlm(l, m)

    Phase factor for projected spherical harmonics

"""
function ϕlm(l::Int, m::Int)
  return m ≥ 0 ? 1.0 : (-1)^m
end