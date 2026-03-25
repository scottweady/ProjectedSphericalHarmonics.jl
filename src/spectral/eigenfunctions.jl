
using AssociatedLegendrePolynomials: λlm as Plm_norm
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
    return 0 * ζ
  end

  r, θ = abs.(ζ), angle.(ζ)

  if any(r .> 1)
    error("Invalid input: |ζ| must be <= 1.")
  end

  return sqrt(2) * Plm_norm(l, abs(m), sqrt.(1 .- r.^2)) .* exp.(im * m * θ)

end

"""
    ∂ylm∂ζ(l, m, ζ)

Complex derivative of projected spherical harmonic of degree `l` and order `m` evaluated at points `ζ`.

# Arguments
- `l` : radial degree
- `m` : azimuthal order
- `ζ` : points on the disk

# Returns
- length(ζ) vector of function values

Warning: Ill-conditioned, use with caution.
"""
function ∂ylm∂ζ(l::Int, m::Int, ζ)
  c = m <= 0 ? 1.0 : -(l + m) * (l - m + 1.0)
  return (c / 2.0) * Nlm(l, m, l, m - 1) * ylm(l, m - 1, ζ) ./ sqrt.(1 .- abs2.(ζ))
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

  return 0.0im

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
  
  if abs(m) > l || abs(q) > p
    return 0.0
  end

  val = (0.5 * log((2l + 1) / (2π)) + 0.5 * (loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1))) - 
        (0.5 * log((2p + 1) / (2π)) + 0.5 * (loggamma(p - abs(q) + 1) - loggamma(p + abs(q) + 1)))

  return exp(val)

end

"""
    Clmn(l, m, n)

Coefficient in the expansion of projected spherical harmonics.

# Arguments
- `l` : radial degree
- `m` : azimuthal order
- `n` : ...

# Returns
- scalar coefficient
"""
function Clmn(l::Int, m::Int, n::Int)

  if abs(m) > l || abs(m + n) > l
    return 0.0
  end

  val = -(n + 2) * log(2) + 
    loggamma((l + m + 1) / 2) + loggamma((l - m - n + 1) / 2) - 
    loggamma((l - m) / 2 + 1) - loggamma((l + m + n) / 2 + 1) + 
    0.5 * (loggamma(l - m + 1) - loggamma(l + m + 1)) - 
    0.5 * (loggamma(l - (m + n) + 1) - loggamma(l + m + n + 1))

  val = exp(val)
  
  if m < 0
    val *= (-1)^abs(m)
  end
  
  if m + n < 0
    val *= (-1)^abs(m + n)
  end

  return val

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
    lm_index(D::Disk, l, m) -> Int

Linear (vectorised) index of mode (l, m) in the coefficient array for disk `D`.
"""
function lm2index(l::Int, m::Int, D::Disk)
    Nm = 2 * D.Mₘ + 1
    nl = l + 1
    nm = m >= 0 ? m + 1 : m + Nm + 1
    return nl + (nm - 1) * (D.Mℓ + 1)
end

"""
    index_lm(D::Disk, idx) -> (l, m)

Mode indices (l, m) corresponding to linear index `idx` in the coefficient array for disk `D`.
"""
function index2lm(idx::Int, D::Disk)
    Nm = 2D.Mₘ + 1
    nl = mod1(idx, D.Mℓ + 1)
    nm = cld(idx, D.Mℓ + 1)
    l = nl - 1
    m = nm <= D.Mₘ + 1 ? nm - 1 : nm - Nm - 1
    return l, m
end


export lm2index, index2lm
