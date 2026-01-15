
using AssociatedLegendrePolynomials: Plm, λlm as Plm_norm
using SpecialFunctions

export ylm, λlm

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
  r, θ = abs.(ζ), angle.(ζ)
  return sqrt(2) * ϕlm(l, m) * Plm_norm(l, m, sqrt.(1 .- r.^2)) .* exp.(im * m * θ)
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

  r, θ = abs.(ζ), angle.(ζ)

  # Compute associated Legendre polynomials
  P = Plm_norm(0 : M, 0 : M, sqrt.(1 .- r.^2)) * sqrt(2)

  # Fill in negative values  
  Y = Array{ComplexF64}(undef, length(ζ), M + 1, 2 * M + 1)
  Y[:, :, (M + 1) : (2 * M + 1)] = P[:, 1, :, 1 : (M + 1)]
  Y[:, :, M : -1 : 1] = P[:, 1, :, 2 : (M + 1)]

  # Compute phase factor and angular part
  for m = -M : M

    nm = (M + 1) + m

    for l = max(abs(m), 0) : M
    
      nl = l + 1
      Y[:, nl, nm] .*= ϕlm(l, m) * exp.(im * m * θ)

    end
  end

  return Y

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

    r, θ = abs.(ζ), angle.(ζ)
    x = sqrt.(1 .- r.^2)

    # Compute associated Legendre polynomials
    P₀ = Plm(0 : M, 0 : M, x)[:, 1, :, :]
    P₁ = Plm(0 : (M + 1), 0 : M, x)[:, 1, 2 : end, :]
    dPdr = Array{Float64}(undef, length(ζ), M + 1, M + 1)

    # Compute derivative using recurrence relation
    for l = 0 : M
        for m = 0 : l
            nl, nm = l + 1, m + 1
            dPdr[:, nl, nm] = 1 ./ (x .* r) .* (-(l + 1) * x .* P₀[:, nl, nm] .+ (l - m + 1) * P₁[:, nl, nm])
        end
    end

    dYdr = Array{ComplexF64}(undef, length(ζ), M + 1, 2 * M + 1)
    dYdr[:, :, (M + 1) : (2 * M + 1)] = dPdr[:, :, 1 : (M + 1)]
    dYdr[:, :, M : -1 : 1] = dPdr[:, :, 2 : (M + 1)]
    
    # Compute phase factor and angular part
    for m = -M : M

        nm = (M + 1) + m
    
        for l = max(abs(m), 0) : M
        
            nl = l + 1

            tmp = 0.5 * (log((2l + 1) / 2π) + loggamma(l - abs(m) + 1) - loggamma(l + abs(m) + 1))
            dYdr[:, nl, nm] .*= ϕlm(l, m) * exp.(im * m * θ) .* exp(tmp)
    
        end

    end

    return dYdr

end

"""
    ∂ylm∂θ(M, ζ)

Angular derivative of projected spherical harmonics of degree `l` and order `m` evaluated at points `ζ` for `l, m <= M`.

# Arguments
- `M` : maximum degree and order
- `ζ` : points on the disk

# Returns
- angular derivative of projected spherical harmonics evaluated on the disk
"""
function ∂ylm∂θ(M::Int, ζ)

  ∂Y∂θ = ylm(M, ζ)
  
  for m = -M : M
    nm = (M + 1) + m
    ∂Y∂θ[:, :, nm] .= im * m .* ∂Y∂θ[:, :, nm]
  end

  return ∂Y∂θ

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

  θ = angle.(ζ)

  if mod(m + l, 2) == 0

    lpm = l + abs(m)
    lmm = l - abs(m)
    tmp = 0.5 * (loggamma(lpm + 1) + loggamma(lmm + 1)) - (loggamma(lpm/2 + 1) + loggamma(lmm/2 + 1)) - l * log(2)

    return ϕlm(l, m) * (-1)^(Int(lpm/2)) * (l + lpm * lmm) * sqrt((2 * l + 1) / 2π) * exp(tmp) .* exp.(im * m * θ)

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
    ϕlm(l, m)

Phase factor for projected spherical harmonics

# Arguments
- `l` : degree
- `m` : order

# Returns
- phase factor
"""
function ϕlm(l::Int, m::Int)
  return m >= 0 ? 1.0 : (-1)^m
end

"""
    μlm(l, m)

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