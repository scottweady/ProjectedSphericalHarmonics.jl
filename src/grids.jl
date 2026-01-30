
using FastGaussQuadrature: gausslegendre

"""
    legpts(N, dom)

Return `N` Gauss–Legendre quadrature points and weights on the interval `dom = [a, b]`.

# Arguments
- `N` : number of points
- `dom` : two-element vector `[a, b]` defining the interval

# Returns
- `x` : points
- `dx` : weights
"""
function legpts(N::Int, dom::AbstractVector{T}=[-1.0, 1.0]) where T<:Real
    x, dx = gausslegendre(N)
    x = dom[1] * (1 .- x) / 2 .+ dom[2] * (1 .+ x) / 2
    dx = dx * (dom[2] - dom[1]) / 2
    return x, dx
end

"""
    trigpts(N, dom)

Return `N` equispaced points and weights on the interval `dom = [a, b]`.

# Arguments
- `N` : number of points
- `dom` : two-element vector `[a, b]` defining the interval

# Returns
- `x` : points
- `dx` : weights
"""
function trigpts(N::Int, dom::AbstractVector{T}=[0.0, 2π]) where T<:Real

  x = range(dom[1], dom[2], N + 1)
  dx = diff(x)
  x = x[1:N]
  return x, dx

end

"""
    diskpts(Nr, Nθ, rspan=[0, 1], θspan=[0, 2π])

Return quadrature points and weights on the unit disk using `Nr` radial and `Nθ` angular points.

# Arguments
- `Nr` : number of radial points
- `Nθ` : number of angular points
- `rspan` : two-element vector `[r_min, r_max]` defining radial interval
- `θspan` : two-element vector `[θ_min, θ_max]` defining angular interval

# Returns
- `r` : radial points
- `θ` : angular points
- `dr` : radial weights
- `dθ` : angular weights
"""
function diskpts(Nr::Int, Nθ::Int, rspan::AbstractVector{T}=[0.0, 1.0], θspan::AbstractVector{T}=[0, 2π]) where T<:Real

  # Radial grid
  s, ds = legpts(Nr, sqrt.(1 .- rspan.^2))
  s, ds = reshape(s, :, 1), reshape(ds, :, 1)
  r = sqrt.(1 .- s.^2)

  # Angular grid
  θ, dθ = trigpts(Nθ, θspan)
  θ, dθ = reshape(θ, 1, :), reshape(dθ, 1, :)

  # Complex grid
  ζ = r .* exp.(im * θ) 
  dζ = -s .* ds .* dθ
  
  # Inner product weight
  dw = -(2π / Nθ) * ds

  return r, θ, ζ, dζ, dw

end
