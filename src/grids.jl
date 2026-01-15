
using FastGaussQuadrature

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
function legpts(N, dom)
    x, dx = gausslegendre(N)
    x = @. dom[1] * (1 - x) / 2 + dom[2] * (1 + x) / 2
    dx = @. dx * (dom[2] - dom[1]) / 2
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
function trigpts(N, dom)

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
- `ζ` : points
- `dζ` : weights
"""
function diskpts(Nr, Nθ, rspan=[0, 1], θspan=[0, 2π])

  # Radial grid
  s, ds = legpts(Nr, sqrt.(1 .- rspan.^2))

  # Angular grid
  θ, dθ = trigpts(Nθ, θspan)
  
  # Complex coordinates
  ζ = sqrt.(1 .- s.^2) * exp.(im * θ')
  ζ = reshape(ζ, :, 1)

  # Volume element
  dζ = -(s .* ds) * dθ'
  dζ = reshape(dζ, :, 1)

  return ζ, dζ

end
