
struct Domain
  D
  f
  df
  z
  dz
end

"""
    domain(f, df, Mℓ, Mₘ)

Discretization of the image of the unit disk under a conformal map.

# Arguments
- `f` : conformal map from disk to domain
- `df` : derivative of conformal map
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Struct containing discretization information
"""
function domain(f, df, Mℓ::Int, Mₘ::Int)
  D = disk(Mℓ, Mₘ)
  z = f.(D.ζ)
  dz = abs2.(df.(D.ζ)) .* D.dζ
  return Domain(D, f, df, z, dz)
end