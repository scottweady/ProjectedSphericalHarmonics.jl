
using SparseArrays: spdiagm, sparse
include("recurrences.jl")

# Define line by line elements of disk struct
struct disk
  shp
  Mr
  Mθ
  Lspan
  Mspan
  odd
  even
  r
  θ
  ζ
  dζ
  w
  dw
  Ŝ
  N̂
  ∂θ̂
  ∂n̂
  a
  am1
  ∂ζ̂
  Ŵ
  Ŵ⁻¹
end

"""
    disk(Mr, Mθ)

Discretization of the unit disk using projected spherical harmonics.

# Arguments
- `Mr` : maximum radial degree
- `Mθ` : maximum azimuthal order

# Returns
- Struct containing discretization information
"""
function disk(Mr::Int, Mθ::Int)

  # Polar coordinate discretization of the disk
  r, θ, ζ, dζ, dw = diskpts(Mr + 1, 2Mθ + 1)

  # Grid size
  shp = (Mr + 1, 2Mθ + 1)

  # Azimuthal and radial mode numbers
  Lspan = 0 : Mr
  Mspan = [0 : Mθ; -Mθ : -1]'

  # Even and odd boolean
  even = (Lspan .+ Mspan) .% 2 .== 0 .&& abs.(Mspan) .<= Lspan
  odd =  (Lspan .+ Mspan) .% 2 .== 1 .&& abs.(Mspan) .<= Lspan

  # Angular derivative operator in coefficient space
  ∂θ̂ = 0 * Lspan .+ im * Mspan

  # Normal derivative operator in coefficient space
  ∂n̂ = ∂ylm∂n.(Lspan, Mspan)

  # Eigenvalues
  λ = λlm.(Lspan, Mspan)

  # Integral operators in coefficient space
  Ŝ = even .* (λ ./ 4.0)
  N̂ = -odd .* (1.0 ./ λ)

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))

  # Weight operators in coefficient space
  Ŵ, Ŵ⁻¹ = build_weight_operators(Lspan, Mspan)

  # Recursion coefficients
  a =    (Nlm.(Lspan .+ 1, Mspan, Lspan, Mspan) .* (2 * Lspan .+ 1) ./ (Lspan .- abs.(Mspan) .+ 1))
  am1 = -(Nlm.(Lspan .+ 1, Mspan, Lspan .- 1, Mspan) .* (Lspan .+ abs.(Mspan)) ./ (Lspan .- abs.(Mspan) .+ 1))

  function ∂ζ̂(l, m)
    c = m <= 0 ? 1.0 : -(l + m) * (l - m + 1.0)
    return (c / 2.0) * Nlm(l, m, l, m - 1)
  end

  ∂ζ̂ = ∂ζ̂.(Lspan, Mspan)

  return disk(shp, Mr, Mθ, Lspan, Mspan, odd, even, r, θ, ζ, dζ, w, dw, Ŝ, N̂, ∂θ̂, ∂n̂, a, am1, ∂ζ̂, Ŵ, Ŵ⁻¹)

end

disk(M::Int) = disk(M, M)
