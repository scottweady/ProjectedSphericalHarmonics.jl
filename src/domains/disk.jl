
include("../spectral/recurrences.jl")

# Define line by line elements of disk struct
struct Disk
  shp
  Mℓ
  Mₘ
  Lspan
  Mspan
  odd
  even
  total
  r
  θ
  ζ
  dζ
  w
  dw
  Ŝ
  N̂
  ∂n̂
  a
  am1
  ∂ζ̂
  Ŵ
  Ŵ⁻¹
  W
end

"""
    disk(Mℓ, Mₘ)

Discretization of the unit disk using projected spherical harmonics.

# Arguments
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Struct containing discretization information
"""
function disk(Mℓ::Int, Mₘ::Int)

  # Polar coordinate discretization of the disk
  r, θ, ζ, dζ, dw = diskpts(Mℓ + 1, 2Mₘ + 1)

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))

  # Grid size
  shp = (Mℓ + 1, 2Mₘ + 1)

  # Azimuthal and radial mode numbers
  Lspan = 0 : Mℓ
  Mspan = transpose([0 : Mₘ; -Mₘ : -1])

  # Short-hand names for convenience
  L, M = Lspan, Mspan

  # Even and odd boolean
  even = (L .+ M) .% 2 .== 0 .&& abs.(M) .<= L
  odd =  (L .+ M) .% 2 .== 1 .&& abs.(M) .<= L
  total = abs.(M) .<= L

  # Normal derivative operator in coefficient space
  ∂n̂ = ∂ylm∂n.(L, M)

  # Integral operators in coefficient space
  Ŝ = even .* (λlm.(L, M) ./ 4.0)
  N̂ = -odd .* (1.0 ./ λlm.(L, M))

  # Recursion coefficients
  a =    (Nlm.(L .+ 1, M, L, M) .* (2 * L .+ 1) ./ (L .- abs.(M) .+ 1))
  am1 = -(Nlm.(L .+ 1, M, L .- 1, M) .* (L .+ abs.(M)) ./ (L .- abs.(M) .+ 1))

  # Weight operators in coefficient space
  Ŵ, Ŵ⁻¹, W = build_weight_operators(L, M)

  # Coefficients for ∂/∂ζ in coefficient space
  ∂ζ̂ = 0.5 * Nlm.(L, M, L, M .- 1) .* ((M .<= 0) .* 1.0 .- (M .> 0) .* ((L .+ M) .* (L .- M .+ 1.0)))

  return Disk(shp, Mℓ, Mₘ, Lspan, Mspan, odd, even, total, r, θ, ζ, dζ, w, dw, Ŝ, N̂, ∂n̂, a, am1, ∂ζ̂, Ŵ, Ŵ⁻¹, W)

end

disk(M::Int) = disk(M, M)
