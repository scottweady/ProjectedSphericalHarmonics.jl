
using LinearAlgebra
using SparseArrays

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
  idx_even
  r
  θ
  ζ
  dζ
  w
  dw
  dθ
  ∂ζ̂
  ∂n̂
  Ŝ
  N̂
  Ĝ
  ŜN̂⁻¹
  a
  am1
  W
  Wqr
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
  r, θ, ζ, dζ, dw, dθ = diskpts(Mℓ + 1, 2Mₘ + 1)

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))

  # Grid size
  shp = (Mℓ + 1, 2Mₘ + 1)

  # Azimuthal and radial mode numbers
  L = 0 : Mℓ
  M = transpose([0 : Mₘ; -Mₘ : -1])

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
  W, Wqr = build_weight_operators(L, M)

  # Coefficients for ∂/∂ζ in coefficient space
  ∂ζ̂ = 0.5 * Nlm.(L, M, L, M .- 1) .* ((M .<= 0) .- (M .> 0) .* ((L .+ M) .* (L .- M .+ 1.0)))

  idx_even = findall(vec(even))
  
  Ĝ = stokes3d_single_layer_matrix(L, M, idx_even)
  ŜN̂⁻¹ = laplace3d_SN⁻¹_matrix(L, M, idx_even)

  return Disk(shp, Mℓ, Mₘ, L, M, odd, even, total, idx_even, r, θ, ζ, dζ, w, dw, dθ, ∂ζ̂, ∂n̂, Ŝ, N̂, Ĝ, ŜN̂⁻¹, a, am1, W, Wqr)

end

disk(M::Int) = disk(M, M)
