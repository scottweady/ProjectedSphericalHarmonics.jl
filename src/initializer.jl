
using SparseArrays: spdiagm, sparse

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
  ∂Y∂r
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

  # Radial derivative operator in grid space, expensive to compute
  # ∂Y∂r = ∂ylm∂r(Mr, Mθ, r, θ)
  ∂Y∂r = [0.0]

  ### TO DO: Cartesian derivatives ###

  # Normal derivative operator in coefficient space
  ∂n̂ = ∂ylm∂n.(Lspan, Mspan)

  # Eigenvalues
  λ = λlm.(Lspan, Mspan)

  # Integral operators in coefficient space
  Ŝ = even .* (λ ./ 4.0)
  N̂ = -odd .* (1.0 ./ λ)

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))
  
  # Recursion coefficients
  a =    (Nlm.(Lspan .+ 1, Mspan, Lspan, Mspan) .* (2 * Lspan .+ 1) ./ (Lspan .- Mspan .+ 1))
  am1 = -(Nlm.(Lspan .+ 1, Mspan, Lspan .- 1, Mspan) .* (Lspan .+ Mspan) ./ (Lspan .- Mspan .+ 1))

  return disk(shp, Mr, Mθ, Lspan, Mspan, odd, even, r, θ, ζ, dζ, w, dw, Ŝ, N̂, ∂θ̂, ∂n̂, a, am1, ∂Y∂r)

end

disk(M::Int) = disk(M, M)
