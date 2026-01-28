
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
  az
  Yr
  Yθ
  ∂Y∂n
  ∂Y∂r
  r
  θ
  ζ
  dζ
  w
  dw
  Ŝ
  N̂
  ∂θ̂
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
  shp = (Mr + 1, 2Mθ + 1)

  # Evaluate eigenfunctions and derivatives
  Yr = ylm(Mr, Mθ, r, [0.0])
  Yθ = ylm(Mr, Mθ, [1.0], θ)
  ∂Y∂r = ∂ylm∂r(Mr, Mθ, r, θ)
  ∂Y∂n = ∂ylm∂n(Mr, Mθ, θ)
  
  # Azimuthal and radial modes
  Lspan = 0 : Mr
  Mspan = [0 : Mθ; -Mθ : -1]'

  # Eigenvalues
  λ = vec(λlm.(Lspan, Mspan))

  # Even and odd boolean
  even = (Lspan .+ Mspan) .% 2 .== 0 .&& abs.(Mspan) .<= Lspan
  odd =  (Lspan .+ Mspan) .% 2 .== 1 .&& abs.(Mspan) .<= Lspan
  even = vec(even)
  odd = vec(odd)

  # Build map from pair index to azimuthal index
  rows_even, rows_odd = Int[], Int[]
  cols_even, cols_odd = Int[], Int[]
  vals_even, vals_odd = Bool[], Bool[]
  neven, nodd = 0, 0

  for (nm, m) in enumerate(Mspan)
    for (_, l) in enumerate(Lspan)
      if (m + l) % 2 == 0 && abs(m) <= l
        neven += 1
        push!(rows_even, neven)
        push!(cols_even, nm)
        push!(vals_even, true)
      elseif (m + l) % 2 == 1 && abs(m) <= l
        nodd += 1
        push!(rows_odd, nodd)
        push!(cols_odd, nm)
        push!(vals_odd, true)
      end

    end
  end

  # Store as sparse
  az_even = sparse(rows_even, cols_even, vals_even, neven, length(Mspan))
  az_odd = sparse(rows_odd, cols_odd, vals_odd, nodd, length(Mspan))
  az = (even = az_even, odd = az_odd)

  # Derivative operators in coefficient space
  iM = vec(0 * Lspan .+ im * Mspan)
  ∂θ̂ = spdiagm(0 => iM[even])

  ### TO DO: Convert to cartesian derivatives ###

  # Integral operators in coefficient space
  Ŝ = spdiagm(0 => λ[even]/4)
  N̂ = spdiagm(0 => -1 ./ λ[odd])

  # Separate by parity
  Yr = (even = Yr[:, even], odd = Yr[:, odd])
  Yθ = (even = Yθ[:, even], odd = Yθ[:, odd])
  ∂Y∂n = (even = ∂Y∂n[:, even], odd = ∂Y∂n[:, odd])
  ∂Y∂r = (even = ∂Y∂r[:, even], odd = ∂Y∂r[:, odd])

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))
  
  return disk(shp, Mr, Mθ, Lspan, Mspan, odd, even, az,
              Yr, Yθ, ∂Y∂n, ∂Y∂r, r, θ, ζ, dζ, w, dw, Ŝ, N̂, ∂θ̂)

end

disk(M::Int) = disk(M, M)
