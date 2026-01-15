
using SparseArrays

export psh_disk

"""
    psh_disk(M)

Discretization of the unit disk using projected spherical harmonics up to degree `M`.

# Arguments
- `M` : maximum degree and order

# Returns
- discretization of the disk
"""
function psh_disk(M::Int)

  Nr = M + 1
  Nθ = 2M + 1

  # Compute interior grid
  ζ, dζ = diskpts(Nr, Nθ)

  # Weight function
  w = sqrt.(1 .- abs2.(ζ))

  # Preallocations
  λ = Array{ComplexF64}(undef, M + 1, 2M + 1) #eigenvalues
  odd = falses(M + 1, 2M + 1) #odd boolean
  even = falses(M + 1, 2M + 1) #even boolean
  modes = Array{Tuple{Int, Int}}(undef, M + 1, 2M + 1) #mode pair

  # Loop over mode numbers and fill in arrays
  for m = -M : M
      
    nm = (M + 1) + m

    for l = max(abs(m), 0) : M

      nl = l + 1

      λ[nl, nm] = λlm(l, m)
      even[nl, nm] = mod(l + m, 2) == 0
      odd[nl, nm] = mod(l + m, 2) == 1
      modes[nl, nm] = (l, m)

    end
  end

  # Polar derivative operator in coefficient space
  iM = im * [modes[i][2] for i in 1 : ((M+1)*(2M+1))]
  iM = iM[vec(even)]
  iM = spdiagm(0 => iM)
  
  ∂̂ = (θ = iM, r = [])

  # Evaluate eigenfunctions and derivatives
  Y = ylm(M, ζ)
  ∂Y∂r = ∂ylm∂r(M, ζ)

  Y = (even = Y[:, even], odd = Y[:, odd])
  ∂Y∂r = (even = ∂Y∂r[:, even], odd = ∂Y∂r[:, odd])

  # Create map from mode pair to index
  modeIndex = Dict{Tuple{Int, Int}, Int}()
  for (idx, mode) in enumerate(modes[even])
      modeIndex[mode] = idx
  end

  for (idx, mode) in enumerate(modes[odd])
    modeIndex[mode] = idx
  end

  # Spectrum of singular operators
  Ŝ = spdiagm(0 => λ[even]/4)
  N̂ = spdiagm(0 => -1 ./ λ[odd])

  # Construct boundary
  θ, _ = trigpts(Nθ, [0, 2π])
  X = exp.(im * θ)

  # Normal derivatives of eigenfunctions on boundary (only even ones are valid)
  ∂Y∂n = ∂ylm∂n(M, X)
  ∂Y∂n = (even = ∂Y∂n[:, even], odd = ∂Y∂n[:, odd])

  # Store
  return (Y = Y, ∂Y∂n = ∂Y∂n, ∂Y∂r = ∂Y∂r, ζ = ζ, dζ = dζ, w = w, Ŝ = Ŝ, N̂ = N̂, odd = odd, even = even, modes = modes, modeIndex = modeIndex, M = M, ∂̂ = ∂̂)

end
