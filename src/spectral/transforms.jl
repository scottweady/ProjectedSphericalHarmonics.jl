
using FFTW

"""
    psh(u, D; parity=:even)

PSH transform of function `u` on disk `D`.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- PSH coefficients
"""
function psh!(u::AbstractMatrix{ComplexF64}, D::Disk; parity=:even)

  fft!(u, 2)

  r = D.r
  w = sqrt.(1 .- r.^2)
  Nr = length(r)

  uw = zeros(ComplexF64, Nr)
  y₋₁ = zeros(ComplexF64, Nr)
  y = zeros(ComplexF64, Nr)
  y₊₁ = zeros(ComplexF64, Nr)

  # Temporary storage for the PSH coefficients of the current mode
  v = zeros(ComplexF64, size(u, 1))

  a, am1 = D.a, D.am1

  for (nm, m) in enumerate(D.Mspan)

    fill!(v, zero(ComplexF64))

    absm = abs(m)
    y₋₁ .= ylm(absm, m, r)
    y .= ylm(absm + 1, m, r)
    uw .= u[:, nm] .* D.dw

    v[absm + 1] = dot(y₋₁, uw)

    if absm == D.Mℓ
      continue
    end

    v[absm + 2] = dot(y, uw)

    for nl = (absm + 2) : D.Mℓ
      y₊₁ .= a[nl, nm] * w .* y .+ am1[nl, nm] * y₋₁
      v[nl + 1] = dot(y₊₁, uw)
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end

    u[:, nm] .= v
    
  end

  u .*= getfield(D, parity)

  return u
  
end

function psh(u::Number, D::Disk; parity=:even)
  return psh!(fill(ComplexF64(u), size(D.ζ)), D, parity=parity)
end

function psh(u::AbstractMatrix, D::Disk; parity=:even)
  return psh!(ComplexF64.(u), D, parity=parity)
end

"""
    ipsh(û, D; parity=:even)

Inverse PSH transform

# Arguments
- `û` : PSH coefficients of function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- grid values
"""
function ipsh!(u::AbstractMatrix{ComplexF64}, D::Disk; parity=:even)

  u .*= getfield(D, parity)
  
  r = D.r
  w = sqrt.(1 .- r.^2)
  Nr = length(r)

  v = zeros(ComplexF64, Nr)
  y₋₁ = zeros(ComplexF64, Nr)
  y = zeros(ComplexF64, Nr)
  y₊₁ = zeros(ComplexF64, Nr)

  a, am1 = D.a, D.am1

  for (nm, m) in enumerate(D.Mspan)

    # Temporary storage for the grid values of the current mode
    fill!(v, zero(ComplexF64))

    # Initialize recurrence relation
    absm = abs(m)
    y₋₁ .= ylm(absm, m, r)
    y .= ylm(absm + 1, m, r)

    # l = m
    v .+= u[absm + 1, nm] * y₋₁

    if absm == D.Mℓ
      continue
    end

    # l = m + 1
    v .+= u[absm + 2, nm] * y

    # Use recursion to compute the grid values for higher radial modes
    for nl = (absm + 2) : D.Mℓ
      y₊₁ .= a[nl, nm] * w .* y .+ am1[nl, nm] * y₋₁
      v .+= u[nl + 1, nm] * y₊₁
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end

    # Store the computed grid values back into `u`
    u[:, nm] .= v

  end

  ifft!(u, 2)
  u .*= D.shp[2]

  return u
  
end

function ipsh(û::AbstractMatrix, D::Disk; parity=:even)
  return ipsh!(ComplexF64.(û), D, parity=parity)
end

function ipsh(û::AbstractMatrix{ComplexF64}, D::Disk, r; parity=:even)

  û .*= getfield(D, parity)
  
  # Compute transform
  u = zeros(ComplexF64, (length(r), D.shp[2]))

  w = sqrt.(1 .- r.^2)

  y₋₁ = zeros(ComplexF64, length(r))
  y = zeros(ComplexF64, length(r))
  y₊₁ = zeros(ComplexF64, length(r))

  for (nm, m) in enumerate(D.Mspan)

    absm = abs(m)
    y₋₁ .= ylm(absm, m, r)
    y .= ylm(absm + 1, m, r)

    u[:, nm] .+= û[absm+1, nm] * y₋₁

    if absm == D.Mℓ
      continue
    end

    u[:, nm] .+= û[absm+2, nm] * y

    for nl = (absm + 2) : D.Mℓ
      y₊₁ .= D.a[nl, nm] * w .* y .+ D.am1[nl, nm] * y₋₁
      u[:, nm] .+= û[nl + 1, nm] * y₊₁
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end
  end

  u = ifft(u, 2) * D.shp[2]

  return u
  
end
