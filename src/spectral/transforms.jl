
using FFTW

"""
    psh(u, D; parity=:even)
    psh!(u, D; parity=:even)

PSH transform of function `u` on disk `D`.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- PSH coefficients
"""
function psh!(u::AbstractArray{ComplexF64}, D; parity=:even)

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
      u[:, nm] .= v
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

  u .*= getproperty(D, parity)

  return u
  
end

# Memory allocating psh
function psh(u::AbstractArray, D; parity=:even)
  u = ComplexF64.(u)
  shp = size(u)
  u = reshape(u, D.shp)
  psh!(u, D, parity=parity)
  return reshape(u, shp)
end

# psh for scalar input
psh(u::Number, D; parity=:even) = psh!(fill(ComplexF64(u), size(D.z)), D, parity=parity)
psh(f::Tuple, D; parity=:even) = map(fi -> psh(fi, D; parity=parity), f)

# psh for flattened input
function psh_vec!(u, D; parity=:even)
  u = reshape(u, D.shp)
  psh!(u, D, parity=parity)
  u = vec(u)
end

"""
    ipsh(û, D; parity=:even)
    ipsh!(û, D; parity=:even)

Inverse PSH transform

# Arguments
- `û` : PSH coefficients of function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- grid values
"""
function ipsh!(u::AbstractArray{ComplexF64}, D; parity=:even)

  u .*= getproperty(D, parity)
  
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
      u[:, nm] .= v
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

# Memory allocating ipsh
function ipsh(û::AbstractArray, D; parity=:even)
  û = ComplexF64.(û)
  shp = size(û)
  û = reshape(û, D.shp)
  ipsh!(û, D, parity=parity)
  return reshape(û, shp)
end

# Evaluation of PSH expansion at arbitrary radial points
function ipsh(û::AbstractArray{ComplexF64}, D, r; parity=:even)

  û .*= getproperty(D, parity)
  
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

# ipsh for tuple input
ipsh(f̂::Tuple, D; parity=:even) = map(fi -> ipsh(fi, D; parity=parity), f̂)

# ipsh for flattened input
function ipsh_vec!(û, D; parity=:even)
  û = reshape(û, D.shp)
  ipsh!(û, D, parity=parity)
  û = vec(û)
end

"""
  upsample(u, D, Nr_new, Nθ_new; parity=:even)

Upsample function `u` on disk `D` to new grid with `Nr_new` radial points and `Nθ_new` azimuthal points.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk
- `Nr_new` : number of radial points in new grid
- `Nθ_new` : number of azimuthal points in new grid
- `parity` : either `:even` or `:odd` expansion

# Returns
- `ζ_new` : new grid points
- upsampled function on new grid
"""
function upsample(u, D, Nr_new::Int, Nθ_new::Int; parity=:even)

  û = psh(u, D, parity=parity)

  s, _ = legpts(Nr_new, [0.0, 1.0])
  r_new = sqrt.(1 .- vec(s).^2)
  θ_new, _ = trigpts(Nθ_new)
  ζ_new = r_new .* exp.(im * transpose(vec(θ_new)))
  w = sqrt.(1 .- r_new.^2)

  u_pad = zeros(ComplexF64, Nr_new, Nθ_new)
  y₋₁ = zeros(ComplexF64, Nr_new)
  y   = zeros(ComplexF64, Nr_new)
  y₊₁ = zeros(ComplexF64, Nr_new)

  for (nm, m) in enumerate(D.Mspan)
    nm_new = m >= 0 ? m + 1 : Nθ_new + m + 1
    absm = abs(m)
    y₋₁ .= ylm(absm, m, r_new)
    y   .= ylm(absm + 1, m, r_new)

    u_pad[:, nm_new] .+= û[absm+1, nm] * y₋₁

    if absm == D.Mℓ
      continue
    end

    u_pad[:, nm_new] .+= û[absm+2, nm] * y

    for nl = (absm + 2) : D.Mℓ
      y₊₁ .= D.a[nl, nm] * w .* y .+ D.am1[nl, nm] * y₋₁
      u_pad[:, nm_new] .+= û[nl+1, nm] * y₊₁
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end
  end

  return ζ_new, ifft(u_pad, 2) * Nθ_new

end


"""
  psh_matrix(D; parity=:even)

  Dense matrix representation of the PSH transform.

# Arguments
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- dense matrix representation of the PSH transform
"""
function psh_matrix(D; parity=:total)
  N = length(D.ζ)
  P = Matrix{ComplexF64}(I, N, N)
  for i = 1 : N
    e = @view P[:, i]
    psh_vec!(e, D, parity=parity)
  end
  return P
end

"""
  ipsh_matrix(D; parity=:even)

  Dense matrix representation of the inverse PSH transform.

# Arguments
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- dense matrix representation of the inverse PSH transform
"""
function ipsh_matrix(D; parity=:total)
  N = length(D.ζ)
  Q = Matrix{ComplexF64}(I, N, N)

  idx = findall(vec(getproperty(D, parity)))

  for i = idx
    e = @view Q[:, i]
    ipsh_vec!(e, D, parity=parity)
  end
  return Q
end

export psh_matrix, ipsh_matrix
