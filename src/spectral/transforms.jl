
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
"""
    psh(u, D; parity=:even)

PSH coefficients of the grid values `u` on `D` (a `Disk` or `Domain`), as a new array; `psh!`
transforms in place. The coefficients have the shape of the grid: entry `[ℓ + 1, j]` is the
coefficient of ``y_ℓ^m`` with ``m`` = `D.Mspan[j]`. Only the coefficients of the requested parity
(`:even`, ℓ + m even, or `:odd`) are computed; the others are zero. Even PSH are polynomials in
``x, y``; odd ones are ``w`` times polynomials. `u` may also be a number (a constant) or a tuple of
arrays (transformed componentwise). The transform is exact for functions in the span of the
retained modes: `psh(ipsh(û, D), D) == û` to rounding.
"""
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
"""
    ipsh(û, D; parity=:even)

Grid values on `D` of the PSH expansion with coefficients `û` (laid out as by `psh`), as a new
array; `ipsh!` transforms in place. Coefficients of the other parity are ignored. A tuple of
coefficient arrays is transformed componentwise. `ipsh(û, D, r)` with real radii `r` evaluates at
those radii instead (on the grid's angles), as `trace` does on the rim, returning a
`length(r) × (2Mₘ + 1)` array; `ipsh(û, D, ζ)` with complex points `ζ` evaluates at arbitrary points
(see below). `û` is not modified.
"""
function ipsh(û::AbstractArray, D; parity=:even)
  û = ComplexF64.(û)
  shp = size(û)
  û = reshape(û, D.shp)
  ipsh!(û, D, parity=parity)
  return reshape(û, shp)
end

# Evaluation of PSH expansion at arbitrary radial points
function ipsh(û::AbstractArray, D, r::Union{Real, AbstractVector{<:Real}}; parity=:even)

  û = reshape(û, D.shp) .* getproperty(D, parity)
  
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
ipsh(f̂::Tuple, D, ζ::AbstractVector{<:Complex}; parity=:even) = map(fi -> ipsh(fi, D, ζ; parity=parity), f̂)

# Linear indices, into a coefficient array of shape D.shp, of the modes of the parity (D.idx_even for :even)
_mode_indices(D, parity) = parity === :even ? D.idx_even : findall(vec(getproperty(D, parity)))

"""
    mode_orders(D; parity=:even)

Azimuthal order ``m`` of each retained mode of the parity, in the order of the columns of
`ipsh_matrix(D, ζ; parity)`: the modes are listed by their linear index in a coefficient array of
shape `D.shp` (for `:even`, the order of `D.idx_even`).
"""
function mode_orders(D; parity=:even)
  CI = CartesianIndices(D.shp)
  return [D.Mspan[CI[k][2]] for k in _mode_indices(D, parity)]
end

# Polar coordinates of the points ζ, with |ζ| clamped to the rim
_polar(ζ) = (min.(abs.(ζ), 1.0), angle.(ζ))

# Calls f(nm, m, Y) for each azimuthal mode m = D.Mspan[nm], with Y[i, ℓ + 1] = y_ℓ^m(rᵢ) (the radial factor,
# real) for |m| ≤ ℓ ≤ Mℓ from the recurrence of `ipsh!`; the columns ℓ < |m| hold stale values
_radial_sweep(f, D, r) = _radial_sweep(f, D.Mspan, D.Mℓ, D.a, D.am1, r)

# (the fields of a disk are untyped, so they are passed to a function barrier)
function _radial_sweep(f, Mspan, Mℓ::Int, a, am1, r)
  w = sqrt.(1 .- r.^2)
  Y = zeros(length(r), Mℓ + 1)
  for (nm, m) in enumerate(Mspan)
    absm = abs(m)
    absm > Mℓ && continue
    Y[:, absm + 1] .= real.(ylm(absm, m, r))
    if absm < Mℓ
      Y[:, absm + 2] .= real.(ylm(absm + 1, m, r))
      for nl = (absm + 2) : Mℓ
        @views Y[:, nl + 1] .= a[nl, nm] .* w .* Y[:, nl] .+ am1[nl, nm] .* Y[:, nl - 1]
      end
    end
    f(nm, m, Y)
  end
end

"""
    ipsh_matrix(D, ζ; parity=:even)

Values of the PSH basis of the parity at the complex points `ζ`: a dense
`length(ζ) × length(mode_orders(D; parity))` matrix with entries ``y_ℓ^m(ζ_i)``. The columns are the
retained modes of the parity, ordered by their linear index in a coefficient array of shape `D.shp`
(for `:even`, the order of `D.idx_even`), so `ipsh_matrix(D, ζ) * vec(û)[D.idx_even] ≈ ipsh(û, D, ζ)`
for coefficients `û` from `psh`. The points are in the reference disk: for a `Domain` they are
preimages under its map (like `Ω.ζ`, not `Ω.z`). Points with ``|ζ| > 1`` are clamped to the rim,
``ζ/|ζ|``.
`ipsh_matrix(D; parity)` (no points) is the matrix of the inverse transform on the grid.
"""
function ipsh_matrix(D, ζ::AbstractVector{<:Complex}; parity=:even)
  idx = _mode_indices(D, parity)
  col = zeros(Int, D.shp)
  col[idx] .= 1 : length(idx)
  r, θ = _polar(ζ)
  return _ipsh_matrix(D, col, length(idx), r, θ)
end

# Column col[ℓ + 1, nm] (0 if not retained) of A holds y_ℓ^m at the points (r, θ)
function _ipsh_matrix(D, col::Matrix{Int}, n, r, θ)
  A = zeros(ComplexF64, length(r), n)
  _radial_sweep(D, r) do nm, m, Y
    eᵐ = cis.(m .* θ)
    for l = abs(m) : size(Y, 2) - 1
      j = col[l + 1, nm]
      j > 0 && (A[:, j] .= view(Y, :, l + 1) .* eᵐ)
    end
  end
  return A
end

"""
    ipsh(û, D, ζ::AbstractVector{<:Complex}; parity=:even)

Values at the complex points `ζ` of the PSH expansion with coefficients `û` (laid out as by `psh`),
``\\sum_{ℓ,m} û_ℓ^m y_ℓ^m(ζ)`` over the modes of the parity; coefficients of the other parity are
ignored and `û` is not modified. The points are in the reference disk (for a `Domain`, preimages
under its map, like `Ω.ζ`); points with ``|ζ| > 1`` are clamped to the rim. Costs
``O(\\mathrm{length}(ζ) ⋅ Mℓ Mₘ)`` without forming `ipsh_matrix(D, ζ)`.
"""
function ipsh(û::AbstractArray, D, ζ::AbstractVector{<:Complex}; parity=:even)
  r, θ = _polar(ζ)
  return _ipsh_points(ComplexF64.(reshape(û, D.shp) .* getproperty(D, parity)), D, r, θ)
end

# Σ Ĉ[ℓ + 1, nm] y_ℓ^m at the points (r, θ)
function _ipsh_points(Ĉ::Matrix{ComplexF64}, D, r, θ)
  u = zeros(ComplexF64, length(r))
  v = zeros(ComplexF64, length(r))
  _radial_sweep(D, r) do nm, m, Y
    l = abs(m) + 1 : size(Y, 2)
    mul!(v, view(Y, :, l), view(Ĉ, l, nm))
    u .+= v .* cis.(m .* θ)
  end
  return u
end

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
    resize_coefficients(û, D, D′; parity=:even)

PSH coefficients `û` on `D` (laid out as by `psh`) re-laid out for `D′`, which may have a different
degree `Mℓ` and order `Mₘ`: modes ``(ℓ, m)`` present on both are copied, modes only on `D′` are
zero, and modes only on `D` are dropped. `Mℓ` and `Mₘ` may each grow or shrink. The basis functions
do not depend on the discretization, so for `D′` at least as fine as `D` this is exact:
`ipsh(resize_coefficients(û, D, D′), D′)` samples the same function on the grid of `D′`. Going to a
coarser disk truncates (the ``L^2`` projection onto the retained modes). Only coefficients of the
requested parity are kept. `û` may be an array of shape `D.shp`, its flattening (the result is then
flattened too), or a tuple of such arrays. `D` and `D′` may be a `Disk` or `Domain`. The matrix of
this map on the retained-mode vectors is `embedding_matrix(D, D′; parity)`.
"""
function resize_coefficients(û::AbstractArray, D, D′; parity=:even)
  length(û) == prod(D.shp) || throw(DimensionMismatch(
    "coefficient array has $(length(û)) entries, expected $(prod(D.shp)) for a disk of shape $(D.shp)"))
  flat = ndims(û) == 1
  û = reshape(û, D.shp)
  mask′ = getproperty(D′, parity)
  v̂ = zeros(eltype(û), D′.shp)
  for (i, i′) in _common_modes(D, D′)
    mask′[i′] && (v̂[i′] = û[i])
  end
  return flat ? vec(v̂) : v̂
end

resize_coefficients(û::Tuple, D, D′; parity=:even) = map(ûi -> resize_coefficients(ûi, D, D′; parity=parity), û)

"""
    embedding_matrix(D, D′; parity=:even)

Sparse 0–1 matrix of `resize_coefficients` on the retained-mode vectors: of size
`count(D′.parity) × count(D.parity)` (with `parity` `:even` or `:odd`), acting on `û[findall(vec(D.parity))]`
(for `:even` that is `û[D.idx_even]`) and returning the corresponding coefficients on `D′`. Thus

    ipsh_matrix(D′; parity)[:, findall(vec(D′.parity))] * embedding_matrix(D, D′; parity)

evaluates `D`-coefficients on the grid of `D′`. For `D′` at least as fine as `D` the matrix is an
injection with `embedding_matrix(D′, D) * embedding_matrix(D, D′) == I`; in general
`embedding_matrix(D′, D) == transpose(embedding_matrix(D, D′))`.
"""
function embedding_matrix(D, D′; parity=:even)
  mask, mask′ = vec(getproperty(D, parity)), vec(getproperty(D′, parity))
  # Position of each grid-layout index in the retained-mode vector (0 if not retained)
  col = cumsum(mask) .* mask
  row = cumsum(mask′) .* mask′
  I = Int[]; J = Int[]
  for (i, i′) in _common_modes(D, D′)
    if mask[i] && mask′[i′]
      push!(I, row[i′]); push!(J, col[i])
    end
  end
  return sparse(I, J, ones(length(I)), count(mask′), count(mask))
end

# Pairs (i, i′) of linear indices into coefficient arrays of `D` and `D′` holding the same mode (ℓ, m),
# over the modes |m| ≤ ℓ common to both. Columns follow `Mspan` (0, …, Mₘ, -Mₘ, …, -1)
function _common_modes(D, D′)
  Mℓ, Mₘ = min(D.Mℓ, D′.Mℓ), min(D.Mₘ, D′.Mₘ)
  L, L′ = LinearIndices(D.shp), LinearIndices(D′.shp)
  column(m, D) = m >= 0 ? m + 1 : D.shp[2] + m + 1
  return [(L[l + 1, column(m, D)], L′[l + 1, column(m, D′)]) for m = -Mₘ : Mₘ for l = abs(m) : Mℓ]
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
  Y, F = _psh_factors(D)
  Nr, Nθ = D.shp
  mask = getproperty(D, parity)
  dw = vec(D.dw)
  # P[(ℓ, m), (i, j)] = conj(y_ℓ^m(r_i)) dw_i e^{-imθ_j}: the radial projection after the FFT in θ
  P = zeros(ComplexF64, Nr, Nθ, Nr, Nθ)
  Threads.@threads for j = 1 : Nθ
    for i = 1 : Nr, nm = 1 : Nθ
      c = dw[i] * F[nm, j]
      for l = 1 : Nr
        mask[l, nm] && (P[l, nm, i, j] = conj(Y[i, l, nm]) * c)
      end
    end
  end
  return reshape(P, Nr * Nθ, Nr * Nθ)
end

# Radial basis Y[i, ℓ+1, nm] = y_ℓ^m(r_i) (zero for ℓ < |m|), from the recurrence of `ipsh!`, and the
# Fourier factors F[nm, j] = e^{-imθ_j} of `fft` along θ
function _psh_factors(D)
  r = vec(D.r)
  w = sqrt.(1 .- r.^2)
  Nr, Nθ = D.shp
  Y = zeros(ComplexF64, Nr, Nr, Nθ)
  for (nm, m) in enumerate(D.Mspan)
    absm = abs(m)
    absm > D.Mℓ && continue
    Y[:, absm + 1, nm] .= ylm(absm, m, r)
    absm == D.Mℓ && continue
    Y[:, absm + 2, nm] .= ylm(absm + 1, m, r)
    for nl = (absm + 2) : D.Mℓ
      Y[:, nl + 1, nm] .= D.a[nl, nm] * w .* Y[:, nl, nm] .+ D.am1[nl, nm] * Y[:, nl - 1, nm]
    end
  end
  F = [cis(-2π * (nm - 1) * (j - 1) / Nθ) for nm = 1 : Nθ, j = 1 : Nθ]
  return Y, F
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
  Y, F = _psh_factors(D)
  Nr, Nθ = D.shp
  mask = getproperty(D, parity)
  # Q[(i, j), (ℓ, m)] = y_ℓ^m(r_i) e^{imθ_j} for the modes of the parity; the other columns are
  # left as columns of the identity
  Q = zeros(ComplexF64, Nr, Nθ, Nr, Nθ)
  Threads.@threads for nm = 1 : Nθ
    for l = 1 : Nr
      if mask[l, nm]
        for j = 1 : Nθ
          c = conj(F[nm, j])
          for i = 1 : Nr
            Q[i, j, l, nm] = Y[i, l, nm] * c
          end
        end
      else
        Q[l, nm, l, nm] = 1
      end
    end
  end
  return reshape(Q, Nr * Nθ, Nr * Nθ)
end

export psh_matrix, ipsh_matrix, mode_orders
