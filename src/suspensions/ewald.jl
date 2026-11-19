using FINUFFT
using SpecialFunctions: erf, erfc, erfcinv

#=
Fast in-plane 3D Stokeslet sums by Ewald splitting + NUFFT, a drop-in for `stfmm3d` on coplanar
points (adapted from OpenSurfaces.jl's `EwaldStokes3D`, self mode only: targets = sources).

    pot[:, i] = Σ_{j ≠ i} K(xᵢ − xⱼ) fⱼ / (8π),   K(r) = I/ρ + r rᵀ/ρ³,   r ∈ R²,

the FMM3D normalization (free-space Stokeslet at unit viscosity), so the output can replace
`stfmm3d(eps, sources; stoklet=stoklets, ppreg=1).pot`.

With φ = 1/ρ = φ_L + φ_S, φ_L = erf(ρ/σ)/ρ (smooth) and φ_S = erfc(ρ/σ)/ρ (short range), and
K[φ](r) := φ I − (φ′(ρ)/ρ) r rᵀ (linear in φ, K[1/ρ] = K), K = K[φ_L] + K[φ_S].

 * Long range: K[φ_L] is smooth. Multiplied by a box window W(r) = w(r₁)w(r₂) that equals 1 (to
   tolerance) for all separations in the bounding box, it has compact support, so its free-space
   convolution equals a periodic one on twice the support. Its Fourier multiplier is computed once
   by FFT of samples; each apply is a type-1 NUFFT, a 2×2 multiplier and a type-2 NUFFT.
 * Short range: pairs closer than r_c = cσ (erfc(c) ≈ tol) are summed through a sparse matrix.
 * i = j is skipped: the short range drops it, and K[φ_L](0) f = (2/(σ√π)) f is subtracted.
   Distinct points are assumed (no other coincident pairs).
=#

@inline function _phiL(ρ, σ)            # erf(ρ/σ)/ρ and its derivative
  x = ρ / σ
  if x < 1e-3
    c = 2 / (σ * sqrt(π))
    return c * (1 - x^2 / 3 + x^4 / 10), c * (-2x / 3 + 2x^3 / 5) / σ
  end
  v = erf(x) / ρ
  d = (2 / (σ * sqrt(π))) * exp(-x^2) / ρ - v / ρ
  v, d
end

@inline function _phiS(ρ, σ)            # erfc(ρ/σ)/ρ and its derivative
  x = ρ / σ
  v = erfc(x) / ρ
  d = -(2 / (σ * sqrt(π))) * exp(-x^2) / ρ - v / ρ
  v, d
end

"K[φ](r) = φ I − (φ′/ρ) r rᵀ, returned as (K11, K12, K22)."
@inline function _Kphi(rx, ry, φ, dφ, ρ)
  a = ρ > 0 ? dφ / ρ : 0.0
  (φ - a * rx^2, -a * rx * ry, φ - a * ry^2)
end

# ── Neighbor search on a cell list ────────────────────────────────────────────

function _cell_list(X, rc)
  lo1, lo2 = minimum(@view X[1, :]), minimum(@view X[2, :])
  cell(x1, x2) = (floor(Int, (x1 - lo1) / rc), floor(Int, (x2 - lo2) / rc))
  cells = Dict{Tuple{Int,Int},Vector{Int}}()
  for j in axes(X, 2)
    push!(get!(cells, cell(X[1, j], X[2, j]), Int[]), j)
  end
  cells, cell
end

function _neighbors!(out, X, i, cells, cell, rc2)
  empty!(out)
  x1, x2 = X[1, i], X[2, i]
  ci, cj = cell(x1, x2)
  for di in -1:1, dj in -1:1
    lst = get(cells, (ci + di, cj + dj), nothing)
    lst === nothing && continue
    for j in lst
      j == i && continue
      ρ2 = (x1 - X[1, j])^2 + (x2 - X[2, j])^2
      (ρ2 < rc2 && ρ2 > 0) && push!(out, j)
    end
  end
  sort!(out)
end

# Threaded `f(buf, i)` over 1:n in chunks, one neighbor buffer per chunk
function _chunked_foreach(f, n)
  nchunk = max(1, min(n, 8 * Threads.nthreads()))
  @sync for rng in Iterators.partition(1:n, cld(max(n, 1), nchunk))
    Threads.@spawn (b = Int[]; for i in rng; f(b, i); end)
  end
end

# σ giving about `nneigh` short-range neighbors per point, from an evenly strided sample
function _calibrate_sigma(X, c, nneigh, σ0)
  rc = c * σ0
  cells, cell = _cell_list(X, rc)
  n = size(X, 2)
  idx = n <= 2000 ? collect(1:n) : round.(Int, range(1, n, length=2000))
  b = Int[]; cnt = 0
  for i in idx
    _neighbors!(b, X, i, cells, cell, rc^2)
    cnt += length(b)
  end
  σ0 * sqrt(nneigh / max(cnt / length(idx), 1.0))
end

"""
Transposed short-range matrix St (2n × 2n): columns 2i-1, 2i hold target i's rows, over the
interleaved source components, so u_sr = St' * [f₁x, f₁y, f₂x, …].
"""
function _sr_matrix(X, rc, σ, scale)
  n = size(X, 2)
  cells, cell = _cell_list(X, rc)
  rc2 = rc^2
  cnt = zeros(Int, n)
  _chunked_foreach(n) do b, i
    _neighbors!(b, X, i, cells, cell, rc2)
    cnt[i] = length(b)
  end
  colptr = Vector{Int}(undef, 2n + 1); colptr[1] = 1
  for i in 1:n
    colptr[2i] = colptr[2i-1] + 2cnt[i]
    colptr[2i+1] = colptr[2i] + 2cnt[i]
  end
  nz = colptr[end] - 1
  rowval = Vector{Int}(undef, nz); nzval = Vector{Float64}(undef, nz)
  _chunked_foreach(n) do b, i
    _neighbors!(b, X, i, cells, cell, rc2)
    p1, p2 = colptr[2i-1], colptr[2i]
    for (m, j) in enumerate(b)
      rx, ry = X[1, i] - X[1, j], X[2, i] - X[2, j]
      ρ = hypot(rx, ry)
      φ, dφ = _phiS(ρ, σ)
      k11, k12, k22 = scale .* _Kphi(rx, ry, φ, dφ, ρ)
      o = 2(m - 1)
      rowval[p1+o] = 2j - 1; nzval[p1+o] = k11; rowval[p1+o+1] = 2j; nzval[p1+o+1] = k12
      rowval[p2+o] = 2j - 1; nzval[p2+o] = k12; rowval[p2+o+1] = 2j; nzval[p2+o+1] = k22
    end
  end
  SparseMatrixCSC(2n, 2n, colptr, rowval, nzval)
end

# ── Plan and apply ────────────────────────────────────────────────────────────

struct StokesEwald
  n::Int
  M::NTuple{2,Int}
  Khat::Array{Float64,3}                 # (M₁, M₂, 3): K11, K12, K22, FFT mode order, scaled
  plan1::FINUFFT.finufft_plan{Float64}   # type 1 at the points
  plan2::FINUFFT.finufft_plan{Float64}   # type 2 at the points
  St::SparseMatrixCSC{Float64,Int}       # transposed short-range matrix
  selfcoef::Float64                      # windowed K[φ_L](0)/(8π), subtracted at i = j
end

"""
    StokesEwald(sources; tol=1e-6, nneigh=150)

Ewald + NUFFT plan for the in-plane Stokeslet sum over the fixed points `sources` (3 × n, z
ignored), the counterpart of `stfmm3d(tol, sources; stoklet, ppreg=1)`. Apply with
`stokes_ewald!(pot, E, stoklets)`. Plan once per point configuration and reuse it.
"""
function StokesEwald(sources::AbstractMatrix; tol::Real=1e-6, nneigh::Real=150)

  X = Matrix{Float64}(sources[1:2, :])
  n = size(X, 2)
  lo = (minimum(X[1, :]), minimum(X[2, :]))
  hi = (maximum(X[1, :]), maximum(X[2, :]))
  c0 = (lo .+ hi) ./ 2
  L  = max.(hi .- lo, 1e-12)

  c  = max(erfcinv(tol), 3.0)
  σ  = _calibrate_sigma(X, c, nneigh, sqrt(nneigh / (π * n / (L[1] * L[2]))) / c)
  rc = c * σ

  # window w(t; ℓ) = ½ erfc((|t| − ℓ − cσ)/σ): 1 − O(tol) for |t| ≤ ℓ, O(tol) beyond ℓ + 2cσ
  P = 2 .* (L .+ 2rc)
  h = π * σ / (2c)
  M = ntuple(d -> 2 * ceil(Int, P[d] / (2h)), 2)
  hx, hy = P[1] / M[1], P[2] / M[2]

  K = zeros(ComplexF64, M[1], M[2], 3)
  Threads.@threads for b in 0:M[2]-1
    ry = (b < M[2] ÷ 2 ? b : b - M[2]) * hy
    wy = erfc((abs(ry) - L[2] - rc) / σ) / 2
    for a in 0:M[1]-1
      rx = (a < M[1] ÷ 2 ? a : a - M[1]) * hx
      wx = erfc((abs(rx) - L[1] - rc) / σ) / 2
      ρ = hypot(rx, ry)
      φ, dφ = _phiL(ρ, σ)
      k11, k12, k22 = _Kphi(rx, ry, φ, dφ, ρ)
      W = wx * wy
      K[a+1, b+1, 1] = W * k11; K[a+1, b+1, 2] = W * k12; K[a+1, b+1, 3] = W * k22
    end
  end
  FFTW.fft!(K, (1, 2))
  # pot = (1/(P₁P₂)) Σ_m K̂(k_m) F_m e^{i k_m x} / (8π),  K̂ ≈ h₁h₂ · FFT(samples)
  Khat = real.(K) .* (hx * hy / (P[1] * P[2] * 8π))

  sx = 2π .* (X[1, :] .- c0[1]) ./ P[1]
  sy = 2π .* (X[2, :] .- c0[2]) ./ P[2]
  ntol = max(tol / 10, 100eps())          # FINUFFT's double-precision floor
  nt = Threads.nthreads()
  plan1 = FINUFFT.finufft_makeplan(1, [M[1], M[2]], -1, 2, ntol; modeord=1, nthreads=nt)
  FINUFFT.finufft_setpts!(plan1, sx, sy)
  plan2 = FINUFFT.finufft_makeplan(2, [M[1], M[2]], +1, 2, ntol; modeord=1, nthreads=nt)
  FINUFFT.finufft_setpts!(plan2, sx, sy)

  St = _sr_matrix(X, rc, σ, 1 / 8π)
  W0 = (erfc((-L[1] - rc) / σ) / 2) * (erfc((-L[2] - rc) / σ) / 2)
  selfcoef = (2 / (σ * sqrt(π))) * W0 / 8π

  StokesEwald(n, M, Khat, plan1, plan2, St, selfcoef)

end

"""
    stokes_ewald!(pot, E, stoklets)

Overwrite `pot[1:2, :]` with the in-plane Stokeslet sum of `stoklets` (3 × n, quadrature weights
included) in the FMM3D normalization; `pot[3, :]` is set to zero.
"""
function stokes_ewald!(pot::AbstractMatrix, E::StokesEwald, stoklets::AbstractMatrix)

  n = E.n
  c = Matrix{ComplexF64}(undef, n, 2)
  @inbounds for j in 1:n
    c[j, 1] = stoklets[1, j]; c[j, 2] = stoklets[2, j]
  end

  F = Array{ComplexF64}(undef, E.M[1], E.M[2], 2)
  FINUFFT.finufft_exec!(E.plan1, c, F)
  K = E.Khat
  @inbounds Threads.@threads for b in 1:E.M[2]
    for a in 1:E.M[1]
      F1, F2 = F[a, b, 1], F[a, b, 2]
      F[a, b, 1] = K[a, b, 1] * F1 + K[a, b, 2] * F2
      F[a, b, 2] = K[a, b, 2] * F1 + K[a, b, 3] * F2
    end
  end
  U = Matrix{ComplexF64}(undef, n, 2)
  FINUFFT.finufft_exec!(E.plan2, F, U)

  # Short range (threaded over targets) and the i = j correction
  St = E.St; cp, rv, nv = St.colptr, St.rowval, St.nzval
  @inbounds Threads.@threads for i in 1:n
    a, b = cp[2i-1], cp[2i]
    s1 = s2 = 0.0
    for m in 0:2:b-a-1
      j = (rv[a+m] + 1) ÷ 2
      x1, x2 = stoklets[1, j], stoklets[2, j]
      s1 += nv[a+m] * x1 + nv[a+m+1] * x2
      s2 += nv[b+m] * x1 + nv[b+m+1] * x2
    end
    pot[1, i] = real(U[i, 1]) + s1 - E.selfcoef * stoklets[1, i]
    pot[2, i] = real(U[i, 2]) + s2 - E.selfcoef * stoklets[2, i]
    pot[3, i] = 0.0
  end

  return pot

end
