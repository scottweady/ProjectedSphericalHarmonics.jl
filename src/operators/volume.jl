using FFTW
using LinearAlgebra
using FastGaussQuadrature: gausslegendre

#=
Volume potentials of smooth densities on the unit disk by product integration.

A smooth density is expanded in even PSH, u = Σ_j û_j y_j, and the potential at a target x is
Σ_j û_j ∫_D K(x - y) y_j(y) dy. The integrals of the kernel against the basis ("weights") are
computed once per target radius: for a target at r e^{iα} the weights are those of the target r on
the positive axis times e^{i m_j α}, since y_j(e^{iα} ζ) = e^{i m_j α} y_j(ζ).

- log|x - y|: the Fourier series in θ of log|ρ e^{iθ} - r| reduces the weights to radial integrals
  of polynomials (and of ρ log ρ times a polynomial), done exactly by Gauss rules.
- |x - y| (and the angular harmonics used for non-conformal maps): polar coordinates about the
  target, in which these integrands are polynomials along each chord; the angular rule is refined
  geometrically where the chord length is not smooth (targets near the rim, outside the disk).
=#

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

# col[ℓ + 1, nm]: position of mode (ℓ, D.Mspan[nm]) in D.idx_even, or 0
function _even_columns(D)
  col = zeros(Int, D.shp)
  for (i, k) in enumerate(D.idx_even)
    col[k] = i
  end
  return col
end

# Radial parts P[q, ℓ + 1] of y_ℓ^m (m = D.Mspan[nm]) at the radii ρ ≤ 1 (zero for ℓ < |m|), by the
# recurrence of `ipsh!`
function _radial_basis(D, nm, ρ::AbstractVector)
  m = D.Mspan[nm]
  a = abs(m)
  P = zeros(length(ρ), D.Mℓ + 1)
  a > D.Mℓ && return P
  w = sqrt.(max.(1 .- ρ.^2, 0.0))
  P[:, a + 1] .= real.(ylm(a, m, ρ))
  a == D.Mℓ && return P
  P[:, a + 2] .= real.(ylm(a + 1, m, ρ))
  for nl = (a + 2) : D.Mℓ
    P[:, nl + 1] .= D.a[nl, nm] .* w .* view(P, :, nl) .+ D.am1[nl, nm] .* view(P, :, nl - 1)
  end
  return P
end

# Gauss–Legendre rule on [0, 1]
function _gl01(n)
  x, w = gausslegendre(n)
  return (x .+ 1) ./ 2, w ./ 2
end

# Product weights ω for ∫_0^1 log(v) g(v) dv on the n Gauss–Legendre nodes v of [0, 1], exact for
# polynomials g of degree < n: ω_i = λ_i Σ_k (2k + 1) μ_k P̃_k(v_i), with the modified moments
# μ_k = ∫_0^1 log(v) P̃_k(v) dv = -1 (k = 0), (-1)^{k+1}/(k(k+1)) (k ≥ 1) of the shifted Legendre P̃_k
function _log_rule01(n)
  v, λ = _gl01(n)
  x = 2 .* v .- 1
  p₋₁, p = zeros(n), ones(n)
  ω = -1.0 .* p
  for k = 1 : n - 1
    p₋₁, p = p, ((2k - 1) .* x .* p .- (k - 1) .* p₋₁) ./ k
    ω .+= (2k + 1) * (-1)^(k + 1) / (k * (k + 1)) .* p
  end
  return v, λ .* ω
end

# ---------------------------------------------------------------------------------------------
# Log weights: Fourier in θ, exact radial quadrature
# ---------------------------------------------------------------------------------------------

#=
With log|ρ e^{iθ} - r| = log max(ρ, r) - Σ_{k ≥ 1} (q^k/k) cos kθ, q = min(ρ, r)/max(ρ, r),

  ∫_D log|ζ - r| P_ℓ^m(ρ) e^{imθ} dA = 2π ∫_0^1 g_m(ρ) P_ℓ^m(ρ) ρ dρ,

g_0 = log max(ρ, r), g_m = -q^{|m|}/(2|m|). On [0, min(r, 1)] and [r, 1] the integrands are
polynomials of degree ≤ 2Mℓ + 1 (P_ℓ^m = ρ^{|m|} times a polynomial in ρ²), except for ρ log ρ
P_ℓ^0(ρ) on [r, 1]: there ∫_0^r log r (⋯) + ∫_r^1 log ρ (⋯) = ∫_0^1 ρ log ρ P dρ - r² ∫_0^1 v log v P(rv) dv,
done with the product weights of `_log_rule01`.
=#
function _volume_log_weights(D, r::Real)
  n = D.Mℓ + 2
  x, wx = _gl01(n)
  v, ωv = _log_rule01(n)
  col = _even_columns(D)
  wlog = zeros(ComplexF64, length(D.idx_even))
  for (nm, m) in enumerate(D.Mspan)
    a = abs(m)
    a > D.Mℓ && continue
    if m == 0
      if r >= 1
        val = log(r) .* (transpose(wx .* x) * _radial_basis(D, nm, x))
      else
        val = transpose(ωv .* v) * _radial_basis(D, nm, v) .-
              r^2 .* (transpose(ωv .* v) * _radial_basis(D, nm, r .* v))
      end
    else
      val = zeros(1, D.Mℓ + 1)
      if r > 0                                          # [0, min(r, 1)], q = ρ/r
        ρ₁, w₁ = min(r, 1.0) .* x, min(r, 1.0) .* wx
        val .+= transpose(w₁ .* ρ₁ .* (ρ₁ ./ r).^a) * _radial_basis(D, nm, ρ₁)
      end
      if r < 1                                          # [r, 1], q = r/ρ
        ρ₂, w₂ = r .+ (1 - r) .* x, (1 - r) .* wx
        val .+= transpose(w₂ .* ρ₂ .* (r ./ ρ₂).^a) * _radial_basis(D, nm, ρ₂)
      end
      val .*= -1 / (2a)
    end
    for l = a : D.Mℓ
      k = col[l + 1, nm]
      k == 0 || (wlog[k] = 2π * val[l + 1])
    end
  end
  return wlog
end

# ---------------------------------------------------------------------------------------------
# Polar quadrature about the target
# ---------------------------------------------------------------------------------------------

# Gauss–Legendre nodes (`ng` per panel) on the panels between the sorted breakpoints `bps`
function _panel_nodes(bps, ng)
  bps = sort(unique(bps))
  x, w = gausslegendre(ng)
  φ, wφ = Float64[], Float64[]
  for i = 1 : length(bps) - 1
    a, b = bps[i], bps[i + 1]
    b - a < 1e-15 && continue
    append!(φ, (a + b) / 2 .+ (b - a) / 2 .* x)
    append!(wφ, (b - a) / 2 .* w)
  end
  return φ, wφ
end

# Breakpoints c + dir δ, δ = δmin, 2δmin, … < δmax: panels refined geometrically toward c
_geometric_breakpoints(c, dir, δmin, δmax) = [c + dir * δ for δ in δmin .* 2.0 .^ (0 : max(0, ceil(Int, log2(δmax / δmin))))]

#=
Angular nodes φ (direction of ζ - r) with weights, and the chord ζ = r + t e^{iφ}, t ∈ [t₁, t₂], of a
target at r ≥ 0 on the positive axis, over the upper half plane φ ∈ [0, π] (the reflection ζ ↦ ζ̄
fixes the target and maps the lower half to the upper, see `_volume_polar_weights`). Integrands that
are polynomials in t along each chord have angular integrals analytic in φ except where the chord
ends are not:
- r < 1: t₂ = -r cos φ + √(r² cos²φ + s²), s² = 1 - r², has branch points at φ = π/2 ± i asinh(s/r);
- r = 1: t₂ = -2 cos φ on [π/2, π], entire;
- r > 1: the chord meets the disk for π - asin(1/r) ≤ φ ≤ π; with r sin φ = sin χ the chord ends
  √(r² - sin²χ) ∓ cos χ are analytic in χ ∈ [0, π/2] up to branch points at χ = π/2 ± i acosh(r).
Panels of width ≤ h = `hfac`/(Mℓ + 4) resolve the degree-Mℓ polynomials; panels are refined
geometrically toward a branch point at distance d < h, from d/4.
=#
function _volume_polar_rule(D, r::Real; ng=16, hfac=10)
  h = min(π / 4, hfac / (D.Mℓ + 4))
  if r < 1
    s = sqrt((1 - r) * (1 + r))
    bps = collect(range(0, π, length=ceil(Int, π / h) + 1))
    d = r > 0 ? asinh(s / r) : Inf
    d < h && append!(bps, _geometric_breakpoints(π/2, -1, d / 4, h), _geometric_breakpoints(π/2, 1, d / 4, h))
    φ, wφ = _panel_nodes(bps, ng)
    c = r .* cos.(φ)
    q = sqrt.(c.^2 .+ s^2)
    t₂ = @. ifelse(c > 0, s^2 / (c + q), q - c)                 # -r cos φ + √(⋯), without cancellation
    return φ, wφ, zero(φ), t₂
  elseif r == 1
    φ, wφ = _panel_nodes(collect(range(π/2, π, length=ceil(Int, π / 2h) + 1)), ng)
    return φ, wφ, zero(φ), -2 .* cos.(φ)
  else
    bps = collect(range(0, π/2, length=ceil(Int, π / 2h) + 1))
    d = acosh(r)
    d < h && append!(bps, _geometric_breakpoints(π/2, -1, d / 4, h))
    χ, wχ = _panel_nodes(bps, ng)
    c = sqrt.(r^2 .- sin.(χ).^2)                                 # -r cos φ
    φ = π .- asin.(sin.(χ) ./ r)
    wφ = wχ .* cos.(χ) ./ c                                     # |dφ| = cos χ dχ / (-r cos φ)
    return φ, wφ, c .- cos.(χ), c .+ cos.(χ)
  end
end

#=
Polar-quadrature weights of a target at r on the positive axis: with `F(φ)` the angular factors (a
matrix with a row per angle and a column per factor, with F(-φ) = conj(F(φ))) and `p` the power of
t beyond the area element t dt,

  out[j, c] = ∫∫ t^p F_c(φ) y_j(r + t e^{iφ}) t dt dφ.

Since y_j(ζ̄) = conj(y_j(ζ)), the lower half plane gives the conjugate of the upper half, and out =
2 Re of the upper half. Along each chord y_j is a polynomial in t of degree ≤ Mℓ, so `nt` Gauss
nodes in t are exact. The radial parts of y_ℓ^{±m} coincide and are computed once.
=#
function _volume_polar_weights(D, r, F, p::Int; kw...)
  return _volume_polar_weights(D, r, F, (p,); kw...)[1]
end

# The same for several powers `ps` at once (one matrix each), on shared nodes. The angular factors
# depend on φ only, so the t-sums along each chord are formed first (per mode and angle) and the
# factors applied once per angle: the cost of the factors does not grow with the number of t nodes.
function _volume_polar_weights(D, r, F, ps::Tuple; kw...)
  φ, wφ, t₁, t₂ = _volume_polar_rule(D, r; kw...)
  Fφ = F(φ)
  nt = cld(D.Mℓ + maximum(ps) + 2, 2)
  x, wx = _gl01(nt)
  col = _even_columns(D)
  Nθ = D.shp[2]
  nl = D.Mℓ + 1
  outs = map(_ -> zeros(length(D.idx_even), size(Fφ, 2)), ps)
  chunk = max(1, 2^15 ÷ nt)                           # angles per chunk
  for i₀ = 1 : chunk : length(φ)
    I = i₀ : min(i₀ + chunk - 1, length(φ))
    nI = length(I)
    t = t₁[I]' .+ (t₂[I] .- t₁[I])' .* x               # nt × |I|
    ζ = vec(r .+ t .* transpose(cis.(φ[I])))
    w₀ = wx .* ((t₂[I] .- t₁[I]) .* wφ[I])' .* t       # area element t dt dφ
    wts = map(p -> vec(w₀ .* t.^p), ps)
    FI = Fφ[I, :]
    ρ, θ = min.(abs.(ζ), 1.0), angle.(ζ)
    for m = 0 : min(D.Mₘ, D.Mℓ)                         # Σ_q y_ℓ^{±m}(ζ_q) w_q F(φ_q)
      Pt = transpose(_radial_basis(D, m + 1, ρ))      # nl × (nt |I|)
      for (nm, e) in (m == 0 ? ((1, 1),) : ((m + 1, 1), (Nθ - m + 1, -1)))
        ph = cis.(e * m .* θ)
        for (out, wt) in zip(outs, wts)
          S = dropdims(sum(reshape(Pt .* transpose(ph .* wt), nl, nt, nI); dims=2); dims=2)   # t-sums: nl × |I|
          G = real.(S * FI)                             # real part of the upper-half integral
          for l = m : D.Mℓ
            k = col[l + 1, nm]
            k == 0 || (out[k, :] .+= 2 .* view(G, l + 1, :))
          end
        end
      end
    end
  end
  return map(out -> ComplexF64.(out), outs)
end

# ∫_D |ζ - r| y_j dA, in the order of D.idx_even
_volume_abs_weights(D, r) = vec(_volume_polar_weights(D, r, φ -> ones(length(φ), 1), 1))

"""
    _volume_ring_weights(D, r; N=0)

Product-integration weights of the 2D volume potentials for the target ``ζ_0 = r ≥ 0`` on the
positive axis (inside the disk, on the rim `r = 1`, or outside), against the even PSH ``y_j`` of
`D`, indexed in the order of `D.idx_even`:

- `wlog[j]` ``= \\int_D \\log|ζ - ζ_0|\\, y_j(ζ)\\, dA``,
- `wabs[j]` ``= \\int_D |ζ - ζ_0|\\, y_j(ζ)\\, dA``,

and for `N > 0` the angular harmonics in ``ϕ = \\arg(ζ - ζ_0)``, `k = -N:N` in column `k + N + 1`
(for non-conformal maps):

- `Wk[j, k]` ``= \\int_D e^{2ikϕ}\\, y_j\\, dA``, `Vk[j, k]` ``= \\int_D |ζ - ζ_0|\\, e^{2ikϕ}\\, y_j\\, dA``.

Returns `(wlog, wabs)` for `N == 0`, `(wlog, wabs, Wk, Vk)` otherwise. For a target ``r e^{iα}``
the weights are these times ``e^{i m_j α}``. `wlog` is exact up to rounding (radial Gauss rules
after a Fourier expansion of the kernel); the others use polar coordinates about the target, in
which the integrands are polynomials along each chord, with the angular rule sized from `D.Mℓ`
(about 1e-14 relative to `maximum(abs, wabs)`).
"""
function _volume_ring_weights(D, r::Real; N::Int=0)
  r >= 0 || throw(ArgumentError("r must be nonnegative"))
  wlog = _volume_log_weights(D, r)
  N == 0 && return wlog, _volume_abs_weights(D, r)
  E(φ) = cis.(2 .* φ .* transpose(-N:N))
  Wk, Vk = _volume_polar_weights(D, r, E, (0, 1))
  return wlog, Vk[:, N + 1], Wk, Vk
end

# ---------------------------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------------------------

# Value of `f(D)` cached in D.cache under `key`
function _disk_cached(f, D, key)
  cache, lk = getfield(D, :cache)
  lock(lk) do
    get!(() -> f(D), cache, key)
  end
end

# Grid operator in coefficient space: G[i, ℓ + 1, nm] = c w(r_i)[(ℓ, m)] for the ring weights w of
# kind `kind` (:log, scaled by 1/2π, or :abs, scaled by 1/8π), so that the potential on ring i is
# the inverse FFT in θ of Σ_ℓ G[i, ℓ + 1, nm] û[ℓ + 1, nm]
function _volume_grid_operator(D, kind::Symbol)
  _disk_cached(D, (:volume_grid, kind)) do D
    Nr, Nθ = D.shp
    G = zeros(ComplexF64, Nr, Nr, Nθ)
    r = vec(D.r)
    rows = Vector{Vector{ComplexF64}}(undef, Nr)
    Threads.@threads for i = 1 : Nr
      rows[i] = kind === :log ? _volume_log_weights(D, r[i]) / 2π : _volume_abs_weights(D, r[i]) / 8π
    end
    for i = 1 : Nr
      g = zeros(ComplexF64, Nr, Nθ)
      g[D.idx_even] .= rows[i]
      G[i, :, :] .= g
    end
    G
  end
end

# Potential on the grid from the even coefficients û (laid out as by `psh`)
function _volume_apply(G, û, D)
  Nr, Nθ = D.shp
  v = zeros(ComplexF64, Nr, Nθ)
  for nm = 1 : Nθ
    v[:, nm] .= view(G, :, :, nm) * view(û, :, nm)
  end
  ifft!(v, 2)
  return v .* Nθ
end

# Potential at arbitrary targets x from the even coefficients û, with ring weights `ringw(r)`
function _volume_targets(ringw, x::AbstractArray{<:Number}, û, D)
  ĉ = vec(û)[D.idx_even]
  m = [D.Mspan[CartesianIndices(D.shp)[k][2]] for k in D.idx_even]
  ρ = abs.(x)
  W = Dict(r => ringw(r) for r in unique(vec(ρ)))
  return map(x) do xi
    sum(W[abs(xi)] .* ĉ .* cis.(m .* angle(xi)))
  end
end

_density(u::Number, D) = fill(ComplexF64(u), D.shp)
_density(u::AbstractArray, D) = reshape(ComplexF64.(u), D.shp)

"""
    laplace2d_volume(u, D)
    laplace2d_volume(x, u, D)

Volume potential of the 2D Laplacian, ``\\frac{1}{2π} \\int_D \\log|x - y|\\, u(y)\\, dy``, of a smooth
density `u` on the grid of `D`, on the grid or at the targets `x`. See `𝒱`.
"""
function laplace2d_volume(u, D::Disk)
  û = psh(_density(u, D), D, parity=:even)
  V = _volume_apply(_volume_grid_operator(D, :log), û, D)
  return u isa AbstractArray ? reshape(V, size(u)) : V
end

function laplace2d_volume(x::AbstractArray{<:Number}, u, D::Disk)
  û = psh(_density(u, D), D, parity=:even)
  return _volume_targets(r -> _volume_log_weights(D, r) / 2π, x, û, D)
end

"""
    𝒱(u, D::Disk)
    𝒱(x, u, D::Disk)

Volume potential of the 2D Laplacian on the unit disk,

```math
\\mathcal{V}[u](x) = \\frac{1}{2\\pi} \\int_D \\log|x - y|\\, u(y)\\, dy,
```

so that ``Δ\\mathcal{V}[u] = u`` inside the disk and `𝒱(1, D)` ``= (|x|^2 - 1)/4``. `u` is a
smooth density given by its values on the grid (or a number); it is expanded in even PSH and the
potential of each basis function is integrated exactly (product integration), so the result is
spectrally accurate: exact to rounding for polynomial densities of degree ≤ `Mℓ`.

`𝒱(u, D)` returns the potential on the grid; the grid operator is built on first use and cached
in `D` (``O(N^{3/2})`` storage and work per application, ``N`` grid points). `𝒱(x, u, D)` evaluates
it at the points `x` (a complex number or array, ``x_1 + i x_2``), which may be inside the disk,
on the rim or outside it (where ``\\mathcal{V}[u](x) → \\frac{\\log|x|}{2π} \\int_D u``), at a cost of
``O(N^{3/2})`` per distinct ``|x|``.

A density that is not smooth (for instance with the ``1/w`` behavior of `𝒮`) converges only
algebraically.
"""
𝒱(u, D::Disk) = laplace2d_volume(u, D)
𝒱(x::AbstractArray{<:Number}, u, D::Disk) = laplace2d_volume(x, u, D)
𝒱(x::Number, u, D::Disk) = laplace2d_volume([x], u, D)[1]

"""
    bilaplace3d_volume(u, D)
    bilaplace3d_volume(x, u, D)

Potential ``\\frac{1}{8π} \\int_D |x - y|\\, u(y)\\, dy`` of a smooth density `u`, on the grid or at the
targets `x`. See `𝒯_vol`.
"""
function bilaplace3d_volume(u, D::Disk)
  û = psh(_density(u, D), D, parity=:even)
  T = _volume_apply(_volume_grid_operator(D, :abs), û, D)
  return u isa AbstractArray ? reshape(T, size(u)) : T
end

function bilaplace3d_volume(x::AbstractArray{<:Number}, u, D::Disk)
  û = psh(_density(u, D), D, parity=:even)
  return _volume_targets(r -> _volume_abs_weights(D, r) / 8π, x, û, D)
end

"""
    𝒯_vol(u, D::Disk)
    𝒯_vol(x, u, D::Disk)

The kernel of `𝒯` (the single layer of the 3D bilaplacian) applied to a smooth density,

```math
\\mathcal{T}_{\\mathrm{vol}}[u](x) = \\frac{1}{8\\pi} \\int_D |x - y|\\, u(y)\\, dy,
```

by product integration in even PSH, like `𝒱`: spectrally accurate for smooth `u` (where `𝒯`
expects the ``1/w`` edge behavior). `𝒯_vol(u, D)` returns the potential on the grid (the grid
operator is built on first use and cached in `D`); `𝒯_vol(x, u, D)` evaluates it at the points
`x`, inside, on or outside the disk. For example `𝒯_vol(0im, 1, D)` ``= 1/12``.

The weights come from a polar quadrature about each target radius, ``O(N^2)`` work per distinct
``|x|``: building the grid operator takes about 0.3 s at `disk(32)` and 8 s at `disk(64)` (8
threads).
"""
𝒯_vol(u, D::Disk) = bilaplace3d_volume(u, D)
𝒯_vol(x::AbstractArray{<:Number}, u, D::Disk) = bilaplace3d_volume(x, u, D)
𝒯_vol(x::Number, u, D::Disk) = bilaplace3d_volume([x], u, D)[1]

# The former 𝒱: a direct sum over the grid with the log singularity subtracted, O(N²) and
# algebraically convergent. Kept for comparison only.
function _laplace2d_volume_direct(u, D::Disk)
  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end
  shp = size(u)
  ζ, dζ = D.ζ, D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)
  δζ = ζ .- transpose(ζ)
  V = (1 / 2π) * log.(abs.(δζ) .+ (δζ .== 0)) .* dζ'
  Vu = V * u .+ ((abs2.(ζ) .- 1) / 4 .- sum(V, dims=2)) .* u
  return reshape(Vu, shp)
end
