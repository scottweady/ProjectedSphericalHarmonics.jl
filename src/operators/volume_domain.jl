# ---- Volume potentials on mapped domains ---------------------------------------------------------------------------
# On Ω = F(D), with the density g = (u ∘ F) J pulled back to the disk (J the Jacobian of F),
#   ∫_Ω k(x_p - y) u(y) dy = ∫_D k(F(ζ_p) - F(η)) g(η) dA(η).
# With the Wirtinger divided differences F(ζ) - F(η) = P Δζ + Q conj(Δζ) of the pair (smooth in both points; see
# `MappedKernels`), μ = Q/P (|μ| < 1) and ϕ = arg Δζ, Δζ = η - ζ_p,
#   log|Δz| = log|Δζ| + log|P| + log|1 + μe^{-2iϕ}|,   |Δz| = |P| |Δζ| |1 + μe^{-2iϕ}|,
#   log|1 + μe^{-2iϕ}| = Σ_{n ≥ 1} (-1)^{n+1}/(2n) (μⁿ e^{-2inϕ} + μ̄ⁿ e^{2inϕ}),
#   |1 + μe^{-2iϕ}|    = H₀(s) + Σ_{n ≥ 1} Hₙ(s) (μⁿ e^{-2inϕ} + μ̄ⁿ e^{2inϕ}),  Hₙ(s) = Σ_l b_{l+n} b_l sˡ,  s = |μ|²,
# with b_j the binomial coefficients of the power 1/2 (`_abs_H`). The coefficients cₖ(P, Q) of the pair are smooth in η,
# so for each k the integrand is a smooth function, cₖ g, times one of the kernels log|Δζ|, e^{2ikϕ} (bounded; only
# its direction is discontinuous at the target) and |Δζ| e^{2ikϕ}. Their product weights against the even PSH, for a
# target on each ring (`_volume_ring_weights`), rotate to every target of the ring, and composed with the transform
# they act on grid values: the disk's grid operators of the three kernels, in tables like `_circulants`. The operator
# on Ω is the sum over k of the coefficients of each pair times these, the same singularity swapping as for `𝒮`:
#   L[p, q] = J_q [ T_log + log|P_pq| T₀ + Σ_{k ≠ 0} cₖ(μ_pq) e^{2ikθ_p} T_k ],   A[p, q] = J_q |P_pq| Σₖ dₖ(μ_pq) e^{2ikθ_p} U_k.
# Expanding with the divided differences of the pair, rather than with μ at the target alone, keeps every factor
# smooth in η: the expansion is exact but for its truncation at |μ|^K, so the potential converges spectrally. With μ
# at the target only, the remainder log|Δz| - (its model) is continuous but not smooth at the target, and the
# plain grid quadrature of it converges only algebraically (a plateau near 1e-5 on a 1.1 × 0.91 ellipse, downstream).

# The pairs of grid points: the map's P, Q for the grid points p, q (`_vpq`), the Jacobian J on the grid and the largest
# s = |μ|² over the pairs (a bound, for the interpolation of the Hₙ)
struct _AffinePQ
  P::ComplexF64
  Q::ComplexF64
end
_vpq(g::_AffinePQ, p, q, mws) = (g.P, g.Q)

struct _ConformalPQ
  ζ::Vector{ComplexF64}
  z::Vector{ComplexF64}
  z′::Vector{ComplexF64}
end
_vpq(g::_ConformalPQ, p, q, mws) = p == q ? (g.z′[p], zero(ComplexF64)) : ((g.z[p] - g.z[q]) / (g.ζ[p] - g.ζ[q]), zero(ComplexF64))

struct _MappedPQ{T}
  map::T
  ζ::Vector{ComplexF64}
end
_vpq(g::_MappedPQ, p, q, mws) = _divdiff(g.map, p, q, g.ζ[p], g.ζ[q], mws)

_vpq_workspace(g::_MappedPQ) = _map_workspace(g.map)
_vpq_workspace(g) = nothing

function _volume_geometry(Ω::Domain)
  N = length(Ω.z)
  ext = getfield(Ω, :ext)
  if ext isa EllipseGeometry
    a, b = ext.a, ext.b
    return _AffinePQ((a + b) / 2, (a - b) / 2), fill(a * b, N), ((a - b) / (a + b))^2
  end
  G = getfield(Ω, :K_S)
  G isa ConformalKernels && return _ConformalPQ(G.ζ, G.z, G.z′), abs2.(G.z′), 0.0
  G isa MappedKernels && return _MappedPQ(G.map, G.ζ), G.J, G.smax
  throw(ArgumentError("volume potentials need a Domain from `ellipse` or `domain`"))
end

# The number of angular harmonics: |μ|^K/K below rounding
_volume_K(smax) = smax < 1e-28 ? 0 : ceil(Int, log(1e-17) / (0.5 * log(smax)))

# Hₙ(s) = Σ_l b_{l+n} b_l sˡ, n = 0 … K (H[n + 1]): the coefficients of |1 + μe^{-2iϕ}|
function _abs_H(K, s)
  L = s < 1e-300 ? 1 : ceil(Int, log(1e-18) / log(s)) + 40
  b = ones(L + K + 2)
  for j = 1 : L + K + 1
    b[j + 1] = b[j] * (0.5 - j + 1) / j
  end
  H = zeros(K + 1)
  for l = L : -1 : 0, n = 0 : K               # smallest terms first
    H[n + 1] += b[l + n + 1] * b[l + 1] * s^l
  end
  return H
end

# The product weights of a target on each ring of D, for |k| ≤ K: they depend only on the grid (and K), so are kept
const _VOLUME_RINGS = Dict{Tuple{Int, Int, Int}, Any}()
const _VOLUME_RINGS_LOCK = ReentrantLock()

function _volume_rings(D, K)
  key = (D.Mℓ, D.Mₘ, K)
  rings = lock(() -> get(_VOLUME_RINGS, key, nothing), _VOLUME_RINGS_LOCK)
  rings === nothing || return rings
  r = vec(D.r)
  rings = Vector{Any}(undef, length(r))
  Threads.@threads for i in eachindex(r)
    rings[i] = _volume_ring_weights(D, r[i]; N=max(K, 1))
  end
  lock(() -> (_VOLUME_RINGS[key] = rings), _VOLUME_RINGS_LOCK)
  return rings
end

# The disk's grid operators of the kernels, for a target (i, 1) on ring i (angle 0) and the source i′ + (d - 1) Nr:
# T0[i, q] of log|Δζ| (kernel = :log) or |Δζ| (:abs), and Tk[k + K + 1, i, q] of e^{2ikϕ} (:log) or |Δζ| e^{2ikϕ}
# (:abs). The target (i, j) and source (i′, j′) is e^{2ikθ_j} times the entry of d = j′ - j mod Nθ
function _volume_tables(D, K, kernel)
  Nr, Nθ = D.shp
  N = Nr * Nθ
  Pₑ = psh_matrix(D; parity=:even)[D.idx_even, :]
  rings = _volume_rings(D, K)
  cols = (max(K, 1) + 1) .+ (-K : K)
  T0 = Matrix{ComplexF64}(undef, Nr, N)
  Tk = Array{ComplexF64, 3}(undef, 2K + 1, Nr, N)
  for i = 1 : Nr
    wlog, wabs, Wk, Vk = rings[i]
    T0[i, :] = transpose(kernel === :log ? wlog : wabs) * Pₑ
    Tk[:, i, :] = transpose((kernel === :log ? Wk : Vk)[:, cols]) * Pₑ
  end
  return T0, Tk
end

# The grid matrix of the kernel log|x - y| (kernel = :log) or |x - y| (:abs) on Ω (see the comment above)
function _volume_matrix(D, geom, J, smax, kernel)
  Nr, Nθ = D.shp
  N = Nr * Nθ
  K = _volume_K(smax)
  T0, Tk = _volume_tables(D, K, kernel)
  θ = vec(D.θ)
  ph = [cis(2n * θ[j]) for n = 0 : K, j = 1 : Nθ]          # e^{2inθ_j}
  cl = [(-1)^(n + 1) / 2n for n = 1 : K]
  smaxc = max(smax * (1 + 1e-12), 1e-8)
  cheb = kernel === :abs ? _harmonic_chebyshev(K, smaxc; H=_abs_H, nH=K + 1) : zeros(0, 0)
  V = Matrix{ComplexF64}(undef, N, N)
  chunks = collect(Iterators.partition(1 : N, cld(N, 4Threads.nthreads())))
  Threads.@threads for cs in chunks
    mws = _vpq_workspace(geom)
    H, b₁, b₂ = zeros(K + 1), zeros(K + 1), zeros(K + 1)
    μⁿ = zeros(ComplexF64, K + 1)
    for q in cs
      i′, j′ = mod1(q, Nr), cld(q, Nr)
      @inbounds for j = 1 : Nθ
        qd = i′ + mod(j′ - j, Nθ) * Nr
        for i = 1 : Nr
          p = i + (j - 1) * Nr
          P, Q = _vpq(geom, p, q, mws)
          μ = Q / P
          μⁿ[1] = 1
          for n = 1 : K
            μⁿ[n + 1] = μⁿ[n] * μ
          end
          if kernel === :log
            acc = T0[i, qd] + log(abs(P)) * Tk[K + 1, i, qd]
            for n = 1 : K
              e = ph[n + 1, j]
              acc += cl[n] * (μⁿ[n + 1] * conj(e) * Tk[K + 1 - n, i, qd] + conj(μⁿ[n + 1]) * e * Tk[K + 1 + n, i, qd])
            end
            V[p, q] = J[q] * acc
          else
            _clenshaw!(H, b₁, b₂, cheb, 2abs2(μ) / smaxc - 1)
            acc = H[1] * Tk[K + 1, i, qd]
            for n = 1 : K
              e = ph[n + 1, j]
              acc += H[n + 1] * (μⁿ[n + 1] * conj(e) * Tk[K + 1 - n, i, qd] + conj(μⁿ[n + 1]) * e * Tk[K + 1 + n, i, qd])
            end
            V[p, q] = J[q] * abs(P) * acc
          end
        end
      end
    end
  end
  return V
end

"""
    volume_matrix(Ω::Domain; kernel=:log)

The volume integral of the kernel ``\\log|x - y|`` (`kernel = :log`) or ``|x - y|`` (`kernel = :abs`) on `Ω`, as a
matrix on the grid: for a smooth function ``f`` sampled at `Ω.z`,

```math
\\int_Ω \\log|z_p - y|\\, f(y)\\, dy ≈ \\sum_q V_{pq}\\, f(z_q),
```

with spectral accuracy. The weights hold the Jacobian of the map and a projection on the even PSH, so ``V_{pq}``
times a smooth function of the pair (such as ``A(|z_p - z_q|)``) still integrates that product against ``f``: this
gives the weights of kernels ``A(ρ)\\log ρ + E(ρ) + ρ\\,O(ρ)`` with smooth ``A, E, O``. Dense, ``N × N`` for ``N``
grid points, made on the first call and kept by `Ω`. See `𝒱`.
"""
function volume_matrix(Ω::Domain; kernel::Symbol=:log)
  kernel === :log || kernel === :abs || throw(ArgumentError("kernel must be :log or :abs, not :$kernel"))
  cache, lk = getfield(Ω, :factors)
  key = kernel === :log ? :volume_log : :volume_abs
  V = lock(() -> get(cache, key, nothing), lk)
  V === nothing || return V
  geom, J, smax = _volume_geometry(Ω)
  V = _volume_matrix(getfield(Ω, :D), geom, J, smax, kernel)
  lock(() -> (cache[key] = V), lk)
  return V
end

"""
    𝒱(u, Ω::Domain)
    𝒱(x, u, Ω::Domain)

Volume potential of the 2D Laplacian on a `Domain` (`ellipse` or `domain`, with any of its maps),

```math
\\mathcal{V}[u](x) = \\frac{1}{2\\pi} \\int_Ω \\log|x - y|\\, u(y)\\, dy,
```

so that ``Δ\\mathcal{V}[u] = u`` in `Ω`. `u` is a smooth function, sampled at `Ω.z`, or a number. Without `x`, the
potential on the grid; it converges spectrally (to about 1e-15 for a smooth `u` on a moderate ellipse at `M = 16`).
The first call makes the dense grid matrix (`volume_matrix`), which `Ω` keeps: seconds at `M = 16`, more for strongly
deformed maps (the number of angular terms grows like ``\\log ε / \\log|μ|``, with ``|μ| < 1`` the anisotropy of the map).

With points `x` (complex numbers), the potential at those points: inside `Ω` for every map (the map is inverted
by Newton's method), and also outside for `ellipse` (whose affine map extends to the plane). Each point costs about
as much as a ring of the grid matrix.
"""
function 𝒱(u, Ω::Domain)
  isa(u, Number) && (u = fill(u, size(Ω.z)))
  return reshape(volume_matrix(Ω) * vec(ComplexF64.(u)), size(u)) ./ 2π
end

function 𝒱(x::AbstractVector{<:Number}, u, Ω::Domain)
  isa(u, Number) && (u = fill(u, size(Ω.z)))
  D = getfield(Ω, :D)
  geom, J, smax = _volume_geometry(Ω)
  g = J .* vec(ComplexF64.(u))
  Pₑ = psh_matrix(D; parity=:even)[D.idx_even, :]
  m = [vec(D.Mspan)[cld(k, D.shp[1])] for k in D.idx_even]
  ζ = vec(D.ζ)
  out = zeros(ComplexF64, length(x))
  for (t, xt) in enumerate(x)
    ζ₀ = _volume_preimage(Ω, geom, xt)
    P, Q = _volume_pq(Ω, geom, ζ₀, ζ)
    μ = Q ./ P
    K = _volume_K(max(smax, maximum(abs2, μ)))
    wlog, _, Wk, _ = _volume_ring_weights(D, abs(ζ₀); N=max(K, 1))
    θ₀ = angle(ζ₀)
    ks = -K : K
    w0 = transpose(cis.(m .* θ₀) .* wlog) * Pₑ
    R = transpose(cis.(m .* θ₀ .+ 2θ₀ .* transpose(ks)) .* Wk[:, (max(K, 1) + 1) .+ ks]) * Pₑ
    acc = zero(ComplexF64)
    for q in eachindex(ζ)
      a = w0[q] + log(abs(P[q])) * R[K + 1, q]
      μq = μ[q]
      μⁿ = one(ComplexF64)
      for n = 1 : K
        μⁿ *= μq
        a += (-1)^(n + 1) / 2n * (μⁿ * R[K + 1 - n, q] + conj(μⁿ) * R[K + 1 + n, q])
      end
      acc += a * g[q]
    end
    out[t] = acc / 2π
  end
  return out
end

# The point ζ₀ of the disk with F(ζ₀) = x: exact for the affine map (anywhere), by Newton's method from the nearest grid
# point for the others (inside Ω only)
function _volume_preimage(Ω, geom, x)
  if geom isa _AffinePQ
    ext = getfield(Ω, :ext)
    return real(x) / ext.a + im * imag(x) / ext.b
  end
  z, ζ = vec(Ω.z), vec(Ω.ζ)
  ζ₀ = ζ[argmin(abs.(z .- x))]
  scale = maximum(abs, z)
  for _ = 1 : 60
    abs(ζ₀) < 1 || break
    r = Ω.f(ζ₀) - x
    abs(r) < 1e-15 * scale && return ζ₀
    d = Ω.df(ζ₀)
    a, b = d isa Tuple ? d : (d, zero(ComplexF64))
    ζ₀ -= (conj(a) * r - b * conj(r)) / (abs2(a) - abs2(b))
  end
  abs(ζ₀) < 1 && abs(Ω.f(ζ₀) - x) < 1e-12 * scale && return ζ₀
  throw(ArgumentError("𝒱 at $x: points outside a domain are supported only for `ellipse` (affine map)"))
end

# P, Q of the pairs (ζ₀, η) for the points η of the grid
_volume_pq(Ω, geom::_AffinePQ, ζ₀, η) = fill(geom.P, length(η)), fill(geom.Q, length(η))

function _volume_pq(Ω, geom::_ConformalPQ, ζ₀, η)
  f₀ = Ω.f(ζ₀)
  # Close pairs by the mean value of F′ along the segment (Gauss rule), to avoid the cancellation of the quotient
  t, w = _gl01(16)
  P = [abs(ζ₀ - e) < 0.1 ? sum(w[g] * Ω.df(e + t[g] * (ζ₀ - e)) for g in eachindex(t)) : (f₀ - Ω.f(e)) / (ζ₀ - e) for e in η]
  return P, zeros(ComplexF64, length(η))
end

function _volume_pq(Ω, geom::_MappedPQ, ζ₀, η)
  if geom.map isa _HarmonicMap
    PQ = [_wirtinger_dd(geom.map.α, geom.map.β, ζ₀, e) for e in η]
    return first.(PQ), last.(PQ)
  end
  F = Ω.f
  F isa WinslowMap || throw(ArgumentError("𝒱 at points: unsupported map"))
  # The mean values of ∂F, ∂̄F along the segments (exact by the Gauss rule; see `_mean_value_tables`)
  ng = length(F.tg)
  pts = [e + F.tg[g] * (ζ₀ - e) for g = 1 : ng, e in η]
  a, b = _derivatives(F, vec(pts))
  return vec(sum(F.wg .* reshape(a, ng, :), dims=1)), vec(sum(F.wg .* reshape(b, ng, :), dims=1))
end

# ---- Volume potentials at rim targets ------------------------------------------------------------------------------
# For the targets e^{iτ_j} on the rim, the same expansion with the pairs (e^{iτ_j}, ζ_q): the disk's weights of the
# kernels for the target 1 (`_volume_ring_weights` at r = 1, shared by all the targets) rotate to e^{iτ}, and composed
# with the transform they act on grid values. The transform is circulant in θ, Pₑ[e, (i′, j′)] = Pₑ[e, (i′, 1)]
# e^{-i m_e (θ_j′ - θ₁)}, so the rotated weights on the grid are one FFT in θ per target and kernel.
function _rim_volume_matrix(Ω, n, kernel)
  D = getfield(Ω, :D)
  geom, J, _ = _volume_geometry(Ω)
  Nr, Nθ = D.shp
  N = Nr * Nθ
  ζ = vec(ComplexF64.(D.ζ))
  α = 2π .* (0 : n - 1) ./ n
  P = Matrix{ComplexF64}(undef, n, N)
  Q = similar(P)
  Threads.@threads for j = 1 : n
    Pj, Qj = _volume_pq(Ω, geom, cis(α[j]), ζ)
    P[j, :] .= Pj
    Q[j, :] .= Qj
  end
  smax = maximum(abs2.(Q ./ P))
  smax < 1 || throw(ArgumentError("volume_matrix: |μ| = $(sqrt(smax)) ≥ 1 for a pair of a rim target and a grid point"))
  K = _volume_K(smax)
  wlog, _, Wk, Vk = _volume_ring_weights(D, 1.0; N=max(K, 1))
  cols = (max(K, 1) + 1) .+ (-K : K)
  W = kernel === :log ? hcat(Wk[:, cols], wlog) : Vk[:, cols]     # column k + K + 1: e^{2ikϕ} (or |Δζ| e^{2ikϕ})
  ncol = size(W, 2)
  # G[nm, i′, c]: the weights of kernel c on the column j′ = 1 of the grid, summed over ℓ for each order m
  Pₑ = psh_matrix(D; parity=:even)[D.idx_even, 1 : Nr]
  nms = [cld(k, Nr) for k in D.idx_even]
  Mspan = vec(D.Mspan)
  G = zeros(ComplexF64, length(Mspan), Nr, ncol)
  for c = 1 : ncol, (e, nm) in enumerate(nms), i′ = 1 : Nr
    G[nm, i′, c] += W[e, c] * Pₑ[e, i′]
  end
  bins = mod.(Mspan, Nθ) .+ 1
  cl = [(-1)^(k + 1) / 2k for k = 1 : K]
  smaxc = max(smax * (1 + 1e-12), 1e-8)
  cheb = kernel === :abs ? _harmonic_chebyshev(K, smaxc; H=_abs_H, nH=K + 1) : zeros(0, 0)
  V = Matrix{ComplexF64}(undef, n, N)
  Threads.@threads for js in collect(Iterators.partition(1 : n, cld(n, Threads.nthreads())))
    A = zeros(ComplexF64, Nθ, Nr, ncol)
    plan = plan_fft!(A, 1)
    H, b₁, b₂ = zeros(K + 1), zeros(K + 1), zeros(K + 1)
    for j in js
      fill!(A, 0)
      for (nm, m) in enumerate(Mspan)
        e = cis(m * α[j])
        for c = 1 : ncol, i′ = 1 : Nr
          A[bins[nm], i′, c] += G[nm, i′, c] * e
        end
      end
      plan * A                                    # A[j′, i′, c]: kernel c rotated to e^{iα_j}, at the grid point (i′, j′)
      ph = [cis(2k * α[j]) for k = 1 : K]
      for j′ = 1 : Nθ, i′ = 1 : Nr
        q = i′ + (j′ - 1) * Nr
        μ = Q[j, q] / P[j, q]
        μⁿ = one(ComplexF64)
        if kernel === :log
          acc = A[j′, i′, 2K + 2] + log(abs(P[j, q])) * A[j′, i′, K + 1]
          for k = 1 : K
            μⁿ *= μ
            acc += cl[k] * (μⁿ * conj(ph[k]) * A[j′, i′, K + 1 - k] + conj(μⁿ) * ph[k] * A[j′, i′, K + 1 + k])
          end
          V[j, q] = J[q] * acc
        else
          _clenshaw!(H, b₁, b₂, cheb, 2abs2(μ) / smaxc - 1)
          acc = H[1] * A[j′, i′, K + 1]
          for k = 1 : K
            μⁿ *= μ
            acc += H[k + 1] * (μⁿ * conj(ph[k]) * A[j′, i′, K + 1 - k] + conj(μⁿ) * ph[k] * A[j′, i′, K + 1 + k])
          end
          V[j, q] = J[q] * abs(P[j, q]) * acc
        end
      end
    end
  end
  return V
end

"""
    volume_matrix(Ω::Domain, n::Int; kernel=:log)

The volume integral of the kernel ``\\log|x - y|`` (`kernel = :log`) or ``|x - y|`` (`kernel = :abs`) on `Ω` at the
`n` boundary points ``x_j = F(e^{iτ_j})``, ``τ_j = 2π(j - 1)/n``, as an `n × N` matrix on the grid: for a smooth
``f`` sampled at `Ω.z`,

```math
\\int_Ω \\log|x_j - y|\\, f(y)\\, dy ≈ \\sum_q V_{jq}\\, f(z_q),
```

with spectral accuracy. As for `volume_matrix(Ω)` (targets on the grid), the weights hold the Jacobian and the even-PSH
projection, so ``V_{jq}`` times a smooth function of the pair (such as ``A(|x_j - z_q|)``) still integrates that
product against ``f``. Made on the first call and kept by `Ω`. See also `line_matrix`.
"""
function volume_matrix(Ω::Domain, n::Int; kernel::Symbol=:log)
  kernel === :log || kernel === :abs || throw(ArgumentError("kernel must be :log or :abs, not :$kernel"))
  n >= 1 || throw(ArgumentError("n must be positive"))
  cache, lk = getfield(Ω, :factors)
  key = (:volume_rim, kernel, n)
  V = lock(() -> get(cache, key, nothing), lk)
  V === nothing || return V
  V = _rim_volume_matrix(Ω, n, kernel)
  lock(() -> (cache[key] = V), lk)
  return V
end
