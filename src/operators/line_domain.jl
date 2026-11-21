# ---- Line integrals over the rim at interior targets of mapped domains -----------------------------------------------
# For a target ζ_p inside the disk and the rim η = e^{iτ}, with the Wirtinger divided differences of the pair,
# F(ζ_p) - F(η) = P Δζ + Q conj(Δζ) (Δζ = η - ζ_p, ϕ = arg Δζ, μ = Q/P, all smooth in τ however close ζ_p is to the rim),
#   log|Δz| = log|Δζ| + log|P| + Σ_{n ≥ 1} (-1)^{n+1}/(2n) (μⁿ e^{-2inϕ} + μ̄ⁿ e^{2inϕ}),
#   |Δz|    = |P| |Δζ| [H₀(s) + Σ_{n ≥ 1} Hₙ(s) (μⁿ e^{-2inϕ} + μ̄ⁿ e^{2inϕ})],   s = |μ|²,
# as for the volume potentials (`volume_domain.jl`). The near-singular factors log|Δζ|, e^{2ikϕ}, |Δζ| e^{2ikϕ} depend
# on the target only through |ζ_p| (and a rotation), and their integrals against the Fourier modes e^{imτ} are done
# once per radius r (`_line_ring_weights`):
# - log|e^{iτ} - r| = -Σ_{m ≥ 1} rᵐ cos(mτ)/m, exactly;
# - e^{-2inϕ} = xⁿ bⁿ with x = e^{-iτ} and the Blaschke factor b = (x - r)/(1 - r x), so its Fourier coefficients are
#   the Taylor coefficients of bⁿ, exactly by a recurrence (and e^{2inϕ} is the conjugate);
# - |Δζ| e^{2ikϕ}: Gauss panels refined geometrically toward τ = 0 (the branch points are at τ = ±i log(1/r)).
# With the trigonometric interpolant of the density on n equispaced rim nodes, these give weights on the nodes (one FFT
# per target and harmonic), and the matrix is the sum over the harmonics of the coefficients of each pair times them.
# As for `volume_matrix`, the expansion is exact but for its truncation at |μ|^K, and every factor besides the kernels
# is smooth in τ: the line integral converges spectrally, uniformly up to the rim.

# The Fourier weights for a target at r ∈ [0, 1) on the positive axis, for m = -mmax : mmax (index m + mmax + 1):
# glog[m] = ∫ log|e^{iτ} - r| e^{imτ} dτ, E[m, k + K + 1] = ∫ e^{2ikϕ} e^{imτ} dτ and (kernel = :abs)
# V[m, k + K + 1] = ∫ |e^{iτ} - r| e^{2ikϕ} e^{imτ} dτ, |k| ≤ K, all over [0, 2π)
function _line_ring_weights(r, mmax, K, kernel; ng=16, hfac=8)
  ms = -mmax : mmax
  glog = [m == 0 ? 0.0 : -π * r^abs(m) / abs(m) for m in ms]
  E = zeros(length(ms), 2K + 1)
  E[mmax + 1, K + 1] = 2π
  c = zeros(mmax + 1)                      # Taylor coefficients of bⁿ, degree ≤ mmax
  c[1] = 1
  t = similar(c)
  for n = 1 : K
    # bⁿ = bⁿ⁻¹ (x - r)/(1 - r x)
    t[1] = -r * c[1]
    for j = 2 : mmax + 1
      t[j] = c[j - 1] - r * c[j]
    end
    c[1] = t[1]
    for j = 2 : mmax + 1
      c[j] = t[j] + r * c[j - 1]
    end
    # ∫ e^{-2inϕ} e^{imτ} dτ = 2π [bⁿ]_{m - n} (m ≥ n), and e^{2inϕ} with -m
    for m = n : mmax
      E[mmax + 1 + m, K + 1 - n] = 2π * c[m - n + 1]
      E[mmax + 1 - m, K + 1 + n] = 2π * c[m - n + 1]
    end
  end
  kernel === :abs || return glog, E, zeros(0, 0)
  # Gauss panels on [-π, π]: width ≤ h, refined toward 0 from d/4 when the branch points at distance d are closer
  h = min(π / 4, hfac / (mmax + 2K + 4))
  bps = collect(range(-π, π, length=2ceil(Int, π / h) + 1))
  d = r > 0 ? log(1 / r) : Inf
  d < h && append!(bps, _geometric_breakpoints(0.0, -1, d / 4, h), _geometric_breakpoints(0.0, 1, d / 4, h))
  τ, wτ = _panel_nodes(bps, ng)
  Δ = cis.(τ) .- r
  a = abs.(Δ)
  u = (Δ ./ a).^2                           # e^{2iϕ}
  Fk = Matrix{ComplexF64}(undef, length(τ), 2K + 1)
  Fk[:, K + 1] .= wτ .* a
  for k = 1 : K
    Fk[:, K + 1 + k] .= Fk[:, K + k] .* u
    Fk[:, K + 1 - k] .= Fk[:, K + 2 - k] .* conj.(u)
  end
  V = transpose(cis.(τ .* transpose(ms))) * Fk
  return glog, E, V
end

# The divided differences P[p, j], Q[p, j] of the targets ζ[p] and the rim nodes η[j]
function _line_pq(Ω, geom, ζ, η)
  P = Matrix{ComplexF64}(undef, length(ζ), length(η))
  Q = similar(P)
  Threads.@threads for p in eachindex(ζ)
    Pp, Qp = _volume_pq(Ω, geom, ζ[p], η)
    P[p, :] .= Pp
    Q[p, :] .= Qp
  end
  return P, Q
end

# The line matrix of the targets ζ (inside the disk) and n equispaced rim nodes (see `line_matrix`)
function _line_matrix(Ω, ζ, n, kernel)
  all(x -> abs(x) < 1, ζ) || throw(ArgumentError("line_matrix: the targets must be inside the disk (|ζ| < 1)"))
  geom, _, _ = _volume_geometry(Ω)
  τ = 2π .* (0 : n - 1) ./ n
  P, Q = _line_pq(Ω, geom, ζ, cis.(τ))
  smax = maximum(abs2.(Q ./ P))
  smax < 1 || throw(ArgumentError("line_matrix: |μ| = $(sqrt(smax)) ≥ 1 for a pair of a target and a rim node"))
  K = _volume_K(smax)
  mmax = n ÷ 2
  # The ring weights, once per radius
  rs = unique(abs.(ζ))
  rw = Vector{Any}(undef, length(rs))
  Threads.@threads for i in eachindex(rs)
    rw[i] = _line_ring_weights(rs[i], mmax, K, kernel)
  end
  ring = Dict(zip(rs, rw))
  cl = [(-1)^(k + 1) / 2k for k = 1 : K]
  smaxc = max(smax * (1 + 1e-12), 1e-8)
  cheb = kernel === :abs ? _harmonic_chebyshev(K, smaxc; H=_abs_H, nH=K + 1) : zeros(0, 0)
  ms = -mmax : mmax
  slot = mod.(ms, n) .+ 1                   # the DFT index of each m (both ±n/2 at n/2 for even n)
  half = [iseven(n) && abs(m) == n ÷ 2 ? 0.5 : 1.0 for m in ms]
  ncol = kernel === :log ? 2K + 2 : 2K + 1
  L = Matrix{Float64}(undef, length(ζ), n)
  chunks = collect(Iterators.partition(eachindex(ζ), cld(length(ζ), 4Threads.nthreads())))
  Threads.@threads for ps in chunks
    C = zeros(ComplexF64, n, ncol)
    plan = plan_fft!(C, 1)
    H, b₁, b₂ = zeros(K + 1), zeros(K + 1), zeros(K + 1)
    for p in ps
      glog, E, V = ring[abs(ζ[p])]
      α = angle(ζ[p])
      G = kernel === :log ? E : V
      # Column 1 + k + K: the weights of e^{2ikϕ} (or |Δζ| e^{2ikϕ}), rotated to the target; column 2K + 2: log|Δζ|
      fill!(C, 0)
      for (i, m) in enumerate(ms)
        e = half[i] * cis(m * α) / n
        for c = 1 : 2K + 1
          C[slot[i], c] += G[i, c] * e
        end
        kernel === :log && (C[slot[i], 2K + 2] += glog[i] * e)
      end
      plan * C
      for k = -K : K
        k == 0 && continue
        C[:, k + K + 1] .*= cis(2k * α)
      end
      for j = 1 : n
        μ = Q[p, j] / P[p, j]
        if kernel === :log
          acc = C[j, 2K + 2] + log(abs(P[p, j])) * C[j, K + 1]
          μⁿ = one(ComplexF64)
          for k = 1 : K
            μⁿ *= μ
            acc += cl[k] * (μⁿ * C[j, K + 1 - k] + conj(μⁿ) * C[j, K + 1 + k])
          end
        else
          _clenshaw!(H, b₁, b₂, cheb, 2abs2(μ) / smaxc - 1)
          acc = H[1] * C[j, K + 1]
          μⁿ = one(ComplexF64)
          for k = 1 : K
            μⁿ *= μ
            acc += H[k + 1] * (μⁿ * C[j, K + 1 - k] + conj(μⁿ) * C[j, K + 1 + k])
          end
          acc *= abs(P[p, j])
        end
        L[p, j] = real(acc)
      end
    end
  end
  return L
end

"""
    line_matrix(Ω::Domain, n::Int; kernel=:log, targets=nothing)

The line integral over the boundary of `Ω` of the kernel ``\\log|x - y|`` (`kernel = :log`) or ``|x - y|``
(`kernel = :abs`) at points inside, as a matrix on `n` equispaced rim nodes ``τ_j = 2π(j - 1)/n``: for a smooth
``2π``-periodic ``g``,

```math
\\int_0^{2π} \\log|F(ζ_p) - F(e^{iτ})|\\, g(τ)\\, dτ ≈ \\sum_j L_{pj}\\, g(τ_j),
```

with ``F`` the map of `Ω`. The arc length is not included (put it in ``g``). The targets are the grid points `Ω.ζ`
(in the order of `vec(Ω.z)`), or the points `targets` of the disk (``|ζ| < 1``). The weights integrate the
trigonometric interpolant of ``g`` exactly against each angular harmonic of the kernel, expanded with the divided
differences of the pair (target, rim node), so the accuracy is spectral and uniform up to the rim (about 1e-14 for
targets at 1e-6 from the rim on a 1.1 × 0.91 ellipse with `n = 65`). As with `volume_matrix`, ``L_{pj}`` times a
smooth function of the pair (such as ``A(|z_p - y_j|)``) still integrates that product against ``g``, which gives the
weights of kernels ``A(ρ)\\log ρ + E(ρ) + ρ\\,O(ρ)`` with smooth ``A, E, O``. Dense, (number of targets) × `n`.
See also `divdiff`.
"""
function line_matrix(Ω::Domain, n::Int; kernel::Symbol=:log, targets=nothing)
  kernel === :log || kernel === :abs || throw(ArgumentError("kernel must be :log or :abs, not :$kernel"))
  n >= 1 || throw(ArgumentError("n must be positive"))
  ζ = targets === nothing ? vec(ComplexF64.(Ω.ζ)) : vec(ComplexF64.(targets))
  return _line_matrix(Ω, ζ, n, kernel)
end

"""
    divdiff(Ω::Domain, ζ, η) -> P, Q

The Wirtinger divided differences of the map ``F`` of `Ω` between the points `ζ` and `η` of the closed disk (scalars,
or arrays of the same shape):

```math
F(ζ) - F(η) = P\\,(ζ - η) + Q\\,\\overline{(ζ - η)},
```

smooth in both points, with ``P = F′`` and ``Q = 0`` for the conformal map and ``P = ∂F, Q = \\bar∂F`` in the limit
``η → ζ``. ``μ = Q/P`` has ``|μ| < 1`` for the maps that `domain` accepts. Exact for `ellipse` and the harmonic map
(polynomials), by Gauss mean values of the derivatives along the segment for the Winslow and conformal maps.
"""
function divdiff(Ω::Domain, ζ::Number, η::Number)
  geom, _, _ = _volume_geometry(Ω)
  P, Q = _volume_pq(Ω, geom, ComplexF64(ζ), [ComplexF64(η)])
  return P[1], Q[1]
end

function divdiff(Ω::Domain, ζ::AbstractArray{<:Number}, η::AbstractArray{<:Number})
  size(ζ) == size(η) || throw(DimensionMismatch("ζ and η must have the same shape"))
  geom, _, _ = _volume_geometry(Ω)
  P, Q = similar(ζ, ComplexF64), similar(ζ, ComplexF64)
  for i in eachindex(ζ, η)
    p, q = _volume_pq(Ω, geom, ComplexF64(ζ[i]), [ComplexF64(η[i])])
    P[i], Q[i] = p[1], q[1]
  end
  return P, Q
end
