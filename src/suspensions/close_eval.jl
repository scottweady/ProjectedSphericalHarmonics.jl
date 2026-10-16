using SpecialFunctions: loggamma, logabsgamma

#=
Close evaluation of the Stokes single layer of an ellipse at points off the ellipse.

For a target x outside ellipse j, set ξ = diag(1/a, 1/b) Q(θⱼ)ᵀ (x - xcmⱼ) = r e^{iϑ}, r > 1.
In the source body frame the velocity due to f = f̃/ω, with f̃ the (even) PSH coefficients,
is

    u(x) = Σ_{ℓ,m} Σ_{n even} Â_n f̃_ℓ^m ψ_ℓ^{m,n}(ξ),

    ψ_ℓ^{m,n}(ξ) = (√π/4) χ_{m+n} (-1)^{(ℓ-m)/2} y_ℓ^m|_rim e^{i(m+n)ϑ} Γ((ℓ+μ+1)/2) F_ℓ^μ(r),

    F_ℓ^μ(r) = r^{-ℓ-1} / (Γ((μ-ℓ+1)/2) Γ(ℓ+3/2)) ₂F₁((ℓ+μ+1)/2, (ℓ-μ+1)/2; ℓ+3/2; 1/r²),

with μ = |m+n|, χ_k = 1 for k ≥ 0 and (-1)^k for k < 0, and Â_n the Fourier coefficients
of A₀ (see `EllipseGeometry`). Terms with μ > ℓ do not vanish outside the disk.
=#

"""
Default near-target radius `c`: targets of body i are corrected for source body j when
|ξⱼ| - 1 < c h κ, with h = 2π/(2Mₘ+1) the azimuthal node spacing and κ = max(a,b)/min(a,b).

Measured relative error of the node quadrature at |ξ| - 1 = c h κ (smooth random density):

| a/b | M  | c = 4  | c = 6  | c = 8  |
|-----|----|--------|--------|--------|
| 1   | 16 | 4e-6   | 2e-7   | 1e-8   |
| 1   | 32 | 9e-9   | 8e-11  | 2e-12  |
| 10  | 16 | 2e-7   | 2e-9   | 5e-11  |
| 10  | 32 | 1e-9   | 5e-12  | 3e-14  |
"""
const CLOSE_RADIUS = 8.0

# Projected spherical harmonic on the rim (ϱ = 0), in the convention of `ylm`
_yrim(l, m) = sqrt(2) * Plm_norm(l, abs(m), 0.0)

_χ(k) = k < 0 ? (-1)^k : 1

"""
    exterior_radial_table!(G, L, μmax, r)

Fill G[ℓ+1, μ+1] = Γ((ℓ+μ+1)/2) F_ℓ^μ(r) for ℓ ≤ L, μ ≤ μmax, ℓ - μ even (zero otherwise).

The ₂F₁ is evaluated through the quadratic transformation
₂F₁(a, b; a+b+½; 4w(1-w)) = ₂F₁(2a, 2b; a+b+½; w), w = (1 - x)/2, x = √(1 - 1/r²).
For μ ≤ ℓ the transformed series has positive terms and w ≤ ½. For μ ≥ ℓ+2 it terminates
and equals the normalized Gegenbauer polynomial C_N^{(ℓ+1)}(x)/C_N^{(ℓ+1)}(1),
N = μ - ℓ - 1, computed for all μ in one pass of the three-term recurrence. A direct
Float64 ₂F₁ is unusable for μ > ℓ (large parameters of opposite sign).
"""
function exterior_radial_table!(G, L, μmax, r)

  x  = sqrt(1 - 1 / r^2)
  w  = (1 - x) / 2
  lr = log(r)
  fill!(G, 0)

  for l in 0 : L

    # Γ((ℓ+μ+1)/2) / (Γ((μ-ℓ+1)/2) Γ(ℓ+3/2)) r^{-ℓ-1}; (μ-ℓ+1)/2 may be a negative half-integer
    lg3 = loggamma(l + 1.5)
    function prefactor(μ)
      lg2, s2 = logabsgamma((μ - l + 1) / 2)
      return s2 * exp(loggamma((l + μ + 1) / 2) - lg2 - lg3 - (l + 1) * lr)
    end

    # μ ≤ ℓ: positive series in w
    for μ in (l % 2) : 2 : min(l, μmax)
      A, B, C = l + μ + 1, l - μ + 1, l + 1.5
      s = t = 1.0
      k = 0
      while true
        t *= (A + k) * (B + k) / ((C + k) * (k + 1)) * w
        s += t
        k += 1
        abs(t) < 1e-17 * s && break
      end
      G[l + 1, μ + 1] = prefactor(μ) * s
    end

    # μ ≥ ℓ+2: R_k = C_k(x)/C_k(1), λ = ℓ+1,
    # R_k = 2x(k+λ-1)/(k+2λ-1) R_{k-1} - (k-1)/(k+2λ-1) R_{k-2}, R_0 = 1, R_1 = x
    λ = l + 1
    k, Rkm2, Rkm1 = 1, 1.0, x
    for μ in l + 2 : 2 : μmax
      N = μ - l - 1
      while k < N
        k += 1
        Rk = 2 * (k + λ - 1) / (k + 2λ - 1) * x * Rkm1 - (k - 1) / (k + 2λ - 1) * Rkm2
        Rkm2, Rkm1 = Rkm1, Rk
      end
      G[l + 1, μ + 1] = prefactor(μ) * Rkm1
    end

  end

  return G

end

exterior_radial_table(L, μmax, r) = exterior_radial_table!(zeros(L + 1, μmax + 1), L, μmax, r)

"""
    exterior_map(geom, Ω, ξ)

Dense map from the even PSH coefficients [f̃ₓ[even]; f̃ᵧ[even]] of f·ω on an ellipse to
the velocity (ux; uy) at points ξ outside the unit disk, in normalized coordinates
ξ = Re(z)/a + i Im(z)/b of the ellipse's body frame. The velocity is in the body frame.

# Arguments
- `geom` : `EllipseGeometry` of the source ellipse
- `Ω`    : domain discretization (provides the PSH modes)
- `ξ`    : complex vector of normalized target points, |ξ| > 1

# Returns
- complex matrix of size (2 length(ξ)) × (2 length(Ω.idx_even))
"""
function exterior_map(geom::EllipseGeometry, Ω, ξ::AbstractVector)

  ns, (Â11, Â12, Â22) = geom.ns, geom.Â
  L, Mₘ = Ω.Mℓ, Ω.Mₘ
  nmax = maximum(ns)
  μmax = Mₘ + nmax

  CI = CartesianIndices(Ω.shp)
  modes = [(Ω.Lspan[CI[ci][1]], Ω.Mspan[CI[ci][2]]) for ci in Ω.idx_even]
  pre   = [sqrt(π) / 4 * (-1)^((l - m) ÷ 2) * _yrim(l, m) for (l, m) in modes]
  Ne, Nt = length(modes), length(ξ)

  T = zeros(ComplexF64, 2Nt, 2Ne)

  Threads.@threads for t in 1 : Nt

    r, ϑ = abs(ξ[t]), angle(ξ[t])
    G = exterior_radial_table(L, μmax, r)

    # χ_M e^{iMϑ} for M = -μmax, …, μmax
    E = [_χ(M) * cis(M * ϑ) for M in -μmax : μmax]

    for (k, (l, m)) in enumerate(modes)
      s11 = s12 = s22 = zero(ComplexF64)
      for (q, n) in enumerate(ns)
        M = m + n
        ψ = E[M + μmax + 1] * G[l + 1, abs(M) + 1]
        s11 += Â11[q] * ψ
        s12 += Â12[q] * ψ
        s22 += Â22[q] * ψ
      end
      T[t, k]           = pre[k] * s11
      T[t, Ne + k]      = pre[k] * s12
      T[Nt + t, k]      = pre[k] * s12
      T[Nt + t, Ne + k] = pre[k] * s22
    end

  end

  return T

end

"""
    exterior_geometry(Ω)

`EllipseGeometry` of the domain for close evaluation, or `nothing` if the closed form does
not apply (domains built from a general conformal map).
"""
exterior_geometry(::Disk)    = EllipseGeometry(1.0, 1.0)
exterior_geometry(Ω::Domain) = getfield(Ω, :ext)

"Close-evaluation correction of the velocity at near targets `tidx` of body i due to body j."
struct CloseEvalPair
  i::Int
  j::Int
  tidx::Vector{Int}           # linear indices into Ω.z of the near targets of body i
  T::Matrix{ComplexF64}       # exterior_map at those targets (body j frame)
end

"""
    close_eval_plan(xcm, θcm, Ω; close_radius=CLOSE_RADIUS)

Find the targets of each body that are close to another body and precompute the closed-form
maps for them. Depends only on the geometry, so build once per solve.

# Arguments
- `xcm` : vector of N complex center positions
- `θcm` : vector of N orientations (radians)
- `Ω`   : domain discretization (shared by all bodies)
- `close_radius` : near-target radius in units of the azimuthal node spacing (see `CLOSE_RADIUS`)

# Returns
- vector of `CloseEvalPair`, or `nothing` if the closed form does not apply to Ω
"""
function close_eval_plan(xcm, θcm, Ω; close_radius=CLOSE_RADIUS)

  geom = exterior_geometry(Ω)
  geom === nothing && return nothing

  a, b = geom.a, geom.b
  R = max(a, b)
  δ = close_radius * 2π / (2Ω.Mₘ + 1) * R / min(a, b)
  zb = vec(Ω.z)

  plan = CloseEvalPair[]
  for i in eachindex(xcm), j in eachindex(xcm)
    (i == j || abs(xcm[i] - xcm[j]) > R * (2 + δ)) && continue
    z = cis(-θcm[j]) .* (xcm[i] .+ cis(θcm[i]) .* zb .- xcm[j])
    ξ = real.(z) ./ a .+ im .* imag.(z) ./ b
    any(abs.(ξ) .<= 1) && throw(ArgumentError("bodies $i and $j overlap"))
    tidx = findall(abs.(ξ) .< 1 + δ)
    isempty(tidx) && continue
    push!(plan, CloseEvalPair(i, j, tidx, exterior_map(geom, Ω, ξ[tidx])))
  end

  return plan

end

"""
    close_eval_correct!(pot, sources, stoklets, plan, f̃_body, θcm, Ω)

Replace the node-quadrature contribution of body j at the near targets of body i by the
closed form, for every pair in `plan`. `pot`, `sources` and `stoklets` are the world-frame
FMM output and inputs (FMM normalization, G/(8π)); `f̃_body` are the even PSH coefficients
of f·ω in each body frame.
"""
function close_eval_correct!(pot, sources, stoklets, plan, f̃_body, θcm, Ω)

  Nz, even = length(Ω.z), Ω.idx_even
  ĉ = Dict(j => vcat(vec(f̃_body[j][1])[even], vec(f̃_body[j][2])[even]) for j in unique(p.j for p in plan))

  for p in plan
    nt = length(p.tidx)
    v  = p.T * ĉ[p.j]
    vx, vy = rotate(real.(v[1:nt]), real.(v[nt+1:end]), θcm[p.j])

    # Node-quadrature contribution of body j, exactly as the FMM sees it
    bi = (p.i - 1) * Nz .+ p.tidx
    bj = (p.j - 1) * Nz .+ (1 : Nz)
    native = st3ddir(sources[:, bj], sources[:, bi]; stoklet=stoklets[:, bj], ppregt=1).pottarg

    pot[1, bi] .+= vx ./ 2 .- native[1, :]
    pot[2, bi] .+= vy ./ 2 .- native[2, :]
  end

  return pot

end
