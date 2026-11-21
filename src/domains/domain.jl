
using LinearAlgebra

"""
    Domain

Discretization of a non-circular domain, the image of the unit disk under a map ``z = f(ζ)``:
returned by `ellipse(a, b, Mℓ, Mₘ)` (an affine map) and `domain(γ, Mℓ, Mₘ)` (a harmonic or conformal map).
It wraps the `Disk` of the reference grid (its fields, such as `even` and `w`, are reachable from
the `Domain`) and adds the mapped points `z`, the area weights `dz` (with the Jacobian of the map),
and the operators of the domain as matrices in coefficient space (`K̂_S`, `K̂_N`, `K̂_G`). Functions
on a `Domain` are sampled at `Ω.z`; the weight `Ω.w` is the disk's, ``\\sqrt{1 - |ζ|^2}``, which
vanishes on the boundary like the square root of the distance to it. For a `domain`, the
operators on the grid (`K_S`, `K_N`, `K_G`) are formed when accessed, as dense matrices.
"""
struct Domain
  D
  f
  df
  z
  dz
  K_S
  K_N
  K_G
  K̂_S
  K̂_N
  K̂_G
  ext       # EllipseGeometry for close evaluation, or nothing
  factors   # LU factorizations of the operators, made by `solve` on first use
end

Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G, ext) =
  Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G, ext, (IdDict{Any, Any}(), ReentrantLock()))

# Factorization of the operator `M` (a matrix, or a 2×2 block of matrices assembled into one), made
# once per domain. The cache is keyed by the identity of `M`, so `M` must not be modified in place
function factorization(Ω::Domain, M)
  cache, lk = getfield(Ω, :factors)
  lock(lk) do
    get!(cache, M) do
      lu(M isa AbstractMatrix{<:AbstractMatrix} ? [M[1, 1] M[1, 2]; M[2, 1] M[2, 2]] : M)
    end
  end
end

"""
    domain(γ, Mℓ, Mₘ; map=:auto)
    domain(γ, M; map=:auto)

Discretization of the inside of the closed curve `γ`, the image of the unit disk under a map
``z = F(ζ)`` with ``F(e^{it}) = γ(t)``.

# Arguments
- `γ` : the curve, sampled at equispaced parameter values ``t``, counterclockwise
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order
- `map` : the map from the disk
  - `:auto` (default): the harmonic map, or the Winslow map if the harmonic map folds (with a
    message)
  - `:harmonic`: the harmonic extension of the samples (see `harmonicmap`). The
    parametrization of `γ` places the grid: the boundary point ``e^{it}`` of the disk goes to
    ``γ(t)``. `γ = a cos t + i b sin t` gives the affine map of `ellipse`. One-to-one for convex
    curves; a map that folds is an error. No hypersingular operator `𝒩`.
  - `:winslow`: the Winslow (elliptic grid generation) map, whose inverse is harmonic (see
    `winslowmap`). One-to-one for any smooth simply connected domain, so it handles non-convex
    shapes on which the harmonic map folds, at a higher cost to build (seconds at `M = 16`). The
    parametrization of `γ` places the grid, as for `:harmonic`. Strongly deformed shapes are an
    error. No hypersingular operator `𝒩`.
  - `:conformal`: the conformal map with ``F(0) = 0`` (see `conformalmap`), which ignores the
    parametrization. The origin must be inside the curve (not checked). It crowds the grid for
    elongated shapes: a 4:1 ellipse gives capacitance errors of about 40% at `M = 32`.

# Returns
- Domain containing discretization information. `Ω.f` is the map; `Ω.df` is ``F′`` for the
  conformal map and ``ζ ↦ (∂F, \\bar∂F)`` for the harmonic and Winslow maps.

!!! warning "Experimental"
    Mapped domains are experimental: their accuracy and interface may change. The operators are
    dense (moderate `M` only), and suspensions of such bodies have no close evaluation. For
    ellipses, use `ellipse`.
"""
function domain(γ, Mℓ::Int, Mₘ::Int; map::Symbol=:auto)
  if map === :auto
    Ω = _harmonic_domain(γ, Mℓ, Mₘ; onfold=:nothing)
    Ω === nothing || return Ω
    @info "domain: the harmonic map of γ folds; using the Winslow map (map = :winslow)"
    return _winslow_domain(γ, Mℓ, Mₘ)
  end
  map === :harmonic && return _harmonic_domain(γ, Mℓ, Mₘ)
  map === :winslow && return _winslow_domain(γ, Mℓ, Mₘ)
  map === :conformal && return _conformal_domain(γ, Mℓ, Mₘ)
  throw(ArgumentError("map must be :auto, :harmonic, :winslow or :conformal, not :$map"))
end

domain(γ, M::Int; map::Symbol=:auto) = domain(γ, M, M; map=map)

function _conformal_domain(γ, Mℓ, Mₘ)

  f, df = conformalmap(γ)
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  z, z′ = similar(ζ, ComplexF64), similar(ζ, ComplexF64)
  Threads.@threads for k in eachindex(ζ)
    z[k], z′[k] = f(ζ[k]), df(ζ[k])
  end
  dz = abs2.(z′) .* dζ

  even, odd = findall(vec(D.even)), findall(vec(D.odd))
  Y, _ = _psh_factors(D)
  dw = vec(D.dw)

  G = ConformalKernels(D, ζ, z, z′, Y, dw, even, odd)
  K̂_S, K̂_N, K̂_G = _domain_operators(G, Y, dw, even, odd)

  # The grid operators K_S, K_N and K_G are formed from G when accessed (see `getproperty`)
  return Domain(D, f, df, z, dz, G, G, G, K̂_S, K̂_N, K̂_G, nothing)

end

# A harmonic map that folds is an ArgumentError, or nothing with onfold = :nothing (for map = :auto)
function _harmonic_domain(γ, Mℓ, Mₘ; onfold=:throw)

  F, α, β = harmonicmap(γ)
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  # The Jacobian |∂F|² - |∂̄F|², on the grid and on the rim, must be positive
  J = Float64[abs2(_harmonic_∂(α, x)) - abs2(_harmonic_∂̄(β, x)) for x in ζ]
  Jrim = minimum(abs2(_harmonic_∂(α, x)) - abs2(_harmonic_∂̄(β, x)) for x in cis.(range(0, 2π, 8 * length(vec(γ)) + 1)))
  if min(minimum(J), Jrim) <= 0
    onfold === :nothing && return nothing
    throw(ArgumentError("the harmonic map of γ folds (its Jacobian reaches $(min(minimum(J), Jrim))): " *
                        "use map=:winslow, or reparametrize γ"))
  end

  z = F.(ζ)
  dz = J .* dζ

  even = findall(vec(D.even))
  Y, _ = _psh_factors(D)
  dw = vec(D.dw)

  G = MappedKernels(D, _HarmonicMap(α, β), vec(J), Y, dw, even; name="harmonic map", onfold)
  G === nothing && return nothing
  K̂_S, _, K̂_G = _domain_operators(G, Y, dw, even, Int[])

  df(x) = (_harmonic_∂(α, x), _harmonic_∂̄(β, x))
  # No hypersingular operator: K_N and K̂_N are empty, as for `ellipse`
  return Domain(D, F, df, z, dz, G, [], G, K̂_S, [], K̂_G, nothing)

end

function _winslow_domain(γ, Mℓ, Mₘ)

  F = winslowmap(γ)
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  # The Jacobian of the computed map on the grid (and near the rim) must be positive
  function jac(x)
    a, b = _derivatives(F, x)
    return abs2.(a) .- abs2.(b)
  end
  J = reshape(jac(vec(ζ)), size(ζ))
  Jrim = minimum(jac(0.999 .* cis.(range(0, 2π, 8 * length(vec(γ)) + 1))))
  min(minimum(J), Jrim) > 0 ||
    throw(ArgumentError("the Winslow map of γ is not resolved (its Jacobian reaches $(min(minimum(J), Jrim)))"))

  z = F(ζ)
  dz = J .* dζ

  even = findall(vec(D.even))
  Y, _ = _psh_factors(D)
  dw = vec(D.dw)

  G = MappedKernels(D, _TabulatedDD(_mean_value_tables(F, vec(ζ))...), vec(J), Y, dw, even; name="Winslow map")
  K̂_S, _, K̂_G = _domain_operators(G, Y, dw, even, Int[])

  df(x) = (∂(F, x), ∂̄(F, x))
  return Domain(D, F, df, z, dz, G, [], G, K̂_S, [], K̂_G, nothing)

end

# ---- Singularity swapping on the grid ---------------------------------------------------------------
# The divided difference m = (z_p - z_q)/(ζ_p - ζ_q) = ρ e^{iα} (z′ on the diagonal) relates distances and
# directions on the domain to those on the disk. With the Jacobian J_q = |z′_q|² of the source point q, the
# operators on the grid are the disk's grid operators Q Ŝ P (see `_circulants`) times smooth factors:
#   K_S = (J/ρ) A,  K_N = (J/ρ³) A_N,  K_G = (J/ρ) [A I + (rotated r̂r̂ᵀ of the disk)],
# where A, A_N are the disk's single layer and hypersingular operators. The direction on the domain has angle
# ψ = ϕ + α, with ϕ the angle on the disk, so with E, Ē the disk operators of the kernels e^{±2iϕ}/|ζ - η|,
#   cos²ψ ↦ A/2 + (e^{2iα} E + e^{-2iα} Ē)/4,  sin²ψ ↦ A/2 - (…)/4,  sinψ cosψ ↦ (e^{2iα} E - e^{-2iα} Ē)/4i.
# A, A_N, E, Ē each shift the azimuthal order by a single κ, so their entries come from small tables.

const _Terms = Vector{Tuple{Vector{ComplexF64}, Array{ComplexF64, 3}}}

# The kernels of a mapped domain: each provides `_kernel_column!` (a column of the grid operators), a workspace
# for it (`_workspace`, one per thread), and whether it has the hypersingular operator
abstract type _DomainKernels end

# Tables of A, A_N, E, Ē (see `_circulants`), each a single shift: A and A_N keep the order (κ = 0), E and Ē
# shift it by ±2, with the phases eᴱ[j] = e^{2iθⱼ} and its conjugate
struct ConformalKernels <: _DomainKernels
  Nr::Int
  Nθ::Int
  ζ::Vector{ComplexF64}
  z::Vector{ComplexF64}
  z′::Vector{ComplexF64}
  S::Array{ComplexF64, 3}     # disk single layer, even modes
  N::Array{ComplexF64, 3}     # disk hypersingular operator, odd modes
  E::Array{ComplexF64, 3}     # kernel e^{2iϕ}/|ζ - η|, even modes
  Ē::Array{ComplexF64, 3}     # kernel e^{-2iϕ}/|ζ - η|, even modes
  eᴱ::Vector{ComplexF64}
end

function ConformalKernels(D, ζ, z, z′, Y, dw, even, odd)
  Nθ = size(Y, 3)
  function single(Ŝ, idx, κ)
    terms = _circulants(Ŝ, idx, Y, dw)
    (length(terms) == 1 && terms[1][1] ≈ [cis(2π * κ * (j - 1) / Nθ) for j = 1 : Nθ]) ||
      error("expected an operator shifting the azimuthal order by $κ")
    return terms[1][2]
  end
  Ê = laplace3d_angular_matrix(ϕ -> cis(2ϕ), D.Lspan, D.Mspan, even)
  Ē = laplace3d_angular_matrix(ϕ -> cis(-2ϕ), D.Lspan, D.Mspan, even)
  return ConformalKernels(size(Y, 1), Nθ, vec(ζ), vec(z), vec(z′),
                          single(sparse(Diagonal(D.K̂_S[even])), even, 0),
                          single(sparse(Diagonal(D.K̂_N[odd])), odd, 0),
                          single(Ê, even, 2), single(Ē, even, -2),
                          [cis(4π * (j - 1) / Nθ) for j = 1 : Nθ])
end

Base.show(io::IO, G::_DomainKernels) =
  print(io, "$(nameof(typeof(G))) on a $(G.Nr) × $(G.Nθ) grid (grid operators formed on access)")

_workspace(::ConformalKernels) = nothing
_has_hypersingular(::ConformalKernels) = true

# Entry (p, q) of a disk grid operator from its tables, for p = (i, j), q = (i′, j′), Δ = j - j′ mod Nθ
@inline function _circ(terms, i, j, i′, Δ)
  acc = zero(ComplexF64)
  @inbounds for (e, C) in terms
    acc += e[j] * C[i, i′, Δ]
  end
  return acc
end

# Column q (a source point) of the grid operators K_S, K_N, K_G[1, 1], K_G[1, 2], K_G[2, 2]
function _kernel_column!(kS, kN, g11, g12, g22, G::ConformalKernels, q, ws=nothing)
  Nr, Nθ = G.Nr, G.Nθ
  i′, j′ = mod1(q, Nr), cld(q, Nr)
  ζq, zq, z′q = G.ζ[q], G.z[q], G.z′[q]
  J = abs2(z′q)
  @inbounds for j = 1 : Nθ
    Δ = mod(j - j′, Nθ) + 1
    eᴱ = G.eᴱ[j]
    for i = 1 : Nr
      p = i + (j - 1) * Nr
      # m = Δz/Δζ = ρ e^{iα}: ρ² = |Δz|²/|Δζ|² and e^{2iα} = (Δz conj(Δζ))²/(|Δz|²|Δζ|²)
      if p == q
        ρ², e²ⁱᵅ = abs2(z′q), z′q^2 / abs2(z′q)
      else
        Δz, Δζ = G.z[p] - zq, G.ζ[p] - ζq
        a = Δz * conj(Δζ)
        ρ², e²ⁱᵅ = abs2(Δz) / abs2(Δζ), a^2 / abs2(a)
      end
      h = J / sqrt(ρ²)
      s = G.S[i, i′, Δ]
      e = (e²ⁱᵅ * eᴱ) * G.E[i, i′, Δ]
      ē = conj(e²ⁱᵅ * eᴱ) * G.Ē[i, i′, Δ]
      kS[p] = h * s
      kN[p] = (h / ρ²) * G.N[i, i′, Δ]
      g11[p] = h * (1.5 * s + (e + ē) / 4)
      g22[p] = h * (1.5 * s - (e + ē) / 4)
      g12[p] = h * (e - ē) * (-im / 4)
    end
  end
end

# The grid operators as dense matrices: K_S, K_N (empty without the hypersingular operator) and the 2×2 blocks
# of K_G
function _grid_matrices(G::_DomainKernels)
  N = G.Nr * G.Nθ
  hyper = _has_hypersingular(G)
  KS, G11, G12, G22 = (Matrix{ComplexF64}(undef, N, N) for _ = 1 : 4)
  KN = hyper ? Matrix{ComplexF64}(undef, N, N) : []
  chunks = collect(Iterators.partition(1 : N, cld(N, 4Threads.nthreads())))
  Threads.@threads for cs in chunks
    ws = _workspace(G)
    bN = zeros(ComplexF64, N)
    for q in cs
      _kernel_column!(view(KS, :, q), hyper ? view(KN, :, q) : bN, view(G11, :, q), view(G12, :, q),
                      view(G22, :, q), G, q, ws)
    end
  end
  K_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K_G[1, 1], K_G[1, 2], K_G[2, 1], K_G[2, 2] = G11, G12, G12, G22
  return KS, KN, K_G
end

# The operators in coefficient space, (P K Q)[idx, idx] for K_S, K_N (if any) and the blocks of K_G, without
# forming the grid operators: each column is generated, transformed and projected (rows), then the rows of the
# result are transformed and projected (columns)
function _domain_operators(G::_DomainKernels, Y, dw, even, odd)

  Nr, _, Nθ = size(Y)
  N = Nr * Nθ
  hyper = _has_hypersingular(G)
  Yₑ, nmₑ = _mode_basis(Y, even)
  Yₒ, nmₒ = _mode_basis(Y, odd)
  Rₑ, Rₒ = Yₑ .* dw, Yₒ .* dw

  # Rows, stored transposed: Tᵀ[q, p] = (P K)[idx[p], q]
  TS, T11, T12, T22 = (Matrix{ComplexF64}(undef, N, length(even)) for _ = 1 : 4)
  TN = Matrix{ComplexF64}(undef, N, hyper ? length(odd) : 0)

  plan = plan_fft!(zeros(ComplexF64, Nr, Nθ), 2; flags=FFTW.ESTIMATE | FFTW.UNALIGNED)
  chunks = collect(Iterators.partition(1 : N, cld(N, 4Threads.nthreads())))
  Threads.@threads for cs in chunks
    ws = _workspace(G)
    bS, bN, b11, b12, b22 = (zeros(ComplexF64, Nr, Nθ) for _ = 1 : 5)
    for q in cs
      _kernel_column!(bS, bN, b11, b12, b22, G, q, ws)
      for b in (bS, b11, b12, b22)
        plan * b
      end
      _project!(TS, q, bS, Rₑ, nmₑ, true, true)
      _project!(T11, q, b11, Rₑ, nmₑ, true, true)
      _project!(T12, q, b12, Rₑ, nmₑ, true, true)
      _project!(T22, q, b22, Rₑ, nmₑ, true, true)
      if hyper
        plan * bN
        _project!(TN, q, bN, Rₒ, nmₒ, true, true)
      end
    end
  end

  # Columns: row p of P K is column p of Tᵀ, and the result is written transposed back
  cols(T, Yc, nms) = _transform_columns(T, Yc, nms, Nr, Nθ, false, true)
  K̂_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K̂_G[1, 1], K̂_G[1, 2], K̂_G[2, 2] = cols(T11, Yₑ, nmₑ), cols(T12, Yₑ, nmₑ), cols(T22, Yₑ, nmₑ)
  K̂_G[2, 1] = K̂_G[1, 2]
  return cols(TS, Yₑ, nmₑ), (hyper ? cols(TN, Yₒ, nmₒ) : []), K̂_G

end

# ---- Harmonic maps ----------------------------------------------------------------------------------
# For a smooth map F that is not conformal, Δz = P Δζ + Q conj(Δζ) with the Wirtinger divided differences P, Q
# (smooth), so with μ = Q/P and ϕ the angle of Δζ on the disk,
#   1/|Δz|        = (1/|P|)    g₀(ϕ)/|Δζ|,   g₀ = |1 + μe^{-2iϕ}|⁻¹,
#   e^{2iψ}/|Δz|  = (P²/|P|³)  g₊(ϕ)/|Δζ|,   g₊ = e^{2iϕ}(1 + μe^{-2iϕ})²/|1 + μe^{-2iϕ}|³,
#   e^{-2iψ}/|Δz| = (P̄²/|P|³)  g₋(ϕ)/|Δζ|,   g₋ = conj(g₊),
# with ψ the direction of Δz. Each g = Σₖ cₖ e^{2ikϕ}, and the operators are Σₖ cₖ(μ) times the disk's grid
# operators of e^{2ikϕ}/|Δζ| (single shifts 2k, |k| ≤ Mₘ), times J/|P| and the phases. A conformal map is μ = 0;
# an affine one, constant μ (the ellipse).
#
# With z = e^{-2iϕ}, g₀ = (1 + μz)^{-1/2} (1 + μ̄/z)^{-1/2} and g₊ = z⁻¹ (1 + μz)^{1/2} (1 + μ̄/z)^{-3/2}, so with
# the binomial coefficients aⱼ, bⱼ, cⱼ of the powers -1/2, 1/2, -3/2 and s = |μ|²,
#   c⁰ₖ = μ̄ᵏ H⁰ₖ(s) (k ≥ 0, and c⁰₋ₖ = conj(c⁰ₖ)),           H⁰ₙ = Σⱼ aⱼ aⱼ₊ₙ sʲ,
#   c⁺ₖ = μ̄ᵏ⁻¹ H⁺ₖ₋₁(s) (k ≥ 1),  c⁺ₖ = μ¹⁻ᵏ H⁻₁₋ₖ(s) (k ≤ 0),  H⁺ₙ = Σⱼ bⱼ cⱼ₊ₙ sʲ,  H⁻ₙ = Σⱼ bⱼ₊ₙ cⱼ sʲ,
#   c⁻ₖ = conj(c⁺₋ₖ).
# The H are smooth on [0, max |μ|²] (singular at s = 1, a fold), so they are interpolated there by one Chebyshev
# series each, evaluated together for each pair.
#
# The map provides P, Q for the grid points p, q (`_divdiff`, with a workspace from `_map_workspace`): the harmonic map
# (`_HarmonicMap`) by its coefficients, the Winslow map (`winslowmap`) by tables of mean values (`_TabulatedDD`).
struct MappedKernels{T} <: _DomainKernels
  Nr::Int
  Nθ::Int
  ζ::Vector{ComplexF64}
  map::T
  J::Vector{Float64}
  ks::Vector{Int}                       # the k with a nonempty disk operator, |k| ≤ K
  phases::Vector{Vector{ComplexF64}}    # e^{2ikθⱼ} of each
  C::Vector{Array{ComplexF64, 3}}       # and its table
  K::Int
  smax::Float64                         # the H are interpolated on s ∈ [0, smax]
  cheb::Matrix{Float64}                 # Chebyshev coefficients: cheb[h, d + 1] of degree d for H number h
end

# H number h: H⁰ₙ is h = n + 1 (n = 0 … K), H⁺ₙ is K + 2 + n (n = 0 … K - 1), H⁻ₙ is 2K + 1 + n (n = 1 … K + 1)
_nH(K) = 3K + 2

# The disk operators of e^{2ikϕ}/|ζ - η|, which depend only on the grid: built once per (Mℓ, Mₘ)
const _HARMONIC_TABLES = Dict{Tuple{Int, Int}, Any}()
const _HARMONIC_TABLES_LOCK = ReentrantLock()

function _harmonic_tables(D, Y, dw, even)
  lock(_HARMONIC_TABLES_LOCK) do
    get!(_HARMONIC_TABLES, (D.Mℓ, D.Mₘ)) do
      Mₘ = D.Mₘ
      Nf = nextpow(2, 4Mₘ + 4)
      ks, phases, C = Int[], Vector{ComplexF64}[], Array{ComplexF64, 3}[]
      for k = -Mₘ : Mₘ
        ĝ = zeros(ComplexF64, Nf)
        ĝ[mod(2k, Nf) + 1] = 1
        terms = _circulants(laplace3d_angular_matrix(ĝ, D.Lspan, D.Mspan, even), even, Y, dw)
        isempty(terms) && continue
        length(terms) == 1 || error("expected the operator of e^{2ikϕ}/r to shift the order by 2k (k = $k)")
        push!(ks, k); push!(phases, terms[1][1]); push!(C, terms[1][2])
      end
      (ks, phases, C)
    end
  end
end

# The series H at s, from their power series (for the interpolation)
function _harmonic_H(K, s)
  L = s < 1e-300 ? 1 : ceil(Int, log(1e-18) / log(s)) + 40
  binom(p, n) = (c = ones(n + 1); for j = 1 : n; c[j + 1] = c[j] * (p - j + 1) / j; end; c)
  a, b, c = binom(-0.5, L + K + 2), binom(0.5, L + K + 2), binom(-1.5, L + K + 2)
  H = zeros(_nH(K))
  for j = L : -1 : 0                      # smallest terms first
    sʲ = s^j
    for n = 0 : K
      H[n + 1] += a[j + 1] * a[j + n + 1] * sʲ
    end
    for n = 0 : K - 1
      H[K + 2 + n] += b[j + 1] * c[j + n + 1] * sʲ
    end
    for n = 1 : K + 1
      H[2K + 1 + n] += b[j + n + 1] * c[j + 1] * sʲ
    end
  end
  return H
end

# Chebyshev coefficients of the H on [0, smax], to the degree at which they have decayed to rounding (`H(K, s)` returns
# the `nH` series at s; the volume potentials use other series, see `_abs_H`)
function _harmonic_chebyshev(K, smax; H=_harmonic_H, nH=_nH(K))
  for d in (16, 32, 64, 128, 256)
    θs = π .* ((0 : d) .+ 0.5) ./ (d + 1)
    Hs = [H(K, smax * (1 + cos(θ)) / 2) for θ in θs]
    cheb = [(m == 0 ? 1 : 2) / (d + 1) * sum(Hs[i][h] * cos(m * θs[i]) for i = 1 : d + 1) for h = 1 : nH, m = 0 : d]
    scale = maximum(abs, cheb)
    tail = maximum(abs, cheb[:, end - 1 : end])
    # The values carry rounding errors of about 1e-16 relative, so the coefficients level off there
    (tail < 1e-14 * scale || d == 256) && return cheb[:, 1 : something(findlast(m -> maximum(abs, cheb[:, m]) > 1e-16 * scale, 1 : d + 1), 1)]
  end
end

function MappedKernels(D, map, J, Y, dw, even; name="map", onfold=:throw)

  Nr, _, Nθ = size(Y)
  ζ = vec(D.ζ)
  N = length(ζ)

  # The largest |μ| over all pairs: the interval of the interpolation. |μ| ≥ 1 for some pair: the expansion in
  # e^{2ikϕ} does not converge there (for the harmonic map, a fold)
  μmax = zeros(N)
  Threads.@threads for q = 1 : N
    mws = _map_workspace(map)
    for p = 1 : N
      P, Q = _divdiff(map, p, q, ζ[p], ζ[q], mws)
      μmax[q] = max(μmax[q], abs2(Q / P))
    end
  end
  smax = maximum(μmax)
  smax < 1 || onfold !== :nothing || return nothing
  smax < 1 || throw(ArgumentError("the $name of γ is too deformed (|μ| = $(sqrt(smax)) ≥ 1 between two grid points): " *
                                  "use a less deformed curve or another map"))
  smax = max(smax * (1 + 1e-12), 1e-8)

  ks, phases, C = _harmonic_tables(D, Y, dw, even)
  K = maximum(abs, ks)
  return MappedKernels(Nr, Nθ, ζ, map, J, ks, phases, C, K, smax, _harmonic_chebyshev(K, smax))

end

_workspace(G::MappedKernels) = (zeros(_nH(G.K)), zeros(_nH(G.K)), zeros(_nH(G.K)), zeros(ComplexF64, G.K + 2),
                                _map_workspace(G.map))
_has_hypersingular(::MappedKernels) = false

# The harmonic map, by the Wirtinger divided differences of its coefficients
struct _HarmonicMap
  α::Vector{ComplexF64}
  β::Vector{ComplexF64}
end
_divdiff(m::_HarmonicMap, p, q, ζ, η, mws) = _wirtinger_dd(m.α, m.β, ζ, η)
_map_workspace(::_HarmonicMap) = nothing

# The H at s (into H), by Clenshaw's recurrence for all the series at once
@inline function _clenshaw!(H, b₁, b₂, cheb, x)
  nH, n = size(cheb)
  fill!(b₁, 0.0); fill!(b₂, 0.0)
  @inbounds for m = n : -1 : 2
    @simd for h = 1 : nH
      b = cheb[h, m] + 2x * b₁[h] - b₂[h]
      b₂[h] = b₁[h]
      b₁[h] = b
    end
  end
  @inbounds @simd for h = 1 : nH
    H[h] = cheb[h, 1] + x * b₁[h] - b₂[h]
  end
  return H
end

# Column q (a source point) of the grid operators K_S, K_G[1, 1], K_G[1, 2], K_G[2, 2] (kN is not written)
function _kernel_column!(kS, kN, g11, g12, g22, G::MappedKernels, q, ws)
  H, b₁, b₂, μⁿ, mws = ws
  Nr, Nθ, K = G.Nr, G.Nθ, G.K
  η = G.ζ[q]
  i′, j′ = mod1(q, Nr), cld(q, Nr)
  J = G.J[q]
  @inbounds for j = 1 : Nθ
    Δ = mod(j - j′, Nθ) + 1
    for i = 1 : Nr
      p = i + (j - 1) * Nr
      P, Q = _divdiff(G.map, p, q, G.ζ[p], η, mws)
      μ = Q / P
      _clenshaw!(H, b₁, b₂, G.cheb, 2abs2(μ) / G.smax - 1)
      μⁿ[1] = 1
      for n = 1 : K + 1
        μⁿ[n + 1] = μⁿ[n] * μ
      end
      s₀ = s₊ = s₋ = zero(ComplexF64)
      for t in eachindex(G.ks)
        k = G.ks[t]
        a = G.phases[t][j] * G.C[t][i, i′, Δ]
        if k >= 0
          c₀ = conj(μⁿ[k + 1]) * H[k + 1]
          c₊ = k >= 1 ? conj(μⁿ[k]) * H[K + 1 + k] : μⁿ[2] * H[2K + 2]
          c₋ = conj(μⁿ[k + 2]) * H[2K + 2 + k]
        else
          c₀ = μⁿ[1 - k] * H[1 - k]
          c₊ = μⁿ[2 - k] * H[2K + 2 - k]
          c₋ = μⁿ[-k] * H[K + 1 - k]
        end
        s₀ += c₀ * a
        s₊ += c₊ * a
        s₋ += c₋ * a
      end
      h = J / abs(P)
      P̂² = P^2 / abs2(P)
      s = h * s₀
      e = h * P̂² * s₊
      ē = h * conj(P̂²) * s₋
      kS[p] = s
      g11[p] = 1.5 * s + (e + ē) / 4
      g22[p] = 1.5 * s - (e + ē) / 4
      g12[p] = (e - ē) * (-im / 4)
    end
  end
end

# ---- The transforms from their factors --------------------------------------------------------------
# The PSH transform is an FFT in θ followed by a radial projection for each azimuthal order:
# P = blockdiag(Rₘ) (F ⊗ I) and Q = (Fᴴ ⊗ I) blockdiag(Yₘ), with Yₘ[i, ℓ] = y_ℓ^m(rᵢ), Rₘ = Yₘᴴ diag(dw)
# and F[m, j] = e^{-imθⱼ} (see `_psh_factors`). The functions below use this structure in place of the
# dense matrices P and Q.

# Radial basis Yc[:, p] = y_ℓ^m(r) of the modes idx[p], and their orders (as column indices nm)
function _mode_basis(Y, idx)
  Nr = size(Y, 1)
  nms = cld.(idx, Nr)
  Yc = similar(Y, Nr, length(idx))
  for p in eachindex(idx)
    Yc[:, p] .= view(Y, :, mod1(idx[p], Nr), nms[p])
  end
  return Yc, nms
end

# Tables of the grid operator Q[:, idx] Ŝ P[idx, :] of a sparse operator Ŝ on the coefficients `idx`. A
# block of Ŝ mapping order m to order m + κ gives e^{iκθⱼ} times a block circulant in θ:
#   (Q Ŝ P)[(i, j), (i′, j′)] = Σ_κ e^{iκθⱼ} C_κ[i, i′, j - j′],   C_κ[:, :, Δ] = Σ_m e^{imθ_Δ} Y_{m+κ} Ŝ_{m+κ,m} Rₘ
# returned as the pairs (e^{iκθ}, C_κ)
function _circulants(Ŝ::SparseMatrixCSC, idx, Y, dw)

  Nr, _, Nθ = size(Y)

  # Blocks Ŝ_{m+κ,m} of each shift κ (mod Nθ), indexed by the order m of the input
  blocks = Dict{Int, Array{ComplexF64, 3}}()
  rows, vals = rowvals(Ŝ), nonzeros(Ŝ)
  for q = 1 : size(Ŝ, 2)
    l, nm = mod1(idx[q], Nr), cld(idx[q], Nr)
    for k = nzrange(Ŝ, q)
      l′, nm′ = mod1(idx[rows[k]], Nr), cld(idx[rows[k]], Nr)
      S = get!(() -> zeros(ComplexF64, Nr, Nr, Nθ), blocks, mod(nm′ - nm, Nθ))
      S[l′, l, nm] += vals[k]
    end
  end

  terms = _Terms()
  for (κ, S) in blocks
    C = zeros(ComplexF64, Nr, Nr, Nθ)
    for nm = 1 : Nθ
      nm′ = mod1(nm + κ, Nθ)
      C[:, :, nm] .= (view(Y, :, :, nm′) * view(S, :, :, nm)) * (view(Y, :, :, nm)' .* transpose(dw))
    end
    bfft!(C, 3)
    push!(terms, ([cis(2π * κ * (j - 1) / Nθ) for j = 1 : Nθ], C))
  end
  return terms

end

# The grid operator Q[:, idx] Ŝ P[idx, :] as a dense matrix
function _grid_operator(Ŝ::SparseMatrixCSC, idx, Y, dw)
  Nr, _, Nθ = size(Y)
  N = Nr * Nθ
  terms = _circulants(Ŝ, idx, Y, dw)
  A = Matrix{ComplexF64}(undef, N, N)
  A4 = reshape(A, Nr, Nθ, Nr, Nθ)
  Threads.@threads for j′ = 1 : Nθ
    for i′ = 1 : Nr, j = 1 : Nθ
      Δ = mod(j - j′, Nθ) + 1
      for i = 1 : Nr
        A4[i, j, i′, j′] = _circ(terms, i, j, i′, Δ)
      end
    end
  end
  return A
end

# Coefficient operator (P K Q)[idx, idx] of a grid operator K. Each column of K, as an (Nr, Nθ) array,
# is transformed by the FFT in θ and projected radially onto the modes in `idx`; then each row of the
# result likewise by the inverse FFT and the radial basis
function _coefficient_operator(K, idx, Y, dw)
  Nr, _, Nθ = size(Y)
  Yc, nms = _mode_basis(Y, idx)
  Tᵀ = _transform_columns(K, Yc .* dw, nms, Nr, Nθ, true, true)     # Tᵀ[c, p] = (P K)[idx[p], c]
  return _transform_columns(Tᵀ, Yc, nms, Nr, Nθ, false, true)
end

# Out[p, c] (Out[c, p] if `transposed`) = Σᵢ conj(B[i, p]) û[i, nms[p]] (forward = true) or
# Σᵢ B[i, p] û[i, nms[p]], where û is the FFT (forward) or unnormalized inverse FFT in θ of column c of A,
# reshaped to (Nr, Nθ). The columns are threaded, each with its own buffer, executing one FFTW plan
# (execution is thread-safe; planning is not)
function _transform_columns(A, B, nms, Nr, Nθ, forward, transposed=false)
  Nc = size(A, 2)
  Out = transposed ? Matrix{ComplexF64}(undef, Nc, size(B, 2)) : Matrix{ComplexF64}(undef, size(B, 2), Nc)
  buf0 = zeros(ComplexF64, Nr, Nθ)
  flags = FFTW.ESTIMATE | FFTW.UNALIGNED
  plan = forward ? plan_fft!(buf0, 2; flags=flags) : plan_bfft!(buf0, 2; flags=flags)
  chunks = collect(Iterators.partition(1 : Nc, cld(Nc, 4Threads.nthreads())))
  Threads.@threads for cs in chunks
    buf = similar(buf0)
    for c in cs
      copyto!(buf, view(A, :, c))
      plan * buf
      _project!(Out, c, buf, B, nms, forward, transposed)
    end
  end
  return Out
end

# Out[p, c] (Out[c, p] if `transposed`) = Σᵢ conj(B[i, p]) û[i, nms[p]] (forward) or Σᵢ B[i, p] û[i, nms[p]]
function _project!(Out, c, û, B, nms, forward, transposed=false)
  Nr = size(B, 1)
  @inbounds for p in axes(B, 2)
    nm, acc = nms[p], zero(ComplexF64)
    @simd for i = 1 : Nr
      acc += (forward ? conj(B[i, p]) : B[i, p]) * û[i, nm]
    end
    if transposed
      Out[c, p] = acc
    else
      Out[p, c] = acc
    end
  end
end

const _domain_fields = fieldnames(Domain)
function Base.getproperty(Ω::Domain, s::Symbol)
  if s === :K_S || s === :K_N || s === :K_G
    K = getfield(Ω, s)
    K isa _DomainKernels || return K
    K_S, K_N, K_G = _grid_matrices(K)
    return s === :K_S ? K_S : s === :K_N ? K_N : K_G
  end
  return s ∈ _domain_fields ? getfield(Ω, s) : getproperty(getfield(Ω, :D), s)
end
