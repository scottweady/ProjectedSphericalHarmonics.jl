
using LinearAlgebra
using AssociatedLegendrePolynomials: λlm!

# ---- The Winslow (elliptic grid generation) map ------------------------------------------------------
# The inverse map ζ(z): Ω → disk is harmonic, with boundary values e^{it} at γ(t). The disk is convex, so by the
# Radó–Kneser–Choquet theorem ζ is a diffeomorphism for any simply connected Ω; its inverse F is the map from the
# disk. F is fitted by a polynomial in ζ and ζ̄ (`PolynomialMap`) from its values at the nodes of a fine grid, which
# are found by inverting ζ(z) by Newton's method.
#
# The real and imaginary parts of ζ(z) are harmonic: each is Re H for a holomorphic H = C[φ], C the Cauchy
# integral over the boundary, with a real density φ solving the second-kind equation (I/2 + K) φ = g (K the double
# layer). H is evaluated from its boundary values by the barycentric Cauchy formula, accurate up to the boundary.

# The upsampled boundary: points, derivative times the trapezoid weight, and the matrix of I/2 + K
struct _WinslowBoundary
  γ::Vector{ComplexF64}
  γ′h::Vector{ComplexF64}
  A::Matrix{Float64}
end

function _WinslowBoundary(γs; α=4)
  γs = vec(ComplexF64.(γs))
  γ′ = derivative_fourier(γs, 2π)
  γ″ = derivative_fourier(γs, 2π, 2)
  γ, γ′, γ″ = upsample(γs, α), upsample(γ′, α), upsample(γ″, α)
  N = length(γ)
  h = 2π / N
  A = Matrix{Float64}(undef, N, N)
  Threads.@threads for j = 1 : N
    @inbounds for i = 1 : N
      A[i, j] = i == j ? 0.5 + imag(γ″[i] / (2γ′[i])) * h / 2π : imag(γ′[j] * h / (γ[j] - γ[i])) / 2π
    end
  end
  return _WinslowBoundary(γ, γ′ .* h, A)
end

# Boundary values of H = C[φ] and of H′: H⁺(w₀) = φ(w₀) + (1/2πi) ∫ (φ(w) - φ(w₀))/(w - w₀) dw (a smooth integrand,
# φ′ on the diagonal), H′ = (dH⁺/dt)/γ′
function _boundary_trace(B::_WinslowBoundary, φ)
  N = length(B.γ)
  φ′h = real.(derivative_fourier(complex.(φ), 2π)) .* (2π / N)
  H = Vector{ComplexF64}(undef, N)
  Threads.@threads for i = 1 : N
    acc = complex(φ′h[i])
    @inbounds for j = 1 : N
      j == i && continue
      acc += (φ[j] - φ[i]) * B.γ′h[j] / (B.γ[j] - B.γ[i])
    end
    H[i] = φ[i] + acc / (2π * im)
  end
  return H, derivative_fourier(H, 2π) .* (2π / N) ./ B.γ′h
end

# The holomorphic function with boundary values f, at z inside: the barycentric Cauchy formula
function _cauchy(B::_WinslowBoundary, f, z)
  num = den = zero(ComplexF64)
  @inbounds for j in eachindex(B.γ)
    d = B.γ[j] - z
    iszero(d) && return complex(f[j])
    q = B.γ′h[j] / d
    num += f[j] * q
    den += q
  end
  return num / den
end

# ζ(z) = Re Hξ + i Re Hη, with its Wirtinger derivatives
struct _InverseWinslow
  B::_WinslowBoundary
  Hξ::Vector{ComplexF64}
  Hξ′::Vector{ComplexF64}
  Hη::Vector{ComplexF64}
  Hη′::Vector{ComplexF64}
end

function _InverseWinslow(γs; α=4)
  B = _WinslowBoundary(γs; α)
  F = lu(B.A)
  t = 2π .* (0 : length(B.γ) - 1) ./ length(B.γ)
  Hξ, Hξ′ = _boundary_trace(B, F \ cos.(t))
  Hη, Hη′ = _boundary_trace(B, F \ sin.(t))
  return _InverseWinslow(B, Hξ, Hξ′, Hη, Hη′)
end

function (G::_InverseWinslow)(z)
  Hξ, Hξ′ = _cauchy(G.B, G.Hξ, z), _cauchy(G.B, G.Hξ′, z)
  Hη, Hη′ = _cauchy(G.B, G.Hη, z), _cauchy(G.B, G.Hη′, z)
  return real(Hξ) + im * real(Hη), (Hξ′ + im * Hη′) / 2, (conj(Hξ′) + im * conj(Hη′)) / 2
end

# z with ζ(z) = ζ₀, by Newton's method from z; nothing if it does not converge. The steps are halved until the
# residual decreases, so that they do not leave the domain (where ζ(z) is meaningless) near a concave boundary
function _invert(G::_InverseWinslow, ζ₀, z; tol=1e-14, maxit=100)
  ζ, a, b = G(z)
  r = ζ₀ - ζ
  for _ = 1 : maxit
    δ = (conj(a) * r - b * conj(r)) / (abs2(a) - abs2(b))
    local znew, ζn, an, bn, rn
    for _ = 1 : 40
      znew = z + δ
      ζn, an, bn = G(znew)
      rn = ζ₀ - ζn
      abs(rn) < abs(r) || abs(δ) < tol * max(1, abs(z)) ? break : (δ /= 2)
    end
    z, a, b, r = znew, an, bn, rn
    abs(δ) < tol * max(1, abs(z)) && return z
  end
  return nothing
end

# ---- The map as a PSH expansion ------------------------------------------------------------------------
# F is smooth on the closed disk, so it is a polynomial in x, y to rounding: an expansion in the even PSH, which are
# orthogonal on the disk (a well-conditioned basis, unlike the monomials ζʲζ̄ᵏ). With ∂y_ℓ^m = k_ℓm y_ℓ^{m-1}/w
# (see `∂ylm∂ζ`) and conj(y_ℓ^m) = y_ℓ^{-m}, w ∂F and w ∂(F̄) are expansions in the odd PSH, and ∂̄F = conj(∂(F̄)).

"""
    WinslowMap

The Winslow map of the disk onto a domain, as computed by `winslowmap`: an expansion of degree `L` in the even PSH.
`F(ζ)` evaluates it; `∂(F, ζ)` and `∂̄(F, ζ)` are its Wirtinger derivatives.
"""
struct WinslowMap
  L::Int
  F̂::Matrix{ComplexF64}       # F = Σ F̂[ℓ + 1, m + L + 1] y_ℓ^m (even modes)
  A::Matrix{ComplexF64}       # w ∂F  = Σ A[ℓ + 1, m + L + 1] y_ℓ^m (odd modes)
  B::Matrix{ComplexF64}       # w ∂F̄ = Σ B[ℓ + 1, m + L + 1] y_ℓ^m
  tg::Vector{Float64}         # Gauss–Legendre rule on [0, 1], exact for degree L - 1 (the mean values along segments)
  wg::Vector{Float64}
end

function WinslowMap(F̂::Matrix{ComplexF64})
  L = size(F̂, 1) - 1
  col(m) = m + L + 1
  A = zeros(ComplexF64, L + 1, 2L + 1)
  B = zeros(ComplexF64, L + 1, 2L + 1)
  k(l, m) = (m <= 0 ? 1.0 : -(l + m) * (l - m + 1.0)) / 2 * Nlm(l, m, l, m - 1)
  for l = 0 : L, m = -l : l
    c = F̂[l + 1, col(m)]
    iszero(c) && continue
    abs(m - 1) <= l && (A[l + 1, col(m - 1)] += c * k(l, m))
    abs(-m - 1) <= l && (B[l + 1, col(-m - 1)] += conj(c) * k(l, -m))
  end
  tg, wg = legpts(cld(L, 2) + 1, [0.0, 1.0])
  return WinslowMap(L, F̂, A, B, vec(tg), vec(wg))
end

# Σ C[ℓ + 1, m + L + 1] y_ℓ^m(xᵢ) for the expansions C in Cs, at the points x (with the normalized Legendre values Λ of
# λlm! as workspace): sums over ℓ for each m, contiguous in the points
function _psh_sums!(out, Cs, x, Λ, s, e, eᵐ, R)
  L = size(Cs[1], 1) - 1
  K = length(x)
  @inbounds for i = 1 : K
    r = abs(x[i])
    s[i] = sqrt(max(0.0, 1 - r^2))
    e[i] = r > 0 ? x[i] / r : one(ComplexF64)
    eᵐ[i] = 1
  end
  λlm!(view(Λ, 1 : K, :, :), L, L, view(s, 1 : K))
  for o in out; fill!(view(o, 1 : K), 0); end
  @inbounds for m = 0 : L
    for (c, C) in enumerate(Cs), sgn in (m == 0 ? (1,) : (1, -1))
      col = sgn * m + L + 1
      fill!(view(R, 1 : K), 0)
      for l = m : L
        a = C[l + 1, col]
        iszero(a) && continue
        @simd for i = 1 : K
          R[i] += a * Λ[i, l + 1, m + 1]
        end
      end
      o = out[c]
      @simd for i = 1 : K
        o[i] += R[i] * (sgn > 0 ? eᵐ[i] : conj(eᵐ[i]))
      end
    end
    @simd for i = 1 : K
      eᵐ[i] *= e[i]
    end
  end
  for o in out; view(o, 1 : K) .*= sqrt(2); end
  return out
end

# Workspace for batches of K points
_psh_workspace(L, K, nout) = (ntuple(_ -> zeros(ComplexF64, K), nout), zeros(K, L + 1, L + 1), zeros(K),
                              zeros(ComplexF64, K), zeros(ComplexF64, K), zeros(ComplexF64, K))

function _psh_eval(Cs, x)
  ws = _psh_workspace(size(Cs[1], 1) - 1, length(x), length(Cs))
  return _psh_sums!(ws[1], Cs, x, ws[2 : end]...)
end

(F::WinslowMap)(x::Number) = _psh_eval((F.F̂,), [x])[1][1]
(F::WinslowMap)(x::AbstractArray) = reshape(_psh_eval((F.F̂,), vec(x))[1], size(x))
∂(F::WinslowMap, x::Number) = _psh_eval((F.A,), [x])[1][1] / sqrt(1 - abs2(x))
∂̄(F::WinslowMap, x::Number) = conj(_psh_eval((F.B,), [x])[1][1]) / sqrt(1 - abs2(x))

# ∂F and ∂̄F at the points x (inside the disk)
function _derivatives(F::WinslowMap, x)
  a, b = _psh_eval((F.A, F.B), vec(x))
  w = sqrt.(1 .- abs2.(vec(x)))
  return reshape(a ./ w, size(x)), reshape(conj.(b) ./ w, size(x))
end

# Mean-value Wirtinger divided differences between all the points ζ, P[p, q] = ∫₀¹ ∂F(ζ_q + t(ζ_p - ζ_q)) dt and
# likewise Q with ∂̄F (exact by the Gauss rule; symmetric in p, q): smooth, with |Q/P| < 1 for strongly deformed maps
function _mean_value_tables(F::WinslowMap, ζ)
  N = length(ζ)
  P = Matrix{ComplexF64}(undef, N, N)
  Q = Matrix{ComplexF64}(undef, N, N)
  ng = length(F.tg)
  batch = max(1, 1024 ÷ ng)                       # targets per batch
  Threads.@threads for q = 1 : N
    ws = _psh_workspace(F.L, batch * ng, 2)
    x = zeros(ComplexF64, batch * ng)
    for p₀ = q : batch : N                        # p ≥ q (symmetry)
      ps = p₀ : min(p₀ + batch - 1, N)
      K = length(ps) * ng
      for (b, p) in enumerate(ps), g = 1 : ng
        x[(b - 1) * ng + g] = ζ[q] + F.tg[g] * (ζ[p] - ζ[q])
      end
      a, c = _psh_sums!(ws[1], (F.A, F.B), view(x, 1 : K), ws[2 : end]...)
      for (b, p) in enumerate(ps)
        Pacc = Qacc = zero(ComplexF64)
        for g = 1 : ng
          i = (b - 1) * ng + g
          w = sqrt(1 - abs2(x[i]))
          Pacc += F.wg[g] * a[i] / w
          Qacc += F.wg[g] * conj(c[i]) / w
        end
        P[p, q] = P[q, p] = Pacc
        Q[p, q] = Q[q, p] = Qacc
      end
    end
  end
  return P, Q
end

# P, Q looked up in the tables (for MappedKernels)
struct _TabulatedDD
  P::Matrix{ComplexF64}
  Q::Matrix{ComplexF64}
end
_divdiff(T::_TabulatedDD, p, q, ζ, η, mws) = (T.P[p, q], T.Q[p, q])
_map_workspace(::_TabulatedDD) = nothing

"""
    winslowmap(γ; tol=1e-8, degrees=(16, 24, 32, 40, 48), α=4) -> F::WinslowMap

The Winslow map of the disk onto the inside of the closed curve `γ` (samples at equispaced parameter values,
counterclockwise): the map ``F`` with ``F(e^{it}) = γ(t)`` whose inverse is harmonic. It is one-to-one for any smooth
simply connected domain. ``F`` is computed at the nodes of a PSH grid of degree ``L`` and expanded in the even PSH,
with ``L`` the first of `degrees` at which it reproduces the boundary to `tol` relative to the size of the curve. `α`
is the upsampling of the boundary.
"""
function winslowmap(γ; tol=1e-8, degrees=(16, 24, 32, 40, 48), α=4)

  γ = vec(ComplexF64.(γ))
  G = _InverseWinslow(γ; α)
  scale = maximum(abs.(γ .- sum(γ) / length(γ)))
  t = 2π .* (0 : length(γ) - 1) ./ length(γ)

  z₀ = _invert(G, 0.0im, sum(γ) / length(γ))
  z₀ === nothing && throw(ArgumentError("winslowmap: Newton's method did not converge for ζ = 0"))

  F, err = nothing, Inf
  for L in degrees
    # F at the nodes of the grid of degree L, marching out along each ray
    D = disk(L)
    rs, θs = vec(D.r), D.θ[1, :]
    order = sortperm(rs)
    Z = Matrix{ComplexF64}(undef, length(rs), length(θs))
    failed = Threads.Atomic{Int}(0)
    Threads.@threads for j in eachindex(θs)
      z = z₀
      for i in order
        znew = _invert(G, rs[i] * cis(θs[j]), z)
        znew === nothing && (Threads.atomic_add!(failed, 1); znew = z)
        z = znew
        Z[i, j] = z
      end
    end
    failed[] == 0 || throw(ArgumentError("winslowmap: Newton's method did not converge at $(failed[]) points"))
    Ẑ = psh(Z, D; parity=:even)
    F̂ = zeros(ComplexF64, L + 1, 2L + 1)
    for (j, m) in enumerate(D.Mspan), l = abs(m) : L
      F̂[l + 1, m + L + 1] = Ẑ[l + 1, j]
    end
    F = WinslowMap(F̂)
    err = maximum(abs.(F(cis.(t)) .- γ)) / scale
    err < tol && return F
  end
  @warn "winslowmap: the boundary is reproduced to $err only (tol = $tol)"
  return F

end
