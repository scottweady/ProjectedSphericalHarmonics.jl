
using FFTW

"""
    harmonicmap(γ; tol=1e-15) -> F, α, β

Harmonic extension to the unit disk of a closed curve sampled at equispaced parameter values,
``γ(t) = Σₙ γ̂ₙ e^{int}``:

```math
F(ζ) = \\sum_{n ≥ 0} α_n ζ^n + \\sum_{n ≥ 1} β_n \\bar ζ^n, \\qquad α_n = \\hat γ_n, \\quad β_n = \\hat γ_{-n},
```

so that ``F(e^{it}) = γ(t)``. Coefficients below `tol` relative to the largest are dropped (and the
Nyquist coefficient of an even number of samples). By the Radó–Kneser–Choquet theorem, ``F`` is a
diffeomorphism onto the inside of the curve when the curve is convex; otherwise it may fold.

# Returns
- `F` : the map, `F(ζ)`
- `α`, `β` : the coefficients, `α[n + 1]` of ``ζ^n`` and `β[n]` of ``\\bar ζ^n``
"""
function harmonicmap(γ; tol=1e-15)
  γ = vec(γ)
  Nγ = length(γ)
  ĝ = fft(ComplexF64.(γ)) ./ Nγ
  nmax = (Nγ - 1) ÷ 2
  scale = maximum(abs, ĝ)
  α = ĝ[1 : nmax + 1]
  β = [ĝ[Nγ - n + 1] for n = 1 : nmax]
  α = α[1 : something(findlast(x -> abs(x) > tol * scale, α), 1)]
  β = β[1 : something(findlast(x -> abs(x) > tol * scale, β), 0)]
  F(ζ) = _harmonic_eval(α, β, ζ)
  return F, α, β
end

_harmonic_eval(α, β, ζ) = evalpoly(ζ, α) + (isempty(β) ? zero(ComplexF64) : conj(ζ) * evalpoly(conj(ζ), β))

# Wirtinger derivatives ∂F = Σ n αₙ ζⁿ⁻¹ and ∂̄F = Σ n βₙ ζ̄ⁿ⁻¹
_harmonic_∂(α, ζ) = length(α) < 2 ? zero(ComplexF64) : evalpoly(ζ, [n * α[n + 1] for n = 1 : length(α) - 1])
_harmonic_∂̄(β, ζ) = isempty(β) ? zero(ComplexF64) : evalpoly(conj(ζ), [n * β[n] for n = 1 : length(β)])

# Wirtinger divided differences of F: F(ζ) - F(η) = P (ζ - η) + Q conj(ζ - η), with
# P = Σ αₙ dₙ(ζ, η), Q = Σ βₙ conj(dₙ(ζ, η)), dₙ = (ζⁿ - ηⁿ)/(ζ - η) = ζ dₙ₋₁ + ηⁿ⁻¹ (exact for ζ = η)
@inline function _wirtinger_dd(α, β, ζ, η)
  nα, nβ = length(α), length(β)
  P = Q = zero(ComplexF64)
  d, ηp = one(ComplexF64), one(ComplexF64)
  @inbounds for n = 1 : max(nα - 1, nβ)
    if n > 1
      ηp *= η
      d = ζ * d + ηp
    end
    n + 1 <= nα && (P += α[n + 1] * d)
    n <= nβ && (Q += β[n] * conj(d))
  end
  return P, Q
end
