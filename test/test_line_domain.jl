println("Testing line integrals on domains...")

using FastGaussQuadrature: gausslegendre

# Reference: Gauss panels refined geometrically (to 1e-10) toward the rim point arg ζ, the kernel by subtraction
function line_brute(F, ζ, kern, g)
  τ₀ = angle(ζ); x, w = gausslegendre(30)
  bps = sort(unique(vcat(τ₀ .+ range(-π, π, 65), τ₀, [τ₀ + s * 1e-10 * 2.0^k for s in (-1, 1) for k = 0 : 29])))
  z₀ = F(ζ)
  return sum((bps[i + 1] - bps[i]) / 2 * w[q] * kern(abs(z₀ - F(cis(t)))) * g(t)
             for i = 1 : length(bps) - 1 for (q, t) in enumerate((bps[i] + bps[i + 1]) / 2 .+ (bps[i + 1] - bps[i]) / 2 .* x))
end

let g = τ -> exp(cos(τ + 0.3)) + 0.5sin(2τ), M = 12, n = 4M + 1
  τ = 2π .* (0 : n - 1) ./ n
  t = 2π .* (0 : 127) ./ 128
  star = @. (1 + 0.1cos(3t)) * cis(t) + 0.05cis(-2t)
  for (name, Ω) in (("ellipse 1.5×0.67", ellipse(1.5, 0.67, 2M, 2M)), ("harmonic star", domain(star, 2M; map=:harmonic)))
    ζ = vec(ComplexF64.(Ω.ζ))
    outer = findall(abs.(ζ) .>= maximum(abs.(ζ)) - 1e-14)[1 : 5 : end]   # 1 - |ζ| ≈ 2.6e-6
    for (kern, kname) in ((log, :log), (identity, :abs))
      L = line_matrix(Ω, n; kernel=kname)
      err = maximum(abs(L[p, :]' * g.(τ) - line_brute(Ω.f, ζ[p], kern, g)) for p in outer)
      check("line_matrix $name, $kname, outermost ring: ", err, 1e-12)
    end
  end
  # divdiff on the rim reproduces the map
  Ω = domain(star, 2M; map=:harmonic)
  ζ, η = 0.3 + 0.9im, cis(1.2)
  P, Q = divdiff(Ω, ζ, η)
  check("divdiff, harmonic map: ", abs(Ω.f(ζ) - Ω.f(η) - P * (ζ - η) - Q * conj(ζ - η)), 1e-14)
end
