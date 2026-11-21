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

# Volume potentials at rim targets, volume_matrix(Ω, n)
# Reference: polar about the rim point ζ₀ = e^{iτ}, ψ ∈ (0, π) from the tangent (chord 2 sin ψ), panels graded toward
# ψ = 0, π, and t = R u³ for the log
function rim_brute(F, J, τ, kern, f)
  ζ₀ = cis(τ); x, w = gausslegendre(30); u, wu = gausslegendre(60); u = (u .+ 1) ./ 2; wu = wu ./ 2
  bps = sort(unique(vcat(range(0, π, 9), [π/8 * 2.0^-k for k = 1 : 40], [π - π/8 * 2.0^-k for k = 1 : 40])))
  z₀ = F(ζ₀); acc = 0.0
  for i = 1 : length(bps) - 1, (q, ψ) in enumerate((bps[i] + bps[i + 1]) / 2 .+ (bps[i + 1] - bps[i]) / 2 .* x)
    e, R = cis(τ + π/2 + ψ), 2sin(ψ)
    for (ui, wi) in zip(u, wu)
      p = ζ₀ + R * ui^3 * e; p /= max(1, abs(p) / (1 - 4e-16))
      d = abs(F(p) - z₀)
      d == 0 || (acc += (bps[i + 1] - bps[i]) / 2 * w[q] * R * ui^3 * 3R * ui^2 * wi * kern(d) * f(F(p)) * J(p))
    end
  end
  return acc
end

let f = z -> exp(real(z)) * cos(imag(z)) + real(z)^2, M = 12, n = 4M + 1
  τ = 2π .* (0 : n - 1) ./ n
  Ω = ellipse(1.5, 0.67, 2M, 2M)
  V = volume_matrix(Ω, n)
  check("volume_matrix(Ω, n) vs 𝒱 at the rim points, 1.5×0.67 ellipse: ",
        maximum(abs, V * vec(f.(Ω.z)) ./ 2π .- 𝒱(1.5 .* cos.(τ) .+ 0.67im .* sin.(τ), f.(Ω.z), Ω)), 1e-13)
  t = 2π .* (0 : 127) ./ 128
  Ω = domain(@.((1 + 0.1cos(3t)) * cis(t) + 0.05cis(-2t)), 2M; map=:harmonic)
  J = p -> ((a, b) = Ω.df(p); abs2(a) - abs2(b))
  for (kern, kname) in ((log, :log), (identity, :abs))
    V = volume_matrix(Ω, n; kernel=kname)
    err = maximum(abs(transpose(V[j, :]) * vec(f.(Ω.z)) - rim_brute(Ω.f, J, τ[j], kern, f)) for j in 1 : 6 : n)
    # disk(24) resolves the density to about 1e-12 here (5e-15 at disk(32))
    check("volume_matrix(Ω, n), harmonic star, $kname, rim targets: ", err, 1e-11)
  end
end
