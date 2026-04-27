"""

Integral and differential operators on non-circular domains.

"""

function 𝒮(u, Ω::Domain)
  û = psh(u .* Ω.w, Ω, parity=:even)
  Ŝu = apply(Ω.K̂_S, û, Ω, parity=:even)
  return ipsh(Ŝu, Ω, parity=:even)
end

function 𝒮⁻¹(f, Ω::Domain)
  f̂ = psh(f, Ω, parity=:even)
  û = solve(Ω.K̂_S, f̂, Ω, parity=:even)
  return ipsh(û, Ω, parity=:even) ./ Ω.w
end

function 𝒩(u, Ω::Domain)
  û = psh(u, Ω, parity=:odd)
  N̂u = apply(Ω.K̂_N, û, Ω, parity=:odd)
  return ipsh(N̂u, Ω, parity=:odd) ./ Ω.w
end

function 𝒩⁻¹(f, Ω::Domain)
  f̂ = psh(f, Ω, parity=:odd)
  û = solve(Ω.K̂_N, f̂, Ω, parity=:odd)
  return ipsh(û, Ω, parity=:odd)
end

function 𝒮_st(μ::Tuple, Ω::Domain; η=1.0)
  μ̂ = psh(μ .* Ω.w, Ω, parity=:even)
  û = apply(Ω.K̂_G, μ̂, Ω, parity=:even) ./ η
  return ipsh(û, Ω, parity=:even)
end

function 𝒮_st⁻¹(u::Tuple, Ω::Domain; η=1.0)
  û = psh(u, Ω, parity=:even)
  f̂ = solve(Ω.K̂_G, û, Ω, parity=:even)
  return ipsh(f̂, Ω, parity=:even) ./ Ω.w
end

function ∂n(u, Ω::Domain)
  return (1 ./ abs.(Ω.df.(exp.(im * Ω.θ)))) .* ∂n(u, Ω.D)
end

function lap(u, Ω::Domain)
  return (1 ./ abs2.(Ω.df.(Ω.ζ))) .* lap(u, Ω.D)
end

function integral(u, Ω::Domain)
  return sum(u .* Ω.dz)
end
