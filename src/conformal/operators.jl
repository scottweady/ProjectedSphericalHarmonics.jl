"""

Integral and differential operators on non-circular domains.

"""

function 𝒮(u, Ω::Domain)
  û = psh(u .* Ω.D.w, Ω.D, parity=:even)
  Ŝu = apply(Ω.K̂_S, û, Ω.D, parity=:even)
  return ipsh(Ŝu, Ω.D, parity=:even)
end

function 𝒮⁻¹(f, Ω::Domain)
  f̂ = psh(f, Ω.D, parity=:even)
  û = solve(Ω.K̂_S, f̂, Ω.D, parity=:even)
  return ipsh(û, Ω.D, parity=:even) ./ Ω.D.w
end

function 𝒩(u, Ω::Domain)
  û = psh(u, Ω.D, parity=:odd)
  N̂u = apply(Ω.K̂_N, û, Ω.D, parity=:odd)
  return ipsh(N̂u, Ω.D, parity=:odd) ./ Ω.D.w
end

function 𝒩⁻¹(f, Ω::Domain)
  f̂ = psh(f, Ω.D, parity=:odd)
  û = solve(Ω.K̂_N, f̂, Ω.D, parity=:odd)
  return ipsh(û, Ω.D, parity=:odd)
end

function 𝒮_st(μ::Tuple, Ω::Domain; η=1.0)
  μ̂ = psh(μ, Ω.D, parity=:even) .* Ω.D.w
  û = apply(Ω.K̂_G, μ̂, Ω.D, parity=:even)
  return ipsh(û, Ω.D, parity=:even)
end

function 𝒮_st⁻¹(u::Tuple, Ω::Domain; η=1.0)
  û = psh(u, Ω.D, parity=:even)
  f̂ = solve(Ω.K̂_G, û, Ω.D, parity=:even)
  return ipsh(f̂, Ω.D, parity=:even) ./ Ω.D.w
end

function ∂n(u, Ω::Domain)
  return (1 ./ abs.(Ω.df.(exp.(im * Ω.D.θ)))) .* ∂n(u, Ω.D)
end

function lap(u, Ω::Domain)
  return (1 ./ abs2.(Ω.df.(Ω.D.ζ))) .* lap(u, Ω.D)
end

function integral(u, Ω::Domain)
  return sum(u .* Ω.dz)
end