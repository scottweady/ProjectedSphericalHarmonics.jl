
using ProjectedSphericalHarmonics

# Discretization
println("Discretizing...")
Mr, Mθ = 64, 16
D = disk(Mr, Mθ)

# Range of β values
βspan = ComplexF64.([0.1, 1.0, 10.0, 100.0])

# β -> ∞ limit
σ₀ = 𝒮⁻¹(1, D)

for (nβ, β) in enumerate(βspan)

  # Define integral operator in coefficient space
  function L̂!(b̂, σ̂)
    σ̂ = reshape(σ̂, size(D.ζ))
    σ̂w = psh(ipsh(σ̂, D) .* D.w, D)
    b̂ .= vec(σ̂ + 2β * D.Ŝ .* σ̂w)
  end

  f̂ = psh(2β, D)
  σ̂ = solve(L̂!, f̂)
  σ = ipsh(σ̂, D)

  # Check difference from β -> ∞ limit
  err = abs.(σ[1] - σ₀[1])
  println("β = $β, |σ(0) - σ₀(0)| = $err")

end
