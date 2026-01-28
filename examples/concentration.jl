
using ProjectedSphericalHarmonics
using Plots

# Discretization
println("Discretizing projected spherical harmonics...")
Mr, Mθ = 32, 0
D = disk(Mr, Mθ)

# Range of β values
βspan = ComplexF64.([0.1, 1.0, 10.0, 100.0])

# β -> ∞ limit
σ₀ = 𝒮⁻¹(1, D)

for (nβ, β) in enumerate(βspan)

  # Define integral operator
  function L!(b, σ)
    b .= σ + 2β * 𝒮(σ, D)
  end

  f = fill(2β, length(D.ζ))
  σ = solve(L!, f)

  # Check difference from β -> ∞ limit
  err = abs.(σ[1] - σ₀[1])
  println("β = $β, |σ(0) - σ₀(0)| = $err")

end
