"""
Iterative solver for the integral equation

  σ + 2β 𝒮(σ) = 2β.

For β -> ∞ the solution converges to (4 / λ₀⁰) / w, which is the solution to  𝒮(σ) = 1.

This example demonstrates the gmres interface for solving linear integral equations. We compare
convergence towards the analytical β -> ∞ limit as a check.
"""

using ProjectedSphericalHarmonics

# Discretization
println("Discretizing...")
Mr, Mθ = 128, 64
D = disk(Mr, Mθ)

# Range of β values
βspan = ComplexF64.([0.1, 1.0, 10.0, 100.0])

# β -> ∞ limit
σ₀ = 𝒮⁻¹(1, D)

# Loop over β values
for (nβ, β) in enumerate(βspan)

  # Define integral operator in coefficient space (inefficient for now, but simple to implement)
  function L̂!(b̂, σ̂)
    σ̂ = reshape(σ̂, size(D.ζ))
    σ̂w = psh(ipsh(σ̂, D) .* D.w, D)
    b̂ .= vec(σ̂ + 2β * D.Ŝ .* σ̂w)
  end

  # Right-hand side in coefficient space
  f̂ = psh(2β, D)

  # Solve linear system
  σ̂ = gmres(L̂!, f̂)

  # Transform back to physical space
  σ = ipsh(σ̂, D)

  # Check difference from β -> ∞ limit
  err = abs.(σ[1] - σ₀[1])
  println("β = $β, |σ(0) - σ₀(0)| = $err")

end
