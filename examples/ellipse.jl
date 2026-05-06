using SpecialFunctions: loggamma, ellipe, ellipk
using ProjectedSphericalHarmonics
using Plots

# Resolution
M = 32

# Ellipse parameters
a,b = 1.5, 1

θ = range(0, 2π, 128)[1:end-1]
γ = a * cos.(θ) + im * b * sin.(θ)

for (n, Ω) in enumerate([ellipse(a, b, M), domain(γ, M)])

  if n == 1
    println("Using angle discretization")
  else
    println("Using conformal discretization")
  end

  # Compute single layer for a uniform charge
  σ = 𝒮⁻¹(1.0, Ω)
  x, y, = real.(Ω.z), imag.(Ω.z)
  err = @. σ * sqrt.(1 .- (x/a).^2 .- (y/b).^2) - 2 / (b * ellipk(1 - (b/a)^2))
  println("Error in single layer potential: ", maximum(abs.(err)))

  # Compute drag force on an ellipse translating in the x direction
  u = (1.0, 0.0)
  f = 𝒮_st⁻¹(u, Ω)
  F = integral(f[1], Ω)

  m = 1 - (b/a)^2
  A = ((2a^2 - b^2) * ellipk(m) - a^2 * ellipe(m)) / (a * (a^2 - b^2))
  Fex = 4π / A

  println("Error in drag force on an ellipse: ", abs(F - Fex))
end