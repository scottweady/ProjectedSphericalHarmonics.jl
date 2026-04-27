using SpecialFunctions: ellipe, ellipk

println("Testing domain operators...")

# Ellipse parameters
a, b = 1.5, 1
M = 32
m = 1 - (b/a)^2

for (label, Ω) in [("angle", ellipse(a, b, M)), ("conformal", domain(range(0, 2π, 128)[1:end-1] .|> θ -> a*cos(θ) + im*b*sin(θ), M))]

  local x, y = real.(Ω.z), imag.(Ω.z)

  # Single layer for uniform charge
  σ = 𝒮⁻¹(1.0, Ω)
  local err = maximum(abs.(σ .* sqrt.(1 .- (x/a).^2 .- (y/b).^2) .- 2 / (b * ellipk(m))))
  print_error("  Max error in 𝒮⁻¹ for uniform charge ($label): ", err)

  # Drag force on ellipse translating in x direction
  f = 𝒮_st⁻¹((1.0, 0.0), Ω)
  F = integral(f[1], Ω)
  Fex = 4π / (((2a^2 - b^2) * ellipk(m) - a^2 * ellipe(m)) / (a * (a^2 - b^2)))
  print_error("  Max error in drag force ($label): ", abs(F - Fex))

end
