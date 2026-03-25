"""
  Force/torque on a translating/rotating disk on the surface of a Stokes fluid.
"""

using ProjectedSphericalHarmonics

println("Discretizing...")
D = disk(128, 64)
x, y = real.(D.ζ), imag.(D.ζ)

println("Solving...")
u = (1.0, 0.0)
f = 𝒢⁻¹(u, D)
F = real(sum(f[1] .* D.dζ))

println("force on a translating disk: ", F)

u = (-y, x)
f = 𝒢⁻¹(u, D)
T = real(sum((-y.*f[1] + x.*f[2]) .* D.dζ))
println("torque on a rotating disk: ", T)
