"""
  Force/torque on a translating/rotating disk on the surface of a Stokes fluid, and the force/torque
  on a disk moving out of its plane (broadside, or tilting) in free space.
"""

using ProjectedSphericalHarmonics

println("Discretizing...")
D = disk(128, 64)
x, y = real.(D.ζ), imag.(D.ζ)

# In-plane motion on the surface of a fluid (the default: μ_bottom = 1, μ_top = 0). Exact: 16/3 for both.
println("Solving...")
u = (1.0, 0.0)
f = 𝒮_st⁻¹(u, D)
F = real(sum(f[1] .* D.dζ))

println("force on a translating disk: ", F)

u = (-y, x)
f = 𝒮_st⁻¹(u, D)
T = real(sum((-y.*f[1] + x.*f[2]) .* D.dζ))
println("torque on a rotating disk: ", T)

# Normal motion in free space (one fluid of viscosity μ above and below the disk). The normal force
# density is fz = 2μ 𝒮⁻¹[uz], the Laplace single layer. Exact: 16μ broadside, 32μ/3 tilting.
μ = 1.0

u = (0.0, 0.0, 1.0)
f = 𝒮_st⁻¹(u, D; μ=μ)
Fz = real(sum(f[3] .* D.dζ))
println("force on a disk moving broadside: ", Fz)

u = (0.0, 0.0, y)
f = 𝒮_st⁻¹(u, D; μ=μ)
Tx = real(sum(y .* f[3] .* D.dζ))
println("torque on a tilting disk: ", Tx)
