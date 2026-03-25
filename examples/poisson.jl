
using ProjectedSphericalHarmonics

# Degree of PSH expansion
Mr, Mθ = 64, 16

# Discretize
D = disk(Mr, Mθ)
ζ = D.ζ #grid points
w = D.w #weight function

f = -1 # right-hand side
g = 0 # boundary condition
u = Δ⁻¹(f, g, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
println("Max error in Δ⁻¹ for f = -1, g = 0: $err")