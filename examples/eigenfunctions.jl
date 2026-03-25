
using ProjectedSphericalHarmonics

# Degree of PSH expansion
Mr, Mθ = 64, 16

# Discretize
D = disk(Mr, Mθ)
ζ = D.ζ #grid points
w = D.w #weight function

# Define a function
l, m = 5, 3
u = ylm(l, m, ζ)

# Evaluate the single layer potential and its inverse
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4.0)
println("Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))

err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
println("Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err))) 