
using ProjectedSphericalHarmonics

# Discretize disk
M = 32
D = psh_disk(M)

# Get grid points and weight function
ζ = D.ζ
w = sqrt.(1 .- abs2.(ζ))

# Test single layer operator
l, m = 5, 3
u = ylm(l, m, ζ)
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4)
println("Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Test hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
println("Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Compare single layer and hypersingular operators
l, m = 5, 2
u = ylm(l, m, ζ)
err = lap(𝒮(u, D), D) - 𝒩(u, D)
println("Max error of lap(𝒮) - 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))
