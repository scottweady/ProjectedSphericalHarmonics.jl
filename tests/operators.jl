
using ProjectedSphericalHarmonics
using BenchmarkTools

# Discretize disk
Mr, Mθ = 64, 16
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w

# Single layer operator
l, m = 5, 3
u = ylm(l, m, ζ)
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4.0)
println("Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
println("Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
println("Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒩⁻¹(u ./ w, D) - (-λlm(l, m) * u)
println("Max error in 𝒩⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Trace operator
u = ζ.^5
ub = trace(u, D)
err = maximum(abs.(ub - exp.(im * 5 * D.θ)))
println("Max error in trace for u = ζ⁵: $err")

# Normal derivative
u = ζ.^5
∂u∂n = ∂n(u, D)
err = maximum(abs.(real.(∂u∂n) .- 5 * cos.(5 * angle.(ζ))))
println("Max error in ∂u∂n for u = ζ⁵: $err")

# Laplace solver
u = Δ⁻¹(-1, 0, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
println("Max error in Δ⁻¹ for f = -1, g = 0: $err")
