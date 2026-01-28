
using ProjectedSphericalHarmonics

# Discretize disk
Mr, Mθ = 64, 16
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w

# Single layer operator
l, m = 5, 3
u = ylm(l, m, ζ)
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4)
println("Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
println("Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Laplace operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = lap(𝒮(u, D), D) - 𝒩(u, D)
println("Max error of lap(𝒮) - 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Gradient operator
u = λlm(1, 0) .* D.w
Su = 𝒮(u, D)
∇Su = grad(Su, D)
errx = maximum(abs.(∇Su[1] + real.(ζ)/2))
erry = maximum(abs.(∇Su[2] + imag.(ζ)/2))
println("Max error in ∇𝒮 for u = λ₁,₀ w: ($errx, $erry)")

# Trace operator
u = ζ.^5
ub = trace(u, D)
err = maximum(abs.(ub - exp.(im * 5 * D.θ')))
println("Max error in trace for u = ζ^5: $err")

# Laplace solver
u = Δ⁻¹(-1, 0, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
println("Max error in Δ⁻¹ for f = -1, g = 0: $err")
