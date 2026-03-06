
using ProjectedSphericalHarmonics

# Discretize disk
Mr, Mθ = 64, 16
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w
x, y = real.(ζ), imag.(ζ)

println("Testing integral operators...")

# Single layer operator
l, m = 5, 3
u = ylm(l, m, ζ)
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4.0)
println("  Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
println("  Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
println("  Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒩⁻¹(u ./ w, D) - (-λlm(l, m) * u)
println("  Max error in 𝒩⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

println("Testing differential operators...")

# Normal derivative
u = ζ.^5
∂u∂n = ∂n(u, D)
err = maximum(abs.(real.(∂u∂n) .- 5 * cos.(5 * angle.(ζ))))
println("  Max error in ∂u∂n for u = ζ⁵: $err")

# Complex differentiation
u = ζ.^2 .* exp.(ζ .* conj.(ζ))
uζ = ζ.^2 .* conj.(ζ) .* exp.(ζ .* conj.(ζ)) + 2 * ζ .* exp.(ζ .* conj.(ζ))
uζ̄ = ζ.^3 .* exp.(ζ .* conj.(ζ))
errζ = maximum(abs.(∂ζ(u, D) .- uζ))
errζ̄ = maximum(abs.(∂ζ̄(u, D) .- uζ̄))
println("  Max error in ∂u/∂ζ for u = ζ² * exp(|ζ|²): $errζ")
println("  Max error in ∂u/∂ζ̄ for u = ζ² * exp(|ζ|²): $errζ̄")

# Gradient 
u = exp.(x .* sin.(y))
∂u∂x = sin.(y) .* u
∂u∂y = x .* cos.(y) .* u
∂u∂x_num, ∂u∂y_num = grad(u, D)
errx = maximum(abs.(∂u∂x_num .- ∂u∂x))
erry = maximum(abs.(∂u∂y_num .- ∂u∂y))
println("  Max error in (∂u/∂x, ∂u/∂y) for u = exp(x * sin(y)): ($errx, $erry)")

# Laplacian
u = exp.(x .* sin.(y))
lapu = (sin.(y).^2 .- x .* sin.(y) + x.^2 .* cos.(y).^2) .* u
lapu_num = lap(u, D)
err = maximum(abs.(lapu_num .- lapu))
println("  Max error in Δu for u = exp(x * sin(y)): $err")

println("Testing solvers...")

# Laplace solver
u = Δ⁻¹(-1, 0, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
println("  Max error in Δ⁻¹ for f = -1, g = 0: $err")
