println("Testing differential operators...")

# Discretize disk
Mr, Mθ = 128, 64
D = disk(Mr, Mθ)
ζ = D.ζ
x, y = real.(ζ), imag.(ζ)

# Normal derivative
u = ζ.^5
∂u∂n = ∂n(u, D)
err = maximum(abs.(real.(∂u∂n) .- 5 * cos.(5 * angle.(ζ))))
check("  Max error in ∂u∂n for u = ζ⁵: ", err, 1e-8)

# Complex differentiation
u = ζ.^2 .* exp.(ζ .* conj.(ζ))
uζ = ζ.^2 .* conj.(ζ) .* exp.(ζ .* conj.(ζ)) + 2 * ζ .* exp.(ζ .* conj.(ζ))
uζ̄ = ζ.^3 .* exp.(ζ .* conj.(ζ))
errζ = maximum(abs.(∂ζ(u, D) .- uζ))
errζ̄ = maximum(abs.(∂ζ̄(u, D) .- uζ̄))
check("  Max error in ∂u/∂ζ for u = ζ² * exp(|ζ|²): ", errζ, 1e-8)
check("  Max error in ∂u/∂ζ̄ for u = ζ² * exp(|ζ|²): ", errζ̄, 1e-8)

# Gradient
u = exp.(x .* sin.(y))
∂u∂x = sin.(y) .* u
∂u∂y = x .* cos.(y) .* u
∂u∂x_num, ∂u∂y_num = grad(u, D)
errx = maximum(abs.(∂u∂x_num .- ∂u∂x))
erry = maximum(abs.(∂u∂y_num .- ∂u∂y))
check("  Max error in ∂u/∂x for u = exp(x * sin(y)): ", errx, 1e-8)
check("  Max error in ∂u/∂y for u = exp(x * sin(y)): ", erry, 1e-8)

# Laplacian
u = exp.(x .* sin.(y))
lapu = (sin.(y).^2 .- x .* sin.(y) + x.^2 .* cos.(y).^2) .* u
lapu_num = lap(u, D)
err = maximum(abs.(lapu_num .- lapu))
check("  Max error in Δu for u = exp(x * sin(y)): ", err, 1e-5)

println("Testing solvers...")

# Laplace solver
u = Δ⁻¹(-1, 0, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
check("  Max error in Δ⁻¹ for f = -1, g = 0: ", err, 1e-12)

# Trace
u = ζ.^5
Tu = trace(u, D)
err = maximum(abs.(real.(Tu) .- cos.(5 * D.θ)))
check("  Max error in trace for u = ζ⁵: ", err, 1e-12)
