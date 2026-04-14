
using ProjectedSphericalHarmonics

function print_error(label, value, tol1=1e-6, tol2=1e-8)
  if value > tol1
      printstyled("$label$value\n"; color=:red)
  elseif value > tol2
      printstyled("$label$value\n"; color=:yellow)
  else
      printstyled("$label$value\n"; color=:green)
  end
end

# Discretize disk
Mr, Mθ = 128, 64
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
print_error("  Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
print_error("  Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
print_error("  Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)))
err = 𝒩⁻¹(u ./ w, D) - (-λlm(l, m) * u)
print_error("  Max error in 𝒩⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)))

# Stokes operator
f₁ = -(4 / π) * y ./ w
f₂ =  (4 / π) * x ./ w
u₁, u₂ = 𝒮_st((f₁, f₂), D)
print_error("  Max error in 𝒮_st for f₁: ", maximum(abs.(u₁ - (-y))))
print_error("  Max error in 𝒮_st for f₂: ", maximum(abs.(u₂ - (x))))

u₁, u₂ = (-y, x)
f₁, f₂ = 𝒮_st⁻¹((u₁, u₂), D)
print_error("  Max error in 𝒮_st⁻¹ for f₁: ", maximum(abs.(f₁ - (-(4 / π) * y ./ w))))
print_error("  Max error in 𝒮_st⁻¹ for f₂: ", maximum(abs.(f₂ - ((4 / π) * x ./ w))))

println("Testing differential operators...")

# Normal derivative
u = ζ.^5
∂u∂n = ∂n(u, D)
err = maximum(abs.(real.(∂u∂n) .- 5 * cos.(5 * angle.(ζ))))
print_error("  Max error in ∂u∂n for u = ζ⁵: ", err)

# Complex differentiation
u = ζ.^2 .* exp.(ζ .* conj.(ζ))
uζ = ζ.^2 .* conj.(ζ) .* exp.(ζ .* conj.(ζ)) + 2 * ζ .* exp.(ζ .* conj.(ζ))
uζ̄ = ζ.^3 .* exp.(ζ .* conj.(ζ))
errζ = maximum(abs.(∂ζ(u, D) .- uζ))
errζ̄ = maximum(abs.(∂ζ̄(u, D) .- uζ̄))
print_error("  Max error in ∂u/∂ζ for u = ζ² * exp(|ζ|²): ", errζ)
print_error("  Max error in ∂u/∂ζ̄ for u = ζ² * exp(|ζ|²): ", errζ̄)

# Gradient 
u = exp.(x .* sin.(y))
∂u∂x = sin.(y) .* u
∂u∂y = x .* cos.(y) .* u
∂u∂x_num, ∂u∂y_num = grad(u, D)
errx = maximum(abs.(∂u∂x_num .- ∂u∂x))
erry = maximum(abs.(∂u∂y_num .- ∂u∂y))
print_error("  Max error in ∂u/∂x for u = exp(x * sin(y)): ", errx)
print_error("  Max error in ∂u/∂y for u = exp(x * sin(y)): ", erry)

# Laplacian
u = exp.(x .* sin.(y))
lapu = (sin.(y).^2 .- x .* sin.(y) + x.^2 .* cos.(y).^2) .* u
lapu_num = lap(u, D)
err = maximum(abs.(lapu_num .- lapu))
print_error("  Max error in Δu for u = exp(x * sin(y)): ", err)

println("Testing solvers...")

# Laplace solver
u = Δ⁻¹(-1, 0, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
print_error("  Max error in Δ⁻¹ for f = -1, g = 0: ", err)

# Trace
u = ζ.^5
Tu = trace(u, D)
err = maximum(abs.(real.(Tu) .- cos.(5 * D.θ)))
print_error("  Max error in trace for u = ζ⁵: ", err)
