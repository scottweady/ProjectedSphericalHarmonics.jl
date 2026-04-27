println("Testing integral operators...")

# Discretize disk
Mr, Mθ = 128, 64
D = disk(Mr, Mθ)
ζ = D.ζ
w = D.w
x, y = real.(ζ), imag.(ζ)

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
