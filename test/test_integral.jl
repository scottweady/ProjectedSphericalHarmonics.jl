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
check("  Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)), 1e-12)
err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
check("  Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)), 1e-6)

# Hypersingular operator
l, m = 5, 2
u = ylm(l, m, ζ)
err = 𝒩(u, D) - (-u ./ w ./ λlm(l, m))
check("  Max error in 𝒩 for (l,m) = ($l,$m): ", maximum(abs.(err)), 1e-6)
err = 𝒩⁻¹(u ./ w, D) - (-λlm(l, m) * u)
check("  Max error in 𝒩⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err)), 1e-10)

# Stokes operator
f₁ = -(4 / π) * y ./ w
f₂ =  (4 / π) * x ./ w
u₁, u₂ = 𝒮_st((f₁, f₂), D)
check("  Max error in 𝒮_st for f₁: ", maximum(abs.(u₁ - (-y))), 1e-12)
check("  Max error in 𝒮_st for f₂: ", maximum(abs.(u₂ - (x))), 1e-12)

u₁, u₂ = (-y, x)
f₁, f₂ = 𝒮_st⁻¹((u₁, u₂), D)
check("  Max error in 𝒮_st⁻¹ for f₁: ", maximum(abs.(f₁ - (-(4 / π) * y ./ w))), 1e-6)
check("  Max error in 𝒮_st⁻¹ for f₂: ", maximum(abs.(f₂ - ((4 / π) * x ./ w))), 1e-6)

# ── integral of a tuple: each component ──────────────────────────────────────
let D = disk(16), x = real.(D.z), y = imag.(D.z)
  I = integral((x .^ 2, 1.0, y .^ 2 .+ 1), D)
  @test I isa Tuple && length(I) == 3
  check("  integral of a tuple, componentwise (π/4, π, 5π/4): ", maximum(abs.(collect(I) .- [π/4, π, 5π/4])), 1e-13)
end
