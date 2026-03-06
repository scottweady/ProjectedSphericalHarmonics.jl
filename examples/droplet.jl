
using ProjectedSphericalHarmonics

# Dimensionless parameters
Ra, β, γ = 0, 1, 1

# List of wavenumbers
mspan = collect(0 : 8)

# Discretize
println("Discretizing...")
D = disk(64, 16)

# O(1) terms
println("Computing base state...")
σ₀ = 2β #concentration density
f₀ = (γ / 2β) * σ₀ .+ (Ra / 16) * 𝒱(σ₀, D)
p₀ = -𝒩⁻¹(f₀, D) #pressure
ψ₀ = 𝒮(p₀, D) .+ (Ra / 16) * ℬ(σ₀, D)
U₀ = -∂n(ψ₀, D) #normal velocity

# Loop over mode numbers
println("Beginning main loop...")
for (nm, m) in enumerate(mspan)

	# O(ϵ) terms
	σ₁ = 0.0 #concentration density
  f₁ = δ𝒩(p₀, m, D) .+ (γ / 2β) * σ₁ .+ (Ra / 16) * (𝒱(σ₁, D) .+ δ𝒱(σ₀, m, D))
	p₁ = -𝒩⁻¹(f₁, D) #pressure
	ψ₁ = 𝒮(p₁, D) .+ δ𝒮(p₀, m, D) .+ (Ra / 16) * (ℬ(σ₁, D) .+ δℬ(σ₀, m, D))
	U₁ = -(m + 1) * U₀ .- ∂n(ψ₁, D) #normal velocity

  # Stability coefficient
	σₘ = real.(U₁[1]) 

	# Print 
	println("(m, σₘ) = ", "($m, ", σₘ, ")")

end
