
using ProjectedSphericalHarmonics

# Discretization
M = 32
D = psh_disk(M)
ζ = D.ζ

# β -> ∞ limit
σ₀ = 𝒮⁻¹(ones(size(ζ)), D)

βspan = [0.1, 1.0, 10.0, 100.0]

for (nβ, β) in enumerate(βspan)

    # Solve for concentration flux
    σ = Lσ⁻¹(β, D)
    err = abs.(σ[1] - σ₀[1])
    println("β = $β, |σ(0) - σ₀(0)| = $err")

end
