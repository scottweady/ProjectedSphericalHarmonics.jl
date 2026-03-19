using ProjectedSphericalHarmonics

#We start by checking some identities

l = 8
m = 4
Mr = 40
D = disk(Mr)

X = real.(D.ζ)
Y = imag.(D.ζ)

# p = exp.(-X.*Y).*sin.(Y.^2)
p = ylm(l,m, D.ζ) + ylm(2,0, D.ζ) + 10*ylm(0,0, D.ζ) + 3im*ylm(l,-m, D.ζ)
# p = ylm(l,-m, D.ζ) +  ylm(l,m, D.ζ)
# p = ylm(2,0, D.ζ) + ylm(0,0, D.ζ) 

∂p = ∂ζ(p,D)
N⁻¹∂p = 𝒩⁻¹(∂p, D)
SN⁻¹∂p = 𝒮(N⁻¹∂p, D)
∂̄SN⁻¹∂p = ∂ζ̄(SN⁻¹∂p, D)
 
∂̄p = ∂ζ̄(p,D)
N⁻¹∂̄p = 𝒩⁻¹(∂̄p, D)
SN⁻¹∂̄p = 𝒮(N⁻¹∂̄p, D)
∂SN⁻¹∂̄p = ∂ζ(SN⁻¹∂̄p, D)
∇SN⁻¹∇p = 2*(∂̄SN⁻¹∂p + ∂SN⁻¹∂̄p)


p̂ = psh_triangular(p,D)
p̂₀ = mode_coefficients(p̂, 0)[1]

norm( p .- p̂₀*ylm(0,0, D.ζ) - ∇SN⁻¹∇p, Inf)
