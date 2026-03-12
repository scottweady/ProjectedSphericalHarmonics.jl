using ProjectedSphericalHarmonics
using LinearAlgebra


# function GenerateMatrixFreq(lmax, m, Mr, Mθ )

#     if m<0
#         error("Negative m not implemented yet")
#     end
#     #we assume m is odd
#     #We assume lmax is even, otherwise we have to add one more row and column to the matrix

#     eⱼ = zeros(ComplexF64, lmax+1)

#     D = disk(Mr, Mθ)

#     ζ = D.ζ
#     w = D.w
#     r, θ = abs.(ζ), angle.(ζ)

#     u = ylm.(0, 0, ζ)

#     num_terms = (lmax - m)÷2 + 1
#     A = zeros(ComplexF64, lmax+1, lmax)

#     for l = (m+1):2:lmax
#         u .= ylm.(l, m, ζ)
#         Au = S(u, D)
#         for k = 0:lmax-1
#             A[k+1, l+1] = sum(w .* Au .* conj.(ylm.(k, m, ζ)))
#         end
#     end


# end


function ∂ζ̄Sylmₗ(ζ, l, m )
    

    if l == m
        return ζ*0
    end    
    α = (2(l-1)+1)/(l-1-m+1) * Nlm(l,m, l-1, m) 
    β = -(l-1+m)/(l-1-m+1) * Nlm(l,m, l-2, m)

    return α*λlm(l-1,m+1)/8*ylm.(l-1,m+1, ζ)* Nlm(l-1,m, l-1, m+1)  + β*∂ζ̄Sylmₗ(ζ, l-2, m )

end




# Discretize disk
Mr, Mθ = 64, 10
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w
r, θ = abs.(ζ), angle.(ζ)

# Single layer operator
l, m = 5, 1


u = ylm.(l, m, ζ)  ./ D.w #+ ylm.(l, m, ζ) +  4*ylm.(l+2, m, ζ)
Su = 𝒮(u, D)

∂ζ_Su = conj.(∂ζ(conj.(Su), D))

abs.(psh(Su, D))[1:8,1:5] .> 10^-10
abs.(psh(∂ζ_Su, D))[1:8,1:5] .> 10^-10 

∂ζ_Su_2 =  ∂ζ̄Sylmₗ.(ζ, l, m )
norm(∂ζ_Su - ∂ζ_Su_2)

coef_grad_big = psh(∂ζ_Su, D)[1:10, m+2] 
coef_compos_big =psh(∂ζ_Su_2, D)[1:10, m+2] 

norm(coef_grad_big - coef_grad_small)
norm(coef_compos_big - coef_compos_small)


######################



##################

p = ylm.(l, m, ζ);
∂ζp = ∂ζ(p, D);
∂ζp = ∂ζp .* (abs.(∂ζp) .> 1e-12);
∂ζ̄p = conj.(∂ζ(conj.(p), D));
∂ζ̄p = ∂ζ̄p .* (abs.(∂ζ̄p) .> 1e-12);
𝒩⁻¹∂ζp = 𝒩⁻¹(∂ζp, D);
𝒩⁻¹∂ζ̄p = 𝒩⁻¹(∂ζ̄p, D);

∂ζ̄𝒩⁻¹∂ζp = conj(∂ζ(conj.( 𝒩⁻¹∂ζp.*D.w ), D))./D.w  + 𝒩⁻¹(∂ζp, D) ./D.w.^2 .* D.ζ/2; 
∂ζ𝒩⁻¹∂ζ̄p = ∂ζ( 𝒩⁻¹∂ζp.*D.w , D) ./D.w  + 𝒩⁻¹(∂ζp, D)./D.w.^2 .* conj(D.ζ)/2; 

abs.(psh(p  , D;parity =:even))[1:10,1:5] .> 10^-10
abs.(psh(𝒩⁻¹∂ζp  , D;parity =:odd))[1:10,1:5] .> 10^-10
abs.(psh(𝒩⁻¹∂ζ̄p  , D;parity =:odd))[1:10,1:5] .> 10^-10
abs.(psh(∂ζ̄𝒩⁻¹∂ζp .* D.w  , D;parity =:even))[1:10,1:5] .> 10^-7

