using ProjectedSphericalHarmonics
using LinearAlgebra
using BandedMatrices
using FFTW
using FastAlmostBandedMatrices
using Revise



# function Inverse_laplacian_coef_m_positive(f̂ᵐ,lmax, m; aliasing = true)

#     #f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
#     #We recall that l+m has to be even, otherwise is 0
#     #We ommit the negative m, since the coefficients are there cvonjugate


#     Δ⁻¹f̂ᵐ = zeros(ComplexF64, length(f̂ᵐ)+2*aliasing)

#     Δ⁻¹f̂ᵐ[1] += -1/( (2m+3)*(2m+1))*f̂ᵐ[1] 
#     Δ⁻¹f̂ᵐ[3] += -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )*f̂ᵐ[1] 

#     for l in m+2:lmax

#         i = (l-m) + 1 

#         if (l+m) % 2 == 0

#             # Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
#             Δ⁻¹f̂ᵐ[i] += -2/((2l-1)*(2l+3))*f̂ᵐ[i] 
#             Δ⁻¹f̂ᵐ[i-2] += -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m)*f̂ᵐ[i]
#             if l <lmax && l+1 < lmax
#                 Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
#             elseif l == lmax && aliasing || (l == lmax - 1 && aliasing)
#                 Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i]
#             end 
        
#         end

#     end

#     return Δ⁻¹f̂ᵐ


# end

# function Inverse_laplacian_coef_m(f̂ᵐ,lmax, m; aliasing = true)

#     #f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
#     #We recall that l+m has to be even, otherwise is 0
#     #We ommit the negative m, since the coefficients are there cvonjugate

#     if m < 0
#         return conj.(Inverse_laplacian_coef_m_positive(conj.(f̂ᵐ), lmax, -m; aliasing = aliasing))
#     else
#         return Inverse_laplacian_coef_m_positive(f̂ᵐ, lmax, m; aliasing = aliasing)
#     end


# end


# function Inverse_laplacian_coef_m_sparse_positive(f̂ᵐ,lmax, m; aliasing = true)

#     # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₂, , ..., f̂ᵐₗ_ₘₐₓ]
#     # We recall that l+m has to be even, otherwise is 0
#     # We ommit the negative m, since the coefficients are there cvonjugate


#     Δ⁻¹f̂ᵐ = zeros(ComplexF64, length(f̂ᵐ)+1*aliasing)

#     Δ⁻¹f̂ᵐ[1] += -1/( (2m+3)*(2m+1))*f̂ᵐ[1] 
#     Δ⁻¹f̂ᵐ[2] += -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )*f̂ᵐ[1] 

#     for l in m+2:2:lmax

#         i = (l-m)÷2 + 1 
#         # Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
#         Δ⁻¹f̂ᵐ[i] += -2/((2l-1)*(2l+3))*f̂ᵐ[i] 
#         Δ⁻¹f̂ᵐ[i-1] += -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m)*f̂ᵐ[i]
#         if (l < lmax && l+1 < lmax) || aliasing
#             Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
#         end 


#     end

#     return Δ⁻¹f̂ᵐ


# end


# function Inverse_laplacian_coef_m_sparse(f̂ᵐ,lmax, m; aliasing = true)

#     # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₂, , ..., f̂ᵐₗ_ₘₐₓ]
#     # We recall that l+m has to be even, otherwise is 0
#     # We ommit the negative m, since the coefficients are there cvonjugate

#     if m < 0
#         return conj.(Inverse_laplacian_coef_m_sparse_positive(conj.(f̂ᵐ), lmax, -m; aliasing = aliasing))
#     else
#         return Inverse_laplacian_coef_m_sparse_positive(f̂ᵐ, lmax, m; aliasing = aliasing)
#     end

# end


# function Inverse_laplacian(f̂)

#     #Here we assume f̂ has a matrix structure, where the first column corresponds to m=0, the second to m=1, etc. and the last to m=-1, etc.
#     lmax = size(f̂, 1) - 1
#     Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))
#     f̂⁰ = @view(f̂[:, 1])
#     m = 0
#     Δ⁻¹f̂⁰ = Inverse_laplacian_coef_m(f̂⁰,lmax, 0; aliasing = false)
#     Δ⁻¹f̂[:, 1] .= Δ⁻¹f̂⁰

#     for m in 1:div(size(f̂, 2)-1, 2)
#         f̂ᵐ = @view(f̂[m+1:end, m+1])
#         Δ⁻¹f̂ᵐ = Inverse_laplacian_coef_m(f̂ᵐ, lmax, m; aliasing = false)
#         Δ⁻¹f̂[m+1:end, m+1] .= Δ⁻¹f̂ᵐ[1:end]
#         f̂⁻ᵐ = @view(f̂[m+1:end, end-(m-1)])
#         Δ⁻¹f̂⁻ᵐ = Inverse_laplacian_coef_m(f̂⁻ᵐ, lmax, -m; aliasing = false)
#         Δ⁻¹f̂[m+1:end, end-(m-1)] .= Δ⁻¹f̂⁻ᵐ[1:end]
#     end

#     return Δ⁻¹f̂

# end

# function GenerateSparseMatrix_InverseLaplacian(lmax, m; rectangular = false)
#     # This function generates the sparse matrix that represents the inverse Laplacian operator for a given m and lmax
#     # The matrix will have dimensions (ceil((lmax-m)/2), ceil((lmax-m)/2)) since we are only considering the coefficients with l+m even

#     n = ceil(Int, (lmax+1 - m) / 2)
#     A = BandedMatrix(BandedMatrices.FillArrays.Zeros(n+rectangular,n), (1,1))
    
#     A[1,1] = -1/( (2m+3)*(2m+1))
#     A[2,1] = -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )

#     for l in m+2:2:lmax

#         i = (l-m)÷2 + 1 
#         # Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i]
#         A[i,i] = -2/((2l-1)*(2l+3))
#         A[i-1, i] = -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m) 
        
#         if l < lmax && l+1 < lmax ||rectangular
#             A[i+1, i] = -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m) 
#         end 


#     end

#     return A

# end




# function DirichletTraceInverseLaplacian_m(f̂ᵐ, lmax, m)

#     #Computes the m-th Fourier mode of the trace of the inverse Laplacian of a function with coefficients f̂ᵐ on the ylm basis. This is used to compute the boundary value correction in the Poisson solver.
#     ûᵐ = Inverse_laplacian_coef_m(f̂ᵐ,lmax, m; aliasing = true)

#     traceûᵐ = 0
#     for l in m:(lmax+2)
#         if (l+m) % 2 == 0
#             traceûᵐ .+= ûᵐ[l-m+1] * ylm(l, m, 1.0)
#         end
#     end

#     return traceûᵐ

# end


# function DirichletTraceColumnLaplacian_m(Am, m)

#     l = m:2:size(Am, 1)*2 .+ (m - 2)
#     trace_column_yml = zeros(ComplexF64, size(Am, 1))
#     for i in 1:size(Am, 1)
#         trace_column_yml[i] = ylm(l[i], m, 1.0)
#     end

#     return Am'* trace_column_yml

# end

# function ModifiedPoissonSystemMatrix(lmax, m, α)

#     #matrix system for equation (αI + Δ̂ᵐ ) ûᵐ  = f̂ᵐ
#     # u = SN⁻¹μ + ylm(m,m, ζ) 
#     # ylm is harmonic

#     A = GenerateSparseMatrix_InverseLaplacian(lmax, m; rectangular = true)
#     Full_system_matrix = zeros(eltype(A), size(A) .+ (0,1) )
#     Full_system_matrix = AlmostBandedMatrix(brand(Float64, size(A)[1], size(A)[1], 1, 1), rand(Float64, 1, size(A)[1]))
#     # n = size(A, 1)
#     # AlmostBandedMatrix(brand(eltype(A), n, n, m + 1, m), rand(Float64, m, n))

#     trace_column = DirichletTraceColumnLaplacian_m(A, m)
#     Full_system_matrix[2:end, 2:end] .= α*A[1:end-1,:] + I
#     Full_system_matrix[2, 1] = α
#     Full_system_matrix[1,1] = ylm(m, m, 1.0)
#     Full_system_matrix[1, 2:end] .= trace_column
    

#     return Full_system_matrix


# end


# function ∂ζΔ⁻¹_m_sparse( μ̂ₘ , m, lmax )

#     #Sends to the (m-1) vector
#     if m < 0
#         return conj.(∂_ζΔ⁻¹_m(conj.(μ̂ₘ), D, -m))
#     end

#     ∂_ζΔ⁻¹μ̂ᵐ = zeros(ComplexF64, length(μ̂ₘ) + 1)


#     if m == 0
#         ∂_ζΔ⁻¹μ̂ᵐ[1] = 0
#     else
#         ∂_ζΔ⁻¹μ̂ᵐ[1] = Nlm(m, m, m-1, m-1)*μ̂ₘ[1]
#     end

#     for l in m+2:2:lmax




#     end



# end


# function ∇Δ⁻¹( μ̂ , D )

#     #Here we assume f̂ has a matrix structure, where the first column corresponds to m=0, the second to m=1, etc. and the last to m=-1, etc.
#     lmax = size(f̂, 1) - 1

#     #Gradient generates mode mixing

#     Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))
#     f̂⁰ = @view(f̂[:, 1])
#     m = 0



#     # Δ⁻¹f̂⁰ = Inverse_laplacian_coef_m(f̂⁰,lmax, 0; aliasing = false)
#     # Δ⁻¹f̂[:, 1] .= Δ⁻¹f̂⁰



# end



# Discretize disk
Mr, Mθ = 32, 16
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w
r, θ = abs.(ζ), angle.(ζ)

# setting test function, fix m
l, m = 8, 0


u = 2*ylm.(abs(m), m, ζ) + ylm.(l, m, ζ) +  4*ylm.(l+2, m, ζ)

#Expected result
u_inv_alt = 𝒮(𝒩⁻¹(u, D), D)


#Constructing with standard coefficient indexing

psh_coeffs = psh(u, D)

ûᵐ = zeros(ComplexF64, length(psh_coeffs[abs(m)+1:end, 1]))

if m >=0
    ûᵐ .= psh_coeffs[m+1:end, m+1]
else
    ûᵐ .= psh_coeffs[abs(m)+1:end, end-(abs(m)-1)]
end


abs.(ûᵐ) .> 1e-10
Δ⁻¹ûᵐ = Inverse_laplacian_coef_m(ûᵐ,Mr, m)

psh_coeffs_inv = zeros(ComplexF64, size(psh_coeffs))
if m >= 0
    psh_coeffs_inv[m+1:end, m+1] .= Δ⁻¹ûᵐ[1:end-2]
else
    psh_coeffs_inv[abs(m)+1:end, end-(abs(m)-1)] .= conj.(Δ⁻¹ûᵐ[1:end-2])
end
# psh_coeffs_inv[m+1:end, m+1] = Δ⁻¹ûᵐ[1:end-2]
u_inv = ipsh(psh_coeffs_inv, D)

norm(u_inv - u_inv_alt) < 1e-12


#Constructing with sparse coefficient indexing
ûᵐ_sparse = zeros(ComplexF64, ceil(Int, length(ûᵐ)/2))
ûᵐ[1:2:end]
ûᵐ_sparse .= ûᵐ[1:2:end]
Δ⁻¹ûᵐ_sparse = Inverse_laplacian_coef_m_sparse(ûᵐ_sparse, Mr, m)
psh_coeffs_inv_sparse = zeros(ComplexF64, size(psh_coeffs))
if m >= 0
    psh_coeffs_inv_sparse[m+1:2:end, m+1] .= Δ⁻¹ûᵐ_sparse[1:end-1]
else
    psh_coeffs_inv_sparse[abs(m)+1:2:end, end-(abs(m)-1)] .= conj.(Δ⁻¹ûᵐ_sparse[1:end-1])
end
# psh_coeffs_inv_sparse[m+1:2:end, m+1] .= Δ⁻¹ûᵐ_sparse[1:end-1]
u_inv_sparse = ipsh(psh_coeffs_inv_sparse, D)   

norm(u_inv_sparse - u_inv_alt) < 1e-12




#Testing wrapper
û = psh(u, D)
Δ⁻¹û = Inverse_laplacian(û)
norm(ipsh(Δ⁻¹û, D) - u_inv_alt) < 1e-12


#General function Inverse Laplacian test


x = real.(ζ)
y = imag.(ζ) 
u = exp.(-x.*y).*cos.(x.^2)

#Expected result
u_inv_alt = 𝒮(𝒩⁻¹(u, D), D)

#coefficient solver

û = psh(u, D)
Δ⁻¹û = Inverse_laplacian(û)
norm(ipsh(Δ⁻¹û, D) - u_inv_alt) < 1e-12



#Testing the sparse matrix construction
Am = GenerateSparseMatrix_InverseLaplacian(Mr, m)

ûᵐ_sparse = zeros(ComplexF64, ceil(Int, length(ûᵐ)/2))
ûᵐ_sparse .= ûᵐ[1:2:end]
# ûᵐ_sparse[2] = 1.0

Δ⁻¹ûᵐ_sparse = Inverse_laplacian_coef_m_sparse(ûᵐ_sparse, Mr, m; aliasing = false)
Δ⁻¹ûᵐ_matrix = Am * ûᵐ_sparse

norm(Δ⁻¹ûᵐ_sparse - Δ⁻¹ûᵐ_matrix)

#Testing rectangular matrix construction
Am = GenerateSparseMatrix_InverseLaplacian(Mr, m; rectangular = true)

ûᵐ_sparse = zeros(ComplexF64, ceil(Int, length(ûᵐ)/2))
# ûᵐ_sparse .= ûᵐ[1:2:end]
ûᵐ_sparse[end] = 1.0

Δ⁻¹ûᵐ_sparse = Inverse_laplacian_coef_m_sparse(ûᵐ_sparse, Mr, m; aliasing = true)
Δ⁻¹ûᵐ_matrix = Am * ûᵐ_sparse

norm(Δ⁻¹ûᵐ_sparse - Δ⁻¹ûᵐ_matrix)


#Modified Poisson system matrix


#Zero of Bessel Function

m = 0

a00 = 2.40482555769577
a04 =11.7915344390142816137430449
Big_System00 = ModifiedPoissonSystemMatrix(Mr, m, a00^2)
Big_System04 = ModifiedPoissonSystemMatrix(Mr, m, a04^2)
U, Σ0, Vᵗ = svd(Big_System00);
U, Σ4, Vᵗ = svd(Big_System04);

Σ0[end] < 10^(-14)
Σ4[end] < 10^(-14)

Σ0[end-1] > 10^(-14)
Σ4[end-1] > 10^(-14)

#Testing ∂ζΔ⁻¹ and ∂ζ̄Δ⁻¹

# setting test function, fix m
#Positive m

#To do: the aliasing for derivative is wrong, going from even to odd m changes the factor.



l, m = 7, 3 

μ = ylm.(abs(m), m, ζ) +  4*ylm.(l+2, m, ζ)
# μ = ylm.(abs(m), m, ζ)

μ̂ = psh(μ , D)
Δ⁻¹μ̂ = Inverse_laplacian(μ̂)
Δ⁻¹μ = ipsh(Δ⁻¹μ̂, D)


#∂ζ

partial_ζΔ⁻¹μ_bruteforce = ∂ζ(Δ⁻¹μ, D)
psh_partial_ζΔ⁻¹μ_bruteforce = psh(partial_ζΔ⁻¹μ_bruteforce, D)

ref_ = (abs.(psh_partial_ζΔ⁻¹μ_bruteforce) .> 1e-10) + 2*(abs.(Δ⁻¹μ̂) .> 1e-10) + 3*(abs.(μ̂) .> 1e-10); 
ref_[m-1:m+8,m:m+2]
ref_[m-1:m+8,m] 
psh_partial_ζΔ⁻¹μ_bruteforce[m:2:end, m]

partial_ζΔ⁻¹μ_sparse = ∂ζΔ⁻¹_m_sparse(μ̂[m+1:2:end, m+1], m, Mr; aliasing = false)

norm(partial_ζΔ⁻¹μ_sparse - psh_partial_ζΔ⁻¹μ_bruteforce[m:2:end, m]) < 10^(-10)

# abs.(partial_ζΔ⁻¹μ_sparse)

#∂ζ̄

partial_ζ̄Δ⁻¹μ_bruteforce = ∂ζ̄(Δ⁻¹μ, D)
psh_partial_ζ̄Δ⁻¹μ_bruteforce = psh(partial_ζ̄Δ⁻¹μ_bruteforce, D)
ref_ = (abs.(psh_partial_ζ̄Δ⁻¹μ_bruteforce) .> 1e-10) + 2*(abs.(Δ⁻¹μ̂) .> 1e-10) + 3*(abs.(μ̂) .> 1e-10); 
ref_[m-1:m+10,m:m+2]

partial_ζ̄Δ⁻¹μ_sparse = ∂ζ̄Δ⁻¹_m_sparse(μ̂[m+1:2:end, m+1], m, Mr; aliasing = false)

norm(partial_ζ̄Δ⁻¹μ_sparse -psh_partial_ζ̄Δ⁻¹μ_bruteforce[m:2:end, m+2]) < 10^(-10)

