using BandedMatrices
using FastAlmostBandedMatrices


function Inverse_laplacian_coef_m_positive(f̂ᵐ,lmax, m; aliasing = true)

    #f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
    #We recall that l+m has to be even, otherwise is 0
    #We ommit the negative m, since the coefficients are there cvonjugate


    Δ⁻¹f̂ᵐ = zeros(ComplexF64, length(f̂ᵐ)+2*aliasing)

    Δ⁻¹f̂ᵐ[1] += -1/( (2m+3)*(2m+1))*f̂ᵐ[1] 
    Δ⁻¹f̂ᵐ[3] += -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )*f̂ᵐ[1] 

    for l in m+2:lmax

        i = (l-m) + 1 

        if (l+m) % 2 == 0

            # Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
            Δ⁻¹f̂ᵐ[i] += -2/((2l-1)*(2l+3))*f̂ᵐ[i] 
            Δ⁻¹f̂ᵐ[i-2] += -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m)*f̂ᵐ[i]
            if l <lmax && l+1 < lmax
                Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
            elseif l == lmax && aliasing || (l == lmax - 1 && aliasing)
                Δ⁻¹f̂ᵐ[i+2] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i]
            end 
        
        end

    end

    return Δ⁻¹f̂ᵐ


end

function Inverse_laplacian_coef_m(f̂ᵐ,lmax, m; aliasing = true)

    #f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
    #We recall that l+m has to be even, otherwise is 0
    #We ommit the negative m, since the coefficients are there cvonjugate

    if m < 0
        return conj.(Inverse_laplacian_coef_m_positive(conj.(f̂ᵐ), lmax, -m; aliasing = aliasing))
    else
        return Inverse_laplacian_coef_m_positive(f̂ᵐ, lmax, m; aliasing = aliasing)
    end


end


function Inverse_laplacian_coef_m_sparse_positive(f̂ᵐ,lmax, m; aliasing = true)

    # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₂, , ..., f̂ᵐₗ_ₘₐₓ]
    # We recall that l+m has to be even, otherwise is 0
    # We ommit the negative m, since the coefficients are there cvonjugate


    Δ⁻¹f̂ᵐ = zeros(ComplexF64, length(f̂ᵐ)+1*aliasing)

    Δ⁻¹f̂ᵐ[1] += -1/( (2m+3)*(2m+1))*f̂ᵐ[1] 
    Δ⁻¹f̂ᵐ[2] += -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )*f̂ᵐ[1] 

    for l in m+2:2:lmax

        i = (l-m)÷2 + 1 
        # Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
        Δ⁻¹f̂ᵐ[i] += -2/((2l-1)*(2l+3))*f̂ᵐ[i] 
        Δ⁻¹f̂ᵐ[i-1] += -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m)*f̂ᵐ[i]
        if (l < lmax && l+1 < lmax) || aliasing
            Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i] 
        end 


    end

    return Δ⁻¹f̂ᵐ


end


function Inverse_laplacian_coef_m_sparse(f̂ᵐ,lmax, m; aliasing = true)

    # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₂, , ..., f̂ᵐₗ_ₘₐₓ]
    # We recall that l+m has to be even, otherwise is 0
    # We ommit the negative m, since the coefficients are there cvonjugate

    if m < 0
        return conj.(Inverse_laplacian_coef_m_sparse_positive(conj.(f̂ᵐ), lmax, -m; aliasing = aliasing))
    else
        return Inverse_laplacian_coef_m_sparse_positive(f̂ᵐ, lmax, m; aliasing = aliasing)
    end

end


function Inverse_laplacian(f̂)

    #Here we assume f̂ has a matrix structure, where the first column corresponds to m=0, the second to m=1, etc. and the last to m=-1, etc.
    lmax = size(f̂, 1) - 1
    Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))
    f̂⁰ = @view(f̂[:, 1])
    m = 0
    Δ⁻¹f̂⁰ = Inverse_laplacian_coef_m(f̂⁰,lmax, 0; aliasing = false)
    Δ⁻¹f̂[:, 1] .= Δ⁻¹f̂⁰

    for m in 1:div(size(f̂, 2)-1, 2)
        f̂ᵐ = @view(f̂[m+1:end, m+1])
        Δ⁻¹f̂ᵐ = Inverse_laplacian_coef_m(f̂ᵐ, lmax, m; aliasing = false)
        Δ⁻¹f̂[m+1:end, m+1] .= Δ⁻¹f̂ᵐ[1:end]
        f̂⁻ᵐ = @view(f̂[m+1:end, end-(m-1)])
        Δ⁻¹f̂⁻ᵐ = Inverse_laplacian_coef_m(f̂⁻ᵐ, lmax, -m; aliasing = false)
        Δ⁻¹f̂[m+1:end, end-(m-1)] .= Δ⁻¹f̂⁻ᵐ[1:end]
    end

    return Δ⁻¹f̂

end

function GenerateSparseMatrix_InverseLaplacian(lmax, m; rectangular = false)
    # This function generates the sparse matrix that represents the inverse Laplacian operator for a given m and lmax
    # The matrix will have dimensions (ceil((lmax-m)/2), ceil((lmax-m)/2)) since we are only considering the coefficients with l+m even

    n = ceil(Int, (lmax+1 - m) / 2)
    A = BandedMatrix(BandedMatrices.FillArrays.Zeros(n+rectangular,n), (1,1))
    
    A[1,1] = -1/( (2m+3)*(2m+1))
    A[2,1] = -Nlm(m, m, m+2, m)/( (2m+3)*(2m+1)*(2m+2) )

    for l in m+2:2:lmax

        i = (l-m)÷2 + 1 
        # Δ⁻¹f̂ᵐ[i+1] += -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m)*f̂ᵐ[i]
        A[i,i] = -2/((2l-1)*(2l+3))
        A[i-1, i] = -(l+m)/( (2l+1)*(2l-1) * (l-m-1))* Nlm(l, m, l-2, m) 
        
        if l < lmax && l+1 < lmax ||rectangular
            A[i+1, i] = -(l-m+1)/( (2l+1) * (2l+3)*(l+m+2))* Nlm(l, m, l+2, m) 
        end 


    end

    return A

end




function DirichletTraceInverseLaplacian_m(f̂ᵐ, lmax, m)

    #Computes the m-th Fourier mode of the trace of the inverse Laplacian of a function with coefficients f̂ᵐ on the ylm basis. This is used to compute the boundary value correction in the Poisson solver.
    ûᵐ = Inverse_laplacian_coef_m(f̂ᵐ,lmax, m; aliasing = true)

    traceûᵐ = 0
    for l in m:(lmax+2)
        if (l+m) % 2 == 0
            traceûᵐ .+= ûᵐ[l-m+1] * ylm(l, m, 1.0)
        end
    end

    return traceûᵐ

end


function DirichletTraceColumnLaplacian_m(Am, m)

    l = m:2:size(Am, 1)*2 .+ (m - 2)
    trace_column_yml = zeros(ComplexF64, size(Am, 1))
    for i in 1:size(Am, 1)
        trace_column_yml[i] = ylm(l[i], m, 1.0)
    end

    return Am'* trace_column_yml

end

function ModifiedPoissonSystemMatrix(lmax, m, α)

    #matrix system for equation (αI + Δ̂ᵐ ) ûᵐ  = f̂ᵐ
    # u = SN⁻¹μ + ylm(m,m, ζ) 
    # ylm is harmonic

    A = GenerateSparseMatrix_InverseLaplacian(lmax, m; rectangular = true)
    Full_system_matrix = zeros(eltype(A), size(A) .+ (0,1) )
    Full_system_matrix = AlmostBandedMatrix(brand(Float64, size(A)[1], size(A)[1], 1, 1), rand(Float64, 1, size(A)[1]))
    # n = size(A, 1)
    # AlmostBandedMatrix(brand(eltype(A), n, n, m + 1, m), rand(Float64, m, n))

    trace_column = DirichletTraceColumnLaplacian_m(A, m)
    Full_system_matrix[2:end, 2:end] .= α*A[1:end-1,:] + I
    Full_system_matrix[2, 1] = α
    Full_system_matrix[1,1] = ylm(m, m, 1.0)
    Full_system_matrix[1, 2:end] .= trace_column
    

    return Full_system_matrix


end


#TODO: Fix Aliasing, as an operator, it should only depend on the evenness of lmax+m
#TODO: Implement for m < 0, it should be the conjugate of the m > 0 case, but we need to check the indexing carefully

function ∂ζΔ⁻¹_m_sparse( μ̂ₘ , m, lmax; aliasing = true )

    #Sends to the (m-1) vector
    #Aliasing is wrong, it only depends on the evenness of lmax and m
    if m < 0
        error("Not yet implemented for m < 0")
        return conj.(∂ζ̄Δ⁻¹_m(conj.(μ̂ₘ), D, -m))
    end

    ∂_ζΔ⁻¹μ̂ᵐ = zeros(ComplexF64, length(μ̂ₘ) + ( (lmax + m)%2 ) + aliasing )


    if m>0
        ∂_ζΔ⁻¹μ̂ᵐ[2] += 1/2/(2m+1)*Nlm(m, m, m+1, m-1)*μ̂ₘ[1]
    end

    for l in m+2:2:lmax

        i = (l-m)÷2 + 1
  
        ∂_ζΔ⁻¹μ̂ᵐ[i] += (l+m)*1/2/(2l+1)*Nlm(l, m, l-1, m-1)*μ̂ₘ[i]
        if aliasing || (l < lmax && l+1 < lmax)
            ∂_ζΔ⁻¹μ̂ᵐ[i+1] += (l-m+1)*1/2/(2l+1)*Nlm(l, m, l+1, m-1)*μ̂ₘ[i]
        end
        
    end

    return ∂_ζΔ⁻¹μ̂ᵐ



end


function ∂ζ̄Δ⁻¹_m_sparse( μ̂ₘ , m, lmax; aliasing = true )

    #Sends to the (m+1) vector
    #Aliasing is wrong, it only depends on the evenness of lmax and m
    if m < 0
        return conj.(∂ζΔ⁻¹_m(conj.(μ̂ₘ), D, -m; aliasing = aliasing))
    end

    ∂_ζ̄Δ⁻¹μ̂ᵐ = zeros(ComplexF64, length(μ̂ₘ) + ( (lmax + m)%2 ) + aliasing)

    if m>0
        ∂_ζ̄Δ⁻¹μ̂ᵐ[2] += -1/4/(2m+1)/(m+1)*Nlm(m, m, m+1, m+1)*μ̂ₘ[1]
    end


    for l in m+2:2:lmax

        i = (l-m)÷2 + 1
  
        ∂_ζ̄Δ⁻¹μ̂ᵐ[i] += -1/(l-m-1)*1/2/(2l+1)*Nlm(l, m, l-1, m+1)*μ̂ₘ[i]
        
        if aliasing || (l < lmax && l+1 < lmax)
            ∂_ζ̄Δ⁻¹μ̂ᵐ[i+1] += -1/2*1/(2l+1)*1/(l+m+2)*Nlm(l, m, l+1, m+1)*μ̂ₘ[i]
        end
        
    end

    return ∂_ζ̄Δ⁻¹μ̂ᵐ

end


# function r∇Δ⁻¹_m_sparse( μ̂ₘ , m, lmax; aliasing = true )
    
    
# end