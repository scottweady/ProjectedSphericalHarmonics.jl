function DirichletTraceInverseLaplacian_m(f̂ᵐ, lmax, m)

    if m<0
        return conj.(DirichletTraceInverseLaplacian_m(conj.(f̂ᵐ), lmax, -m))
        
    end

    # Computes the m-th Fourier mode of the trace of the inverse Laplacian of a function with coefficients f̂ᵐ
    # on the ylm basis. This is used to compute the boundary value correction in the Poisson solver.
    ûᵐ = Inverse_laplacian_coef_m_sparse(f̂ᵐ, lmax, m; aliasing=true)
    traceûᵐ = 0
    for l in m:(lmax + 2)
        if (l + m) % 2 == 0
            traceûᵐ += ûᵐ[l - m + 1] * ylm(l, m, 1.0)
        end
    end

    return traceûᵐ
end

function DirichletTrace(ûᵐ, lmax, m; aliasing=false)

    p = abs(m)
    traceûᵐ = 0
    for l in p:2:(lmax + 2*aliasing)
        idx = (l - p) ÷ 2 + 1
        traceûᵐ += ûᵐ[idx] * ylm(l, m, 1.0)
    end

    return traceûᵐ

end

function NeumannTraceΔ⁻¹_sparse(f̂ᵐ, lmax, m)

    # Computes the m-th Fourier mode of the trace of the normal derivative of the inverse Laplacian of a function
    # with coefficients f̂ᵐ on the ylm basis. This is used to compute the boundary value correction in the Poisson solver.
    p = abs(m)
    r∇ûᵐ = r_dot_∇Δ⁻¹(f̂ᵐ, lmax, m; aliasing=true)
    traceûᵐ = 0
    for l in p:2:(lmax + 2)
        idx = (l - p) ÷ 2 + 1
        traceûᵐ += r∇ûᵐ[idx] * ylm(l, m, 1.0)
    end

    return traceûᵐ
end

function DirichletTraceColumnLaplacian_m(Am, m)
    if m < 0
        return conj.(DirichletTraceColumnLaplacian_m(Am, -m))
    end
    l = m:2:size(Am, 1) * 2 + (m - 2)
    trace_column_yml = zeros(ComplexF64, size(Am, 1))
    for i in 1:size(Am, 1)
        trace_column_yml[i] = ylm(l[i], m, 1.0)
    end

    return Am' * trace_column_yml
end

function ModifiedPoissonSystemMatrix(lmax, m, α)

    # Matrix system for equation (αI + Δ̂ᵐ) ûᵐ = f̂ᵐ
    # u = SN⁻¹μ + ylm(m,m, ζ)
    # ylm is harmonic

    A = GenerateSparseMatrix_InverseLaplacian(lmax, abs(m); rectangular=true)
    Full_system_matrix = zeros(eltype(A), size(A) .+ (0, 1))
    Full_system_matrix = AlmostBandedMatrix(brand(ComplexF64, size(A)[1], size(A)[1], 1, 1), rand(ComplexF64, 1, size(A)[1]))

    trace_column = DirichletTraceColumnLaplacian_m(A, m)
    Full_system_matrix[2:end, 2:end] .= α * A[1:end-1, :] + I
    Full_system_matrix[2, 1] = α
    Full_system_matrix[1, 1] = ylm(abs(m), m, 1.0)
    Full_system_matrix[1, 2:end] .= trace_column

    return Full_system_matrix
end
