function trace(ûᵐ, lmax, m; aliasing = false)
    p = abs(m)
    trace_ûᵐ = zero(eltype(ûᵐ))

    for l in p:2:(lmax + 2 * aliasing)
        idx = (l - p) ÷ 2 + 1
        trace_ûᵐ += ûᵐ[idx] * ylm(l, m, 1.0)
    end

    return trace_ûᵐ
end

function traceĜ(f̂ᵐ, lmax, m; boundary_condition = :dirichlet)
    if m < 0
        return conj(traceĜ(conj.(f̂ᵐ), lmax, -m; boundary_condition = boundary_condition))
    end

    if boundary_condition == :dirichlet
        ûᵐ = Ĝᵐ(f̂ᵐ, lmax, m; aliasing = true)
        return trace(ûᵐ, lmax, m; aliasing = true)
    elseif boundary_condition == :neumann
        r∇ûᵐ = r_∂Ĝᵐ∂r(f̂ᵐ, lmax, m; aliasing = true)
        return trace(r∇ûᵐ, lmax, m; aliasing = true)
    end

    throw(ArgumentError("Unsupported boundary condition `$boundary_condition`. Expected `:dirichlet` or `:neumann`."))
end

function neumann_traceĜ(f̂ᵐ, lmax, m)
    return traceĜ(f̂ᵐ, lmax, m; boundary_condition = :neumann)
end

function traceĜ_column(Am, m)
    if m < 0
        return conj.(traceĜ_column(Am, -m))
    end

    l = m:2:size(Am, 1) * 2 + (m - 2)
    trace_column_ylm = zeros(ComplexF64, size(Am, 1))
    for i in 1:size(Am, 1)
        trace_column_ylm[i] = ylm(l[i], m, 1.0)
    end

    return Am' * trace_column_ylm
end

function helmholtz_matrix(lmax, m, α; boundary_condition = :dirichlet)
    A = inverse_laplacian_matrix_sparse(lmax, abs(m); rectangular = true)
    system_matrix = zeros(eltype(A), size(A) .+ (0, 1))
    system_matrix = AlmostBandedMatrix(brand(ComplexF64, size(A, 1), size(A, 1), 1, 1), rand(ComplexF64, 1, size(A, 1)))

    if boundary_condition != :dirichlet
        throw(ArgumentError("`helmholtz_matrix` currently supports only `:dirichlet` boundary conditions."))
    end

    trace_column = traceĜ_column(A, m)
    system_matrix[2:end, 2:end] .= α * A[1:end-1, :] + I
    system_matrix[2, 1] = α
    system_matrix[1, 1] = ylm(abs(m), m, 1.0)
    system_matrix[1, 2:end] .= trace_column

    return system_matrix
end
