function Inverse_laplacian(f̂)

    # Here we assume f̂ has a matrix structure, where the first column corresponds to m=0,
    # the second to m=1, etc. and the last to m=-1, etc.
    lmax = size(f̂, 1) - 1
    Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))
    f̂⁰ = @view(f̂[:, 1])
    Δ⁻¹f̂⁰ = Inverse_laplacian_coef_m(f̂⁰, lmax, 0; aliasing=false)
    Δ⁻¹f̂[:, 1] .= Δ⁻¹f̂⁰

    for m in 1:div(size(f̂, 2) - 1, 2)
        f̂ᵐ = @view(f̂[m + 1:end, m + 1])
        Δ⁻¹f̂ᵐ = Inverse_laplacian_coef_m(f̂ᵐ, lmax, m; aliasing=false)
        Δ⁻¹f̂[m + 1:end, m + 1] .= Δ⁻¹f̂ᵐ

        f̂⁻ᵐ = @view(f̂[m + 1:end, end - (m - 1)])
        Δ⁻¹f̂⁻ᵐ = Inverse_laplacian_coef_m(f̂⁻ᵐ, lmax, -m; aliasing=false)
        Δ⁻¹f̂[m + 1:end, end - (m - 1)] .= Δ⁻¹f̂⁻ᵐ
    end

    return Δ⁻¹f̂
end

function ∂ζ̄Δ⁻¹(f̂)

    # Here we assume f̂ has a matrix structure, where the first column corresponds to m=0,
    # the second to m=1, etc. and the last to m=-1, etc.
    lmax = size(f̂, 1) - 1
    ∂ζ̄Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))
    f̂⁰ = @view(f̂[1:2:end, 1])
    ∂ζ̄Δ⁻¹f̂⁰ = ζ∂ζΔ⁻¹_m_sparse(f̂⁰, lmax, 0; aliasing=false)
    ∂ζ̄Δ⁻¹f̂[2:2:end, 2] .= ∂ζ̄Δ⁻¹f̂⁰

    for m in 1:div(size(f̂, 2) - 1, 2)
        f̂ᵐ = @view(f̂[m + 1:2:end, m + 1])
        ∂ζ̄Δ⁻¹f̂ᵐ = ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=false)
        ∂ζ̄Δ⁻¹f̂[m + 1:2:end, m + 1] .= ∂ζ̄Δ⁻¹f̂ᵐ

        f̂⁻ᵐ = @view(f̂[m + 1:2:end, end - (m - 1)])
        ∂ζ̄Δ⁻¹f̂⁻ᵐ = ζ∂ζΔ⁻¹_m_sparse(f̂⁻ᵐ, lmax, -m; aliasing=false)
        ∂ζ̄Δ⁻¹f̂[m + 1:2:end, end - (m - 1)] .= ∂ζ̄Δ⁻¹f̂⁻ᵐ
    end

    return ∂ζ̄Δ⁻¹f̂
end

function ∂ζΔ⁻¹(f̂)

    # Here we assume f̂ has a matrix structure, where the first column corresponds to m=0,
    # the second to m=1, etc. and the last to m=-1, etc.
    lmax = size(f̂, 1) - 1
    ∂ζΔ⁻¹f̂ = zeros(ComplexF64, size(f̂))
    f̂⁰ = @view(f̂[1:2:end, 1])
    ∂ζΔ⁻¹f̂⁰ = ∂ζΔ⁻¹_m_sparse(f̂⁰, lmax, 0; aliasing=false)
    ∂ζΔ⁻¹f̂[2:2:end, 2] .= ∂ζΔ⁻¹f̂⁰

    for m in 1:div(size(f̂, 2) - 1, 2)
        f̂ᵐ = @view(f̂[m + 1:2:end, m + 1])
        ∂ζΔ⁻¹f̂ᵐ = ∂ζΔ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=false)
        ∂ζΔ⁻¹f̂[m + 1:2:end, m + 1] .= ∂ζΔ⁻¹f̂ᵐ

        f̂⁻ᵐ = @view(f̂[m + 1:2:end, end - (m - 1)])
        ∂ζΔ⁻¹f̂⁻ᵐ = ∂ζΔ⁻¹_m_sparse(f̂⁻ᵐ, lmax, -m; aliasing=false)
        ∂ζΔ⁻¹f̂[m + 1:2:end, end - (m - 1)] .= ∂ζΔ⁻¹f̂⁻ᵐ
    end

    return ∂ζΔ⁻¹f̂
end

function r_dot_∇Δ⁻¹(f̂)
    lmax = size(f̂, 1) - 2
    r_dot_∇Δ⁻¹f̂ = zeros(ComplexF64, size(f̂))

    f̂⁰ = @view f̂[1:2:end, 1]
    r_dot_∇Δ⁻¹f̂⁰ = ζ∂ζΔ⁻¹_m_sparse(f̂⁰, lmax, 0; aliasing=false) .+
                    ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂⁰, lmax, 0; aliasing=false)
    r_dot_∇Δ⁻¹f̂[1:2:end, 1] .= r_dot_∇Δ⁻¹f̂⁰

    for m in 1:min(div(size(f̂, 2) - 1, 2), lmax)
        f̂ᵐ = @view f̂[m + 1:2:end, m + 1]
        r_dot_∇Δ⁻¹f̂ᵐ = ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=false) .+
                        ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=false)
        r_dot_∇Δ⁻¹f̂[m + 1:2:end, m + 1] .= r_dot_∇Δ⁻¹f̂ᵐ

        f̂⁻ᵐ = @view f̂[m + 1:2:end, end - (m - 1)]
        r_dot_∇Δ⁻¹f̂⁻ᵐ = ζ∂ζΔ⁻¹_m_sparse(f̂⁻ᵐ, lmax, -m; aliasing=false) .+
                         ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂⁻ᵐ, lmax, -m; aliasing=false)
        r_dot_∇Δ⁻¹f̂[m + 1:2:end, end - (m - 1)] .= r_dot_∇Δ⁻¹f̂⁻ᵐ
    end

    return r_dot_∇Δ⁻¹f̂
end

"""
    Inverse_laplacian(u, D)

Inverse Laplacian on the disk with homogeneous Dirichlet boundary conditions,
computed in coefficient space.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- solution to Δv = u with v|∂D = 0
"""
function Inverse_laplacian(u, D)
    return ipsh(Inverse_laplacian(psh(u, D)), D)
end

"""
    ∂ζΔ⁻¹(u, D)

Complex derivative of the inverse Laplacian, computed in coefficient space.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- ∂ζ applied to the solution of Δv = u with v|∂D = 0
"""
function ∂ζΔ⁻¹(u, D)
    return ipsh(∂ζΔ⁻¹(psh(u, D)), D)
end

"""
    ∂ζ̄Δ⁻¹(u, D)

Complex conjugate derivative of the inverse Laplacian, computed in coefficient space.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- ∂ζ̄ applied to the solution of Δv = u with v|∂D = 0
"""
function ∂ζ̄Δ⁻¹(u, D)
    return ipsh(∂ζ̄Δ⁻¹(psh(u, D)), D)
end

"""
    r_dot_∇Δ⁻¹(u, D)

Radial gradient operator (r·∇) applied to the inverse Laplacian, computed in coefficient space.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- r·∇ applied to the solution of Δv = u with v|∂D = 0
"""
function r_dot_∇Δ⁻¹(u, D)
    return ipsh(r_dot_∇Δ⁻¹(psh(u, D)), D)
end

function NeumannTraceΔ⁻¹_sparse(f̂)
    lmax = size(f̂, 1) - 1
    trace_coeffs = zeros(ComplexF64, size(f̂, 2))

    f̂⁰ = @view f̂[1:2:end, 1]
    trace_coeffs[1] = NeumannTraceΔ⁻¹_sparse(f̂⁰, lmax, 0)

    for m in 1:div(size(f̂, 2) - 1, 2)
        f̂ᵐ = @view f̂[m + 1:2:end, m + 1]
        trace_coeffs[m + 1] = NeumannTraceΔ⁻¹_sparse(f̂ᵐ, lmax, m)

        f̂⁻ᵐ = @view f̂[m + 1:2:end, end - (m - 1)]
        trace_coeffs[end - (m - 1)] = NeumannTraceΔ⁻¹_sparse(f̂⁻ᵐ, lmax, -m)
    end

    return trace_coeffs
end

"""
    NeumannTraceΔ⁻¹_sparse(u, D)

Normal derivative on the boundary of the inverse Laplacian, computed in coefficient space.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- normal derivative on ∂D of the solution to Δv = u with v|∂D = 0
"""
function NeumannTraceΔ⁻¹_sparse(u, D)
    trace_coeffs = NeumannTraceΔ⁻¹_sparse(psh(u, D))
    return ifft(trace_coeffs) * length(trace_coeffs)
end
