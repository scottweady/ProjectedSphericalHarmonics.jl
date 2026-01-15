
using IterativeSolvers, LinearAlgebra, LinearOperators

export Lσ⁻¹

"""
    Lσ⁻¹(β, D)

Solve the linear system Lσ(σ) = f for σ, where
    Lσ(σ) = σ + 2β * 𝒮(σ)

# Arguments
- `β` : metabolic efficieny (constant)
- `D` : discretization of the disk

# Returns
- solution vector σ
"""
function Lσ⁻¹(β::Float64, D)

    # Define the linear operator Lσ
    function Lσ!(b::AbstractVector, σ::AbstractVector)
        b .= σ + 2β * real.(𝒮(σ, D))
    end

    # Solve using GMRES
    N = length(D.ζ)
    f = 2β * ones(N)
    op = LinearOperator(Float64, N, N, false, false, Lσ!)
    σ, history = gmres(op, f; log=true, reltol=1e-10)

    # Compute residual norm
    f̂ = similar(f)
    Lσ!(f̂, σ)
    err = norm(f̂ - f) / norm(f)

    # Display convergence information
    println("GMRES converged in $(history.iters) iterations (residual norm: $err).")

    return σ

end

