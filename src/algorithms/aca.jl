
"""
    aca(A; tol...) -> U, V

Adaptive Cross Approximation: low-rank decomposition A ≈ U * V' with |A - U*V'| < tol.

Algorithm: greedily selects pivot entries to build rank-1 updates of the residual.
Returns matrices U, V with k columns such that A ≈ U * V'.
"""
function aca(A::AbstractMatrix; tol=1e-6)

    E = copy(A)
    Us = Vector{Vector{eltype(A)}}()
    Vs = Vector{Vector{eltype(A)}}()

    for _ in 1:min(size(A)...)

        idx = argmax(abs.(E))
        ik, jk = idx[1], idx[2]
        pivot = E[ik, jk]

        abs(pivot) < tol && break

        col = E[:, jk]
        row = E[ik, :]

        E .-= col * (row' / pivot)

        push!(Us, col / pivot)
        push!(Vs, conj.(row))

    end

    isempty(Us) && return zeros(eltype(A), size(A, 1), 0), zeros(eltype(A), size(A, 2), 0)
    return hcat(Us...), hcat(Vs...)

end