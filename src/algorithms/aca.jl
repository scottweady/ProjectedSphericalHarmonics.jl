
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

    # scale = maximum(abs, A)
    scale = 1.0

    for _ in 1:min(size(A)...)

        idx = argmax(abs.(E))
        ik, jk = idx[1], idx[2]
        pivot = E[ik, jk]

        abs(pivot) < tol * scale && break

        col = E[:, jk]
        row = E[ik, :]

        E .-= col * (row' / pivot)

        push!(Us, col / pivot)
        push!(Vs, conj.(row))

    end

    isempty(Us) && return zeros(eltype(A), size(A, 1), 0), zeros(eltype(A), size(A, 2), 0)
    return hcat(Us...), hcat(Vs...)

end

"""
    aca(K, ζ; tol=1e-6) -> U, V

Partially pivoted Adaptive Cross Approximation for a kernel matrix A[i,j] = K(ζ[i], ζ[j]).

# Arguments
- `K`    : kernel function `K(x, y)` returning a scalar
- `ζ`    : vector of points

# Keyword Arguments
- `tol`  : relative tolerance for low-rank approximation

# Returns
- `U`, `V` : low-rank factors such that A ≈ U * V'
"""
function aca(K::Function, ζ; tol=1e-10)

    ζ = vec(ζ)
    N = length(ζ)
    T = typeof(K(ζ[1], ζ[1]))

    getcol(j) = T[K(ζ[i], ζ[j]) for i in 1:N]
    getrow(i) = T[K(ζ[i], ζ[j]) for j in 1:N]

    Us = Vector{Vector{T}}()
    Vs = Vector{Vector{T}}()

    frob² = 0.0
    scale = -1.0
    used_rows = falses(N)
    i = argmax(abs(K(ζ[ii], ζ[ii])) for ii in 1:N)

    for _ in 1:min(N, N)

        used_rows[i] = true

        # Residual row: A[i,:] - Σ_k u_k[i] * v_k
        r = getrow(i)
        for k in eachindex(Us)
            r .-= Us[k][i] .* Vs[k]
        end

        j = argmax(abs.(r))
        pivot = r[j]
        scale < 0 && (scale = abs(pivot))
        abs(pivot) < tol * scale && break

        # Residual column: A[:,j] - Σ_k u_k * v_k[j]
        c = getcol(j)
        for k in eachindex(Us)
            c .-= Us[k] .* Vs[k][j]
        end

        u_new = c ./ pivot
        v_new = conj.(r)

        push!(Us, u_new)
        push!(Vs, v_new)

        # Stopping: new rank-1 update small relative to accumulated Frobenius norm
        δ = norm(u_new)^2 * norm(v_new)^2
        for k in 1:(length(Us)-1)
            frob² += 2 * real(dot(Us[k], u_new) * dot(Vs[k], v_new))
        end
        frob² += δ
        frob² > 0 && sqrt(δ) < tol * sqrt(frob²) && break

        # Next pivot row: largest entry of u_new among unused rows
        i = 0; best = -Inf
        for ii in 1:N
            used_rows[ii] && continue
            a = abs(u_new[ii])
            if a > best; best = a; i = ii; end
        end
        i == 0 && break

    end

    isempty(Us) && return zeros(T, N, 0), zeros(T, N, 0)
    return hcat(Us...), hcat(Vs...)

end

export aca