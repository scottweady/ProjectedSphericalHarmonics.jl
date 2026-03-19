function recurrence_one_step(y₊₁, y, y₋₁, w, nl, nm, D)
    y₊₁ .= D.a[nl, nm] * w .* y .+ D.am1[nl, nm] * y₋₁
end

@inline function recurrence_two_steps!(y₊₂, y, y₋₂, w², nl, nm, D)
    @inbounds begin
        coef_l1   =  D.a[nl+1, nm] * D.a[nl, nm]
        coef_l0   =  D.am1[nl+1, nm] + (D.a[nl+1, nm] * D.am1[nl, nm]) / D.a[nl-1, nm]
        coel_l₋₂  = -(D.a[nl+1, nm] * D.am1[nl, nm]) * (D.am1[nl-1, nm] / D.a[nl-1, nm])
    end
    @. y₊₂ = (coef_l1 * w² + coef_l0) * y + coel_l₋₂ * y₋₂
    return nothing
end

m_cutoffs = [50, 100]
r_cutoff  = [0.47, 0.7]

function find_cutoff(m)
    m_to_cut = m_cutoffs .< abs(m)
    return any(m_to_cut), findlast(m_to_cut)
end

function psh!(û::TriangularCoeffArray{T,N,P,O}, u, D) where {T,N,P,O}
    O == :fft || error("ordering $O not yet implemented")

    uᵐ      = fft(u, 2)
    uw      = similar(uᵐ[:, 1])
    w²_full = 1 .- D.r .^ 2
    shift_p = P == :even ? 0 : 1

    for (nm, m) in enumerate(û.Mspan)

        ûᵐ = mode_coefficients(û, m)
        length(ûᵐ) == 0 && continue

        doIcut, wheredoIcut = find_cutoff(m)
        i0 = doIcut ? findfirst(vec(D.r .> r_cutoff[wheredoIcut])) : 1

        r      = @view D.r[i0:end]
        w²     = @view w²_full[i0:end]
        dw     = @view D.dw[i0:end]
        uᵐ_col = @view uᵐ[i0:end, nm]
        uw_col = @view uw[i0:end]

        uw_col .= uᵐ_col .* dw

        y₋₂ = ylm(abs(m) + shift_p, m, r)
        @inbounds ûᵐ[1] = dot(y₋₂, uw_col)
        length(ûᵐ) == 1 && continue

        y   = ylm(abs(m) + 2 + shift_p, m, r)
        y₊₂ = similar(y)
        @inbounds ûᵐ[2] = dot(y, uw_col)

        @inbounds for j in 2:(length(ûᵐ)-1)
            l  = abs(m) + 2 * j + shift_p
            nl = l - 1
            recurrence_two_steps!(y₊₂, y, y₋₂, w², nl, nm, D)
            ûᵐ[j+1] = dot(y₊₂, uw_col)
            y₋₂, y, y₊₂ = y, y₊₂, y₋₂
        end

    end

    return nothing
end

"""
    ipsh!(u, û, D)

In-place inverse PSH transform for triangular arrays.

Fills `u` (a nodal-space matrix) from a `TriangularCoeffArray` by accumulating
`ûᵐ[j] * Yˡʲₘ(r)` using the same two-step recurrence as `psh!`, then applying
an in-place inverse FFT. The parity (`:even` or `:odd`) is read from the type
parameter of `û`.

# Arguments
- `u`  : output matrix of size `(Nr, Nθ)`, overwritten with the nodal values
- `û`  : `TriangularCoeffArray` holding per-frequency coefficients
- `D`  : disk discretization

# Returns
- `u` (modified in-place)
"""
function ipsh!(u::AbstractMatrix, û::TriangularCoeffArray{T,N,P,O}, D) where {T,N,P,O}
    fill!(u, zero(eltype(u)))
    w²      = 1 .- D.r .^ 2
    shift_p = P == :even ? 0 : 1

    for (nm, m) in enumerate(û.Mspan)

        ûᵐ    = mode_coefficients(û, m)
        u_col = @view u[:, nm]
        length(ûᵐ) == 0 && continue

        y₋₂ = ylm(abs(m) + shift_p, m, D.r)
        @inbounds u_col .+= ûᵐ[1] .* y₋₂
        length(ûᵐ) == 1 && continue

        y   = ylm(abs(m) + 2 + shift_p, m, D.r)
        y₊₂ = similar(y)
        @inbounds u_col .+= ûᵐ[2] .* y

        @inbounds for j in 2:(length(ûᵐ)-1)
            l  = abs(m) + 2 * j + shift_p
            nl = l - 1
            recurrence_two_steps!(y₊₂, y, y₋₂, w², nl, nm, D)
            u_col .+= ûᵐ[j+1] .* y₊₂
            y₋₂, y, y₊₂ = y, y₊₂, y₋₂
        end

    end

    ifft!(u, 2)
    u .*= D.shp[2]
    return u
end

"""
    psh_triangular(u, D; parity=:even)

Out-of-place PSH transform for triangular arrays.

Allocates a zero `TriangularCoeffArray` from `D`, then delegates to `psh!`.

# Arguments
- `u`      : physical-space values on the quadrature grid
- `D`      : disk discretization (provides `Mr`, `Mspan`, quadrature weights, recurrence coefficients)
- `parity` : `:even` (default) or `:odd` — which `l+m` parity modes to store

# Returns
- a new `TriangularCoeffArray` containing the PSH coefficients of `u`
"""
function psh_triangular(u, D; parity::Symbol = :even)
    Mspan = vec(Array(D.Mspan))
    û = TriangularCoeffArray{Float64}(D.Mr, Mspan; parity = parity, ordering = :fft)
    psh!(û, u, D)
    return û
end
