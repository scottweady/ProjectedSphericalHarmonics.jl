"""
    EllipticSolution{T<:Real}

Coefficient-space representation of the solution to an elliptic problem on the
disk, stored as the decomposition

    u = Δ⁻¹μ + h,

where `h` is harmonic (`Δh = 0`) and `μ` is the Poisson density.

This is the intermediate representation between the raw `TriangularCoeffArray`
level and `DiskFunction`. It preserves the decomposition, which enables exact
derivative computation and reuse of the density across time steps.

The ordering of `harmonic` and `density` is assumed to be the same:
`harmonic[i]` corresponds to frequency `density.Mspan[i]`.

`sol.Mspan` is a convenience accessor that delegates to `sol.density.Mspan`.

# Fields
- `harmonic :: Vector{Complex{T}}` : harmonic coefficients; `harmonic[i]` is the
                                     coefficient at frequency `density.Mspan[i]`
- `density  :: TriangularCoeffArray{T}` : Poisson density μ̂ in PSH coefficient space
"""
struct EllipticSolution{T<:Real}
    harmonic :: Vector{Complex{T}}
    density  :: TriangularCoeffArray{T}
end

function Base.getproperty(sol::EllipticSolution, name::Symbol)
    name === :Mspan && return getfield(sol, :density).Mspan
    return getfield(sol, name)
end

# ─── Display ─────────────────────────────────────────────────────────────────

function Base.show(io::IO, sol::EllipticSolution)
    T    = real(eltype(sol.harmonic))
    Mθ   = length(sol.Mspan) ÷ 2
    lmax = sol.density.lmax
    print(io, "EllipticSolution{", T, "}(lmax=", lmax, ", Mθ=", Mθ, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sol::EllipticSolution)
    T    = real(eltype(sol.harmonic))
    Mθ   = length(sol.Mspan) ÷ 2
    lmax = sol.density.lmax
    println(io, "EllipticSolution{", T, "}")
    println(io, "  lmax : ", lmax)
    println(io, "  Mθ   : ", Mθ)
end

# ─── Evaluation ──────────────────────────────────────────────────────────────

"""
    _add_harmonic!(û_tri, harmonic)

Add the harmonic coefficients into the first row (l = |m| mode) of each
frequency column of `û_tri`. The frequency ordering is read from `û_tri.Mspan`.
"""
function _add_harmonic!(û_tri::TriangularCoeffArray, harmonic::AbstractVector)
    lmax  = û_tri.lmax
    for (i, m) in enumerate(û_tri.Mspan)
        size_current_m(lmax, m) > 0 || continue
        mode_coefficients(û_tri, m)[1] += harmonic[i]
    end
    return û_tri
end

"""
    evaluate!(u, sol, D; derivative=(0,0))

In-place: fill the nodal matrix `u` with the values of `∂ζⁱ∂ζ̄ʲ(Δ⁻¹μ + h)` on `D.r`,
where `(i, j) = derivative`.

Supported derivative orders: `(0,0)`, `(1,0)`, `(0,1)`, `(2,0)`.

# Arguments
- `u`          : output matrix of size `(Nr, Nθ)`, overwritten in place
- `sol`        : `EllipticSolution` in coefficient space
- `D`          : disk discretization
- `derivative` : tuple `(i, j)` selecting ∂ζⁱ∂ζ̄ʲ (default `(0,0)`)

# Returns
- `u` (modified in-place)
"""
function evaluate!(u::AbstractMatrix, sol::EllipticSolution, D::disk; derivative = (0,0))
    Mspan = sol.Mspan
    dû    = zeros(ComplexF64, length(sol.harmonic))

    if derivative == (0,0)
        û_tri = Ĝ(sol.density)
        _add_harmonic!(û_tri, sol.harmonic)

    elseif derivative == (1,0)
        û_tri = ∂Ĝ∂ζ(sol.density)
        ∂ζ_harmonic!(dû, sol.harmonic, Mspan)
        _add_harmonic!(û_tri, dû)

    elseif derivative == (0,1)
        û_tri = ∂Ĝ∂ζ̄(sol.density)
        ∂ζ̄_harmonic!(dû, sol.harmonic, Mspan)
        _add_harmonic!(û_tri, dû)

    elseif derivative == (2,0)
        û_tri = ∂²Ĝ∂ζ²(sol.density)
        tmp = similar(dû)
        ∂ζ_harmonic!(tmp, sol.harmonic, Mspan)
        ∂ζ_harmonic!(dû, tmp, Mspan)
        _add_harmonic!(û_tri, dû)

    else
        throw(ArgumentError("unsupported derivative $derivative; supported: (0,0), (1,0), (0,1), (2,0)"))
    end

    ipsh!(u, û_tri, D)
    return u
end

"""
    evaluate(sol, D)

Reconstruct nodal values of `u = Δ⁻¹μ + h` on the native grid `D.r`.

Applies `Ĝ` to the stored density, then adds the harmonic coefficient to the
first row (lowest-degree mode `l = |m|`) of each frequency column.

# Arguments
- `sol` : `EllipticSolution` in coefficient space
- `D`   : disk discretization

# Returns
- Matrix of nodal values of `u` on `D.ζ`
"""
function evaluate(sol::EllipticSolution, D::disk)
    T = real(eltype(sol.harmonic))
    u = zeros(Complex{T}, length(D.r), length(sol.Mspan))
    evaluate!(u, sol, D)
    return u
end

"""
    evaluate(sol, D, r)

Reconstruct nodal values of `u = Δ⁻¹μ + h` at custom radial points `r`.

# Arguments
- `sol` : `EllipticSolution` in coefficient space
- `D`   : disk discretization
- `r`   : radial evaluation points

# Returns
- Matrix of nodal values of `u` evaluated at radii `r`
"""
function evaluate(sol::EllipticSolution, D::disk, r)
    û_tri = Ĝ(sol.density)
    _add_harmonic!(û_tri, sol.harmonic)
    return ipsh(TriangularArrayToPSH(û_tri, D), D, r)
end

# ─── Arithmetic ──────────────────────────────────────────────────────────────

"""
    lmul!(α, sol)

In-place: scale both the harmonic coefficients and the density of `sol` by `α`.

# Returns
- `sol` (mutated in place)
"""
function LinearAlgebra.lmul!(α::Number, sol::EllipticSolution)
    sol.harmonic .*= α
    lmul!(α, sol.density)
    return sol
end

LinearAlgebra.rmul!(sol::EllipticSolution, α::Number) = lmul!(α, sol)

function Base.:*(α::Number, sol::EllipticSolution)
    result = deepcopy(sol)
    lmul!(α, result)
    return result
end

Base.:*(sol::EllipticSolution, α::Number) = α * sol
Base.:/(sol::EllipticSolution, α::Number) = (1/α) * sol

function Base.:(+)(s1::EllipticSolution, s2::EllipticSolution)
    return EllipticSolution(s1.harmonic .+ s2.harmonic,
                            s1.density  +  s2.density)
end

function Base.:(-)(s1::EllipticSolution, s2::EllipticSolution)
    return EllipticSolution(s1.harmonic .- s2.harmonic,
                            s1.density  -  s2.density)
end

Base.:(-)(sol::EllipticSolution) = (-1) * sol
