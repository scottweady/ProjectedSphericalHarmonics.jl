# arithmetic.jl
# Out-of-place arithmetic (+, -, *, /), in-place Krylov primitives
# (axpy!, axpby!, rmul!, lmul!, copyto!, convert), and the broadcasting
# machinery (TriangularCoeffArrayStyle) for TriangularCoeffArray.

using LinearAlgebra

# ─── Out-of-place arithmetic ──────────────────────────────────────────────────

function Base.:+(A::TriangularCoeffArray{T,N,P,O}, B::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [A.data[i] .+ B.data[i] for i in eachindex(A.data)]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.:-(A::TriangularCoeffArray{T,N,P,O}, B::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [A.data[i] .- B.data[i] for i in eachindex(A.data)]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.:-(A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [.-v for v in A.data]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.:*(α::Number, A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [α .* v for v in A.data]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

Base.:*(A::TriangularCoeffArray, α::Number) = α * A
Base.:/(A::TriangularCoeffArray, α::Number) = (1/α) * A

# ─── In-place Krylov primitives ───────────────────────────────────────────────

"""
    axpy!(α, x, y)

In-place `y += α * x`. Operates column-by-column on the underlying data.
"""
function LinearAlgebra.axpy!(α, x::TriangularCoeffArray, y::TriangularCoeffArray)
    for i in eachindex(x.data)
        axpy!(α, x.data[i], y.data[i])
    end
    return y
end

"""
    axpby!(α, x, β, y)

In-place `y = α * x + β * y`. Operates column-by-column on the underlying data.
"""
function LinearAlgebra.axpby!(α, x::TriangularCoeffArray, β, y::TriangularCoeffArray)
    for i in eachindex(x.data)
        axpby!(α, x.data[i], β, y.data[i])
    end
    return y
end

LinearAlgebra.rmul!(A::TriangularCoeffArray, α::Number) = (foreach(v -> rmul!(v, α), A.data); A)
LinearAlgebra.lmul!(α::Number, A::TriangularCoeffArray) = (foreach(v -> lmul!(α, v), A.data); A)

Base.zero(A::TriangularCoeffArray) = fill!(similar(A), 0)

function Base.copyto!(dest::TriangularCoeffArray, src::TriangularCoeffArray)
    for i in eachindex(dest.data)
        copyto!(dest.data[i], src.data[i])
    end
    return dest
end

function Base.copyto!(dest::TriangularCoeffArray, src::AbstractVector)
    length(dest) == length(src) ||
        throw(DimensionMismatch("length $(length(dest)) ≠ $(length(src))"))
    for k in eachindex(src)
        dest[k] = src[k]
    end
    return dest
end

"""
    convert(::Type{TriangularCoeffArray{T, N, P, O}}, v)

Reconstruct a `TriangularCoeffArray{T, N, P, O}` from a flat vector `v` by
copying element-wise into a `similar` of the registered prototype for `(N, P, O)`.
Called by KrylovKit during GMRES restarts.
"""
function Base.convert(::Type{TriangularCoeffArray{T, N, P, O}}, v::AbstractVector) where {T, N, P, O}
    ref = get(_TCA_REGISTRY, (N, P, O), nothing)
    ref === nothing && error("No TriangularCoeffArray prototype registered for " *
                             "(N=$N, parity=$(QuoteNode(P)), ordering=$(QuoteNode(O))). " *
                             "Construct a TriangularCoeffArray with these parameters first.")
    result = similar(ref, Complex{T})
    copyto!(result, v)
    return result
end

# ─── Broadcasting ─────────────────────────────────────────────────────────────

struct TriangularCoeffArrayStyle <: Base.Broadcast.AbstractArrayStyle{1} end
TriangularCoeffArrayStyle(::Val{1}) = TriangularCoeffArrayStyle()
TriangularCoeffArrayStyle(::Val{N}) where N = Base.Broadcast.DefaultArrayStyle{N}()

Base.BroadcastStyle(::Type{<:TriangularCoeffArray}) = TriangularCoeffArrayStyle()
Base.BroadcastStyle(::TriangularCoeffArrayStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = TriangularCoeffArrayStyle()

"""Return the first `TriangularCoeffArray` found in a broadcast argument tree."""
_find_tca(bc::Base.Broadcast.Broadcasted) = _find_tca(bc.args)
_find_tca(args::Tuple)                    = _find_tca(_find_tca(args[1]), Base.tail(args))
_find_tca(A::TriangularCoeffArray, ::Any)   = A
_find_tca(A::TriangularCoeffArray, ::Tuple) = A
_find_tca(::Any, rest::Tuple)               = _find_tca(rest)
_find_tca(A::TriangularCoeffArray)        = A
_find_tca(::Any)                          = nothing
_find_tca(::Tuple{})                      = nothing

function Base.similar(bc::Base.Broadcast.Broadcasted{TriangularCoeffArrayStyle}, ::Type{ElType}) where ElType
    A  = _find_tca(bc)
    CT = ElType <: Complex ? ElType : Complex{real(ElType)}
    R  = real(CT)
    P  = parity(A)
    O  = ordering(A)
    data = [Vector{CT}(undef, length(v)) for v in A.data]
    return TriangularCoeffArray{R,P,O}(A.Mspan, data, copy(A._offsets))
end
