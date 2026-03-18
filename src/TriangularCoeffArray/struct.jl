# struct.jl
# Defines the TriangularCoeffArray struct, the module-level prototype registry,
# the bridge constructor, user-facing constructors, internal allocation helpers,
# and instance-level allocation (similar, copy, fill!).

"""
    TriangularCoeffArray{T<:Real, N, P, O}

A flat 1D view of PSH sparse coefficient data stored as a vector of vectors.

Each column `data[i]` holds the radial coefficients for frequency `Mspan[i]`,
retaining only modes of parity `P` (`:even` for `l+m` even, `:odd` for `l+m`
odd), with lengths that vary by frequency (triangular structure).
The 1D interface exposes all columns concatenated in order, making this suitable
as a Krylov vector type.

Type parameters:
- `T` : underlying real type (elements are `Complex{T}`)
- `N` : `length(Mspan)`, used by `Base.convert` during GMRES restarts
- `P` : parity symbol, `:even` or `:odd`
- `O` : ordering symbol, `:fft`, `:natural`, `:positive`, or `:other`

# Fields
- `Mspan::Vector{Int}`               : frequency list (layout determined by `O`)
- `data::Vector{Vector{Complex{T}}}` : per-frequency coefficient vectors
- `_offsets::Vector{Int}`            : cumulative column offsets for O(log n) indexing
"""
struct TriangularCoeffArray{T<:Real, N, P, O} <: AbstractArray{Complex{T}, 1}
    Mspan    :: Vector{Int}
    data     :: Vector{Vector{Complex{T}}}
    _offsets :: Vector{Int}
end

# Module-level registry: maps (N, P, O) → prototype for Base.convert
const _TCA_REGISTRY = Dict{Tuple{Int,Symbol,Symbol}, Any}()

"""
    TriangularCoeffArray{T, P, O}(Mspan, data, offsets)

Bridge constructor: computes `N = length(Mspan)`, instantiates
`TriangularCoeffArray{T, N, P, O}`, and registers it as the prototype for
`Base.convert`. All internal allocation paths go through this constructor.
"""
function TriangularCoeffArray{T, P, O}(Mspan::Vector{Int},
                                       data::Vector{Vector{Complex{T}}},
                                       offsets::Vector{Int}) where {T<:Real, P, O}
    N   = length(Mspan)
    obj = TriangularCoeffArray{T, N, P, O}(Mspan, data, offsets)
    _TCA_REGISTRY[(N, P, O)] = obj
    return obj
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    TriangularCoeffArray(Mspan, data; parity=:even, ordering=:fft)

Construct a `TriangularCoeffArray` from a frequency list and per-frequency data
vectors. Precomputes cumulative offsets for efficient linear indexing.

# Arguments
- `Mspan`   : frequency list (layout must match `ordering`)
- `data`    : `Vector{Vector{Complex{T}}}`, one inner vector per frequency
- `parity`  : `:even` (default) or `:odd` — which `l+m` parity modes are stored
- `ordering`: `:fft` (default), `:natural`, `:positive`, or `:other`
"""
function TriangularCoeffArray(Mspan::Vector{Int},
                              data::Vector{Vector{Complex{T}}};
                              parity::Symbol  = :even,
                              ordering::Symbol = :fft) where T<:Real
    length(Mspan) == length(data) ||
        throw(ArgumentError("Mspan and data must have the same length"))
    offsets = _build_offsets(data)
    return TriangularCoeffArray{T, parity, ordering}(Mspan, data, offsets)
end

"""
    TriangularCoeffArray{T}(lmax, Mspan; parity=:even, ordering=:fft)

Allocate a zero-filled `TriangularCoeffArray` for the given `lmax` and `Mspan`.
"""
function TriangularCoeffArray{T}(lmax::Int, Mspan::Vector{Int};
                                 parity::Symbol   = :even,
                                 ordering::Symbol = :fft) where T<:Real
    data    = [zeros(Complex{T}, _n_coeff_modes(lmax, m, Val(parity))) for m in Mspan]
    offsets = _build_offsets(data)
    return TriangularCoeffArray{T, parity, ordering}(Mspan, data, offsets)
end

"""Return the number of parity-matching radial modes for frequency `m` up to `lmax`."""
_n_coeff_modes(lmax::Int, m::Int, ::Val{:even}) = length(abs(m):2:lmax)
_n_coeff_modes(lmax::Int, m::Int, ::Val{:odd})  = length((abs(m)+1):2:lmax)

"""Build cumulative offset vector: `_offsets[i]` = total elements before column `i`."""
function _build_offsets(data::Vector{<:AbstractVector})
    offsets = Vector{Int}(undef, length(data) + 1)
    offsets[1] = 0
    for i in eachindex(data)
        offsets[i+1] = offsets[i] + length(data[i])
    end
    return offsets
end

# ─── Instance-level allocation ────────────────────────────────────────────────

function Base.similar(A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [similar(v) for v in A.data]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.similar(A::TriangularCoeffArray{T,N,P,O}, ::Type{S}) where {T,N,P,O,S<:Complex}
    R = real(S)
    data = [Vector{S}(undef, length(v)) for v in A.data]
    return TriangularCoeffArray{R,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.similar(A::TriangularCoeffArray{T,N,P,O}, ::Type{S}, dims::Dims) where {T,N,P,O,S}
    length(dims) == 1 && dims[1] == length(A) ||
        error("TriangularCoeffArray does not support arbitrary resizing")
    CT = S <: Complex ? S : Complex{real(S)}
    return similar(A, CT)
end

function Base.copy(A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    data = [copy(v) for v in A.data]
    return TriangularCoeffArray{T,P,O}(A.Mspan, data, copy(A._offsets))
end

function Base.fill!(A::TriangularCoeffArray, v)
    for col in A.data
        fill!(col, v)
    end
    return A
end
