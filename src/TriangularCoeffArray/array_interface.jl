# array_interface.jl
# Implements the AbstractArray interface (size, getindex, setindex!, IndexStyle),
# the parity/ordering accessors, and the convenience accessors mode_coefficients
# and ncolumns.

# ─── Type-level accessors ─────────────────────────────────────────────────────

"""
    parity(A) -> Symbol

Return the parity type parameter of `A`: `:even` (l+m even) or `:odd` (l+m odd).
"""
parity(::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O} = P

"""
    ordering(A) -> Symbol

Return the Mspan ordering type parameter of `A`:
`:fft`, `:natural`, `:positive`, or `:other`.
"""
ordering(::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O} = O

# ─── AbstractArray interface ──────────────────────────────────────────────────

Base.size(A::TriangularCoeffArray)   = (A._offsets[end],)
Base.length(A::TriangularCoeffArray) = A._offsets[end]

@inline function Base.getindex(A::TriangularCoeffArray, k::Int)
    @boundscheck checkbounds(A, k)
    col = searchsortedlast(A._offsets, k - 1)
    row = k - A._offsets[col]
    return A.data[col][row]
end

@inline function Base.setindex!(A::TriangularCoeffArray, v, k::Int)
    @boundscheck checkbounds(A, k)
    col = searchsortedlast(A._offsets, k - 1)
    row = k - A._offsets[col]
    A.data[col][row] = v
end

Base.IndexStyle(::Type{<:TriangularCoeffArray}) = IndexLinear()

# ─── mode_coefficients dispatch ───────────────────────────────────────────────

"""
    mode_coefficients(A, m)

Return the coefficient vector for azimuthal frequency `m` (a view into the
corresponding entry of `A.data`).

The lookup strategy is determined by `ordering(A)`:
- `:fft`      — O(1) formula for FFT-ordered Mspan
- `:natural`  — O(1) formula for naturally-ordered (centered) Mspan
- `:positive` — O(1) formula for positive-only Mspan `[0, 1, …, M]`
- `:other`    — O(N) `findfirst` fallback for arbitrary Mspan

Throws `BoundsError` if `m` is not a valid frequency for `A`.
"""
@inline mode_coefficients(A::TriangularCoeffArray, m::Int) =
    _mode_col(A, m, Val(ordering(A)))

# :fft — [0, 1, …, M(or M-1), -M, …, -1]
# Formula m ≥ 0 ? m+1 : N+m+1 is valid for both odd and even length(Mspan).
@inline _mode_col(A::TriangularCoeffArray, m::Int, ::Val{:fft}) =
    A.data[m ≥ 0 ? m + 1 : length(A.Mspan) + m + 1]

# :natural — first = -M (odd MT) or -(M_pos+1) (even MT), last = A.Mspan[end]
# General formula: col = m - first(A.Mspan) + 1 (correct for both MT parities).
@inline _mode_col(A::TriangularCoeffArray, m::Int, ::Val{:natural}) =
    A.data[m - A.Mspan[1] + 1]

# :positive — [0, 1, …, M], col = m+1
@inline _mode_col(A::TriangularCoeffArray, m::Int, ::Val{:positive}) =
    A.data[m + 1]

# :other — arbitrary ordering, O(N) findfirst fallback
@inline function _mode_col(A::TriangularCoeffArray, m::Int, ::Val{:other})
    col = findfirst(==(m), A.Mspan)
    col === nothing && throw(BoundsError(A, m))
    return A.data[col]
end

# ─── Other convenience accessors ──────────────────────────────────────────────

"""
    ncolumns(A)

Return the number of frequency columns in `A`.
"""
@inline ncolumns(A::TriangularCoeffArray) = length(A.Mspan)
