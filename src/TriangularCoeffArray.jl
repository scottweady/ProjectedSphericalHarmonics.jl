using Printf
using LinearAlgebra

"""
    TriangularCoeffArray{T<:Real, N}

A flat 1D view of PSH sparse coefficient data stored as a vector of vectors.

Each column `data[i]` holds the even-parity radial coefficients for frequency
`Mspan[i]`, with lengths that vary by frequency (triangular structure).
The 1D interface exposes all columns concatenated in order, making this suitable
as a Krylov vector type.

The type parameter `N = length(Mspan)` is used by `Base.convert` to reconstruct
a `TriangularCoeffArray` from a flat `AbstractVector` during GMRES restarts.

# Fields
- `Mspan::Vector{Int}`               : FFT-ordered frequency list
- `data::Vector{Vector{Complex{T}}}` : per-frequency coefficient vectors
- `_offsets::Vector{Int}`            : cumulative column offsets for O(log n) indexing
"""
struct TriangularCoeffArray{T<:Real, N} <: AbstractArray{Complex{T}, 1}
    Mspan    :: Vector{Int}
    data     :: Vector{Vector{Complex{T}}}
    _offsets :: Vector{Int}
end

# Module-level registry: maps N (= length(Mspan)) → prototype for Base.convert
const _TCA_REGISTRY = Dict{Int, Any}()

"""
    TriangularCoeffArray{T}(Mspan, data, offsets)

Bridge constructor: computes `N = length(Mspan)`, instantiates
`TriangularCoeffArray{T, N}`, and registers it as the prototype for `Base.convert`.
All internal allocation paths go through this constructor.
"""
function TriangularCoeffArray{T}(Mspan::Vector{Int}, data::Vector{Vector{Complex{T}}}, offsets::Vector{Int}) where T<:Real
    N   = length(Mspan)
    obj = TriangularCoeffArray{T, N}(Mspan, data, offsets)
    _TCA_REGISTRY[N] = obj
    return obj
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    TriangularCoeffArray(Mspan, data)

Construct a `TriangularCoeffArray` from a frequency list and per-frequency data vectors.
Precomputes cumulative offsets for efficient linear indexing.

# Arguments
- `Mspan` : FFT-ordered frequency list
- `data`  : `Vector{Vector{Complex{T}}}`, one inner vector per frequency
"""
function TriangularCoeffArray(Mspan::Vector{Int}, data::Vector{Vector{Complex{T}}}) where T<:Real
    length(Mspan) == length(data) ||
        throw(ArgumentError("Mspan and data must have the same length"))
    offsets = _build_offsets(data)
    return TriangularCoeffArray{T}(Mspan, data, offsets)
end

"""
    TriangularCoeffArray{T}(lmax, Mspan)

Allocate a zero-filled `TriangularCoeffArray` for the given `lmax` and `Mspan`.
"""
function TriangularCoeffArray{T}(lmax::Int, Mspan::Vector{Int}) where T<:Real
    data    = [zeros(Complex{T}, _n_coeff_modes(lmax, m)) for m in Mspan]
    offsets = _build_offsets(data)
    return TriangularCoeffArray{T}(Mspan, data, offsets)
end

"""Return the number of even-parity radial modes for frequency `m` up to `lmax`."""
_n_coeff_modes(lmax::Int, m::Int) = length(abs(m):2:lmax)

"""Build cumulative offset vector: `_offsets[i]` = total elements before column `i`."""
function _build_offsets(data::Vector{<:AbstractVector})
    offsets = Vector{Int}(undef, length(data) + 1)
    offsets[1] = 0
    for i in eachindex(data)
        offsets[i+1] = offsets[i] + length(data[i])
    end
    return offsets
end

# ─── AbstractArray interface ──────────────────────────────────────────────────

Base.size(A::TriangularCoeffArray)       = (A._offsets[end],)
Base.length(A::TriangularCoeffArray)     = A._offsets[end]

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

# ─── Convenience accessors ────────────────────────────────────────────────────

"""
    column(A, m)

Return the coefficient vector for azimuthal frequency `m` (a view into the
corresponding entry of `A.data`).

Throws `BoundsError` if `m` is not a valid frequency for `A`.
"""
@inline function column(A::TriangularCoeffArray, m::Int)
    col = _m_to_idx(m, length(A.Mspan))
    @boundscheck (1 ≤ col ≤ length(A.data)) || throw(BoundsError(A, m))
    return A.data[col]
end

"""
    ncolumns(A)

Return the number of frequency columns in `A`.
"""
@inline ncolumns(A::TriangularCoeffArray) = length(A.Mspan)

# ─── Display ─────────────────────────────────────────────────────────────────

"""Infer the maximum radial degree from the stored column lengths and Mspan."""
function _infer_lmax(A::TriangularCoeffArray)
    isempty(A.data) && return 0
    return maximum(abs(m) + 2*(length(A.data[i]) - 1) for (i, m) in enumerate(A.Mspan))
end

"""Format a complex number in scientific notation with 4 decimal places."""
function _fmt_complex(z::Complex)
    r = @sprintf("%.3e", real(z))
    i = @sprintf("%.3e", imag(z))
    return "$(r)$(i)im"
    if iszero(imag(z))
        return r
    elseif imag(abs(z) ) > 0
        return "$(r)+$(i)im"
    else
        return "$(r)$(i)im"
    end
end

function Base.show(io::IO, A::TriangularCoeffArray{T}) where T
    lmax = _infer_lmax(A)
    Mθ   = length(A.Mspan) ÷ 2
    print(io, "TriangularCoeffArray{$T}(lmax=$lmax, Mθ=$Mθ, $(length(A)) coefficients)")
end

function Base.show(io::IO, ::MIME"text/plain", A::TriangularCoeffArray{T}) where T
    lmax  = _infer_lmax(A)
    nM    = length(A.Mspan)
    Mθ    = nM ÷ 2
    cw    = 16        # column width per (l, m) entry
    gutter = 6        # width of "l=XX │"

    term_rows, term_cols = displaysize(io)

    # ── Column range: symmetric around m=0, always shown ─────────────────────
    n_cols_avail = (term_cols - gutter) ÷ cw
    if 2Mθ + 1 ≤ n_cols_avail
        k_max         = Mθ
        show_col_dots = false
    else
        # Reserve one column width for the "⋯" indicator
        k_max         = max(0, (n_cols_avail - 2) ÷ 2)
        show_col_dots = (2k_max + 1) < n_cols_avail   # room for ⋯ after data cols
    end

    # ── Row range: from l=0, always shown ────────────────────────────────────
    n_rows_avail = term_rows - 4   # title + blank + header + separator
    if lmax + 1 ≤ n_rows_avail
        l_max_shown   = lmax
        show_row_dots = false
    else
        # Reserve one row for the "⋮" indicator
        l_max_shown   = max(0, n_rows_avail - 2)
        show_row_dots = true
    end

    println(io, "TriangularCoeffArray{$T} — lmax=$lmax, Mθ=$Mθ, $(length(A)) coefficients")
    println(io)

    # Header row
    print(io, " "^gutter)
    for m in -k_max:k_max
        print(io, lpad("m=$m", cw))
    end
    show_col_dots && print(io, lpad("⋯", cw))
    println(io)

    # Separator
    n_sep = (2k_max + 1) + (show_col_dots ? 1 : 0)
    println(io, " "^gutter * "─"^(cw * n_sep))

    # Data rows
    for l in 0:l_max_shown
        print(io, "l=$(lpad(string(l), 2)) │")
        for m in -k_max:k_max
            if abs(m) ≤ l && iseven(l + m)
                col = _m_to_idx(m, nM)
                row = (l - abs(m)) ÷ 2 + 1
                s   = _fmt_complex(A.data[col][row])
                print(io, lpad(s, cw))
            else
                print(io, lpad("·", cw))
            end
        end
        show_col_dots && print(io, lpad("⋯", cw))
        println(io)
    end

    # Row truncation indicator
    if show_row_dots
        print(io, " "^gutter)
        for _ in -k_max:k_max
            print(io, lpad("⋮", cw))
        end
        show_col_dots && print(io, lpad("⋱", cw))
        println(io)
    end
end

# ─── Nodal conversion ────────────────────────────────────────────────────────

"""
    NodalToTriangularArray(u, D)

Transform nodal values `u` on the disk grid into a `TriangularCoeffArray` by
computing the PSH transform and retaining only the even-parity coefficients
for each azimuthal frequency.

# Arguments
- `u` : nodal values on the disk grid (AbstractMatrix, size `(Nr, Nθ)`)
- `D` : disk discretization

# Returns
- `TriangularCoeffArray` whose column for frequency `m` holds the even-parity
  radial coefficients `û[l+m even, m]`
"""
function NodalToTriangularArray(u::AbstractMatrix, D::disk)
    T     = Float64
    û     = psh(u, D)
    Mspan = vec(Array(D.Mspan))
    data  = [Vector{Complex{T}}(û[D.even[:,i], i]) for i in eachindex(Mspan)]
    return TriangularCoeffArray(Mspan, data)
end

"""
    TriangularArrayToPSH(û_tri, D)

Convert a `TriangularCoeffArray` back into the full PSH coefficient matrix
expected by `ipsh`.

Each column `i` of the output matrix corresponds to frequency `D.Mspan[i]`,
and the even-parity rows (selected by `D.even[:,i]`) are filled from
`column(û_tri, D.Mspan[i])`; all other entries remain zero.

# Arguments
- `û_tri` : `TriangularCoeffArray` holding even-parity coefficients per frequency
- `D`     : disk discretization

# Returns
- PSH coefficient matrix of size `(D.Mr + 1, length(D.Mspan))`
"""
function TriangularArrayToPSH(û_tri::TriangularCoeffArray, D::disk)
    Mspan = vec(Array(D.Mspan))
    û_psh = zeros(ComplexF64, D.Mr + 1, length(Mspan))
    for (i, m) in enumerate(Mspan)
        û_psh[D.even[:,i], i] .= column(û_tri, m)
    end
    return û_psh
end

# ─── Arithmetic ───────────────────────────────────────────────────────────────

function Base.similar(A::TriangularCoeffArray{T}) where T
    data = [similar(v) for v in A.data]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
end

function Base.similar(A::TriangularCoeffArray{T}, ::Type{S}) where {T, S<:Complex}
    R = real(S)
    data = [Vector{S}(undef, length(v)) for v in A.data]
    return TriangularCoeffArray{R}(A.Mspan, data, copy(A._offsets))
end

function Base.similar(A::TriangularCoeffArray, ::Type{S}, dims::Dims) where {S}
    length(dims) == 1 && dims[1] == length(A) ||
        error("TriangularCoeffArray does not support arbitrary resizing")
    CT = S <: Complex ? S : Complex{real(S)}
    return similar(A, CT)
end

function Base.copy(A::TriangularCoeffArray{T}) where T
    data = [copy(v) for v in A.data]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
end

function Base.fill!(A::TriangularCoeffArray, v)
    for col in A.data
        fill!(col, v)
    end
    return A
end

# ─── Out-of-place arithmetic ──────────────────────────────────────────────────

function Base.:+(A::TriangularCoeffArray{T}, B::TriangularCoeffArray{T}) where T
    data = [A.data[i] .+ B.data[i] for i in eachindex(A.data)]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
end

function Base.:-(A::TriangularCoeffArray{T}, B::TriangularCoeffArray{T}) where T
    data = [A.data[i] .- B.data[i] for i in eachindex(A.data)]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
end

function Base.:-(A::TriangularCoeffArray{T}) where T
    data = [.-v for v in A.data]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
end

function Base.:*(α::Number, A::TriangularCoeffArray{T}) where T
    data = [α .* v for v in A.data]
    return TriangularCoeffArray{T}(A.Mspan, data, copy(A._offsets))
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
    convert(::Type{TriangularCoeffArray{T, N}}, v)

Reconstruct a `TriangularCoeffArray{T, N}` from a flat vector `v` by copying
element-wise into a `similar` of the registered prototype for `N`.
Called by KrylovKit during GMRES restarts.
"""
function Base.convert(::Type{TriangularCoeffArray{T, N}}, v::AbstractVector) where {T, N}
    ref = get(_TCA_REGISTRY, N, nothing)
    ref === nothing && error("No TriangularCoeffArray prototype registered for N=$N. " *
                             "Construct a TriangularCoeffArray with length(Mspan)=$N first.")
    result = similar(ref, Complex{T})
    copyto!(result, v)
    return result
end

# ─── AbstractArray performance hint ──────────────────────────────────────────

Base.IndexStyle(::Type{<:TriangularCoeffArray}) = IndexLinear()

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
    data = [Vector{CT}(undef, length(v)) for v in A.data]
    return TriangularCoeffArray{R}(A.Mspan, data, copy(A._offsets))
end
