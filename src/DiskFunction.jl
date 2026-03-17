"""
    DiskFunction{T<:Real}

A solution to Δu = f on the unit disk with homogeneous Dirichlet boundary
conditions, stored in PSH coefficient space.

Indexed as `df[l, m, i, j]`, where:
- `l`   : radial degree, `0 ≤ l ≤ lmax`
- `m`   : azimuthal frequency, `-Mθ ≤ m ≤ Mθ`
- `i,j` : derivative orders for ∂ζⁱ∂ζ̄ʲu, with `i,j ∈ {0,1,2}` and `i+j ≤ 2`

Only even-parity modes (`(l+m) % 2 == 0`, `abs(m) ≤ l`) are valid; all other
`(l, m)` combinations throw `BoundsError`.

Derivative coefficient arrays (beyond (0,0) and (1,1)) are lazily populated.

# Fields
- `lmax::Int`                                                  : maximum radial degree
- `Mspan::Vector{Int}`                                         : FFT-ordered frequency list
- `aliasing_factors::Vector{Complex{T}}`                       : aliased radial coefficient per frequency
- `_coeffs::Vector{Union{Nothing, TriangularCoeffArray{T}}}`   : lazy derivative slots (6 total)
- `is_real::Bool`                                              : whether the represented function is real-valued
"""
mutable struct DiskFunction{T<:Real} <: AbstractDiskFunction{T, 4}
    lmax             :: Int
    Mspan            :: Vector{Int}
    aliasing_factors :: Vector{Complex{T}}
    _coeffs          :: Vector{Union{Nothing, TriangularCoeffArray{T}}}
    is_real          :: Bool
end

# ─── AbstractArray interface ──────────────────────────────────────────────────

function Base.axes(df::DiskFunction)
    Mθ = length(df.Mspan) ÷ 2
    return (0:df.lmax, -Mθ:Mθ, 0:2, 0:2)
end

Base.size(df::DiskFunction) = (df.lmax + 1, length(df.Mspan), 3, 3)

@inline function Base.getindex(df::DiskFunction, l::Int, m::Int, i::Int, j::Int)
    @boundscheck checkbounds(df, l, m, i, j)
    (i + j > 2 || abs(m) > l || isodd(l + m)) && throw(BoundsError(df, (l, m, i, j)))
    k   = _ij_to_idx(i, j)
    df._coeffs[k] === nothing && error("derivative (i=$i, j=$j) has not been computed yet")
    col = _m_to_idx(m, length(df.Mspan))
    row = (l - abs(m)) ÷ 2 + 1
    return df._coeffs[k].data[col][row]
end

@inline function Base.setindex!(df::DiskFunction, v, l::Int, m::Int, i::Int, j::Int)
    @boundscheck checkbounds(df, l, m, i, j)
    (i + j > 2 || abs(m) > l || isodd(l + m)) && throw(BoundsError(df, (l, m, i, j)))
    k   = _ij_to_idx(i, j)
    df._coeffs[k] === nothing && error("derivative (i=$i, j=$j) has not been computed yet")
    col = _m_to_idx(m, length(df.Mspan))
    row = (l - abs(m)) ÷ 2 + 1
    df._coeffs[k].data[col][row] = v
end

# ─── Display ──────────────────────────────────────────────────────────────────

"""Map derivative order `(i, j)` to a Unicode label for display."""
function _slot_label(i::Int, j::Int)
    prefix = Dict((0,0) => "u", (1,0) => "∂ζu", (0,1) => "∂ζ̄u",
                  (2,0) => "∂ζ²u", (1,1) => "∂ζ∂ζ̄u", (0,2) => "∂ζ̄²u")
    return prefix[(i, j)]
end

function Base.show(io::IO, df::DiskFunction{T}) where T
    Mθ      = length(df.Mspan) ÷ 2
    defined = [_slot_label(i, j) for (k, (i, j)) in enumerate(_IDX_TO_IJ) if df._coeffs[k] !== nothing]
    print(io, "DiskFunction{$T}(lmax=$(df.lmax), Mθ=$Mθ, slots=[$(join(defined, ", "))])")
end

function Base.show(io::IO, ::MIME"text/plain", df::DiskFunction{T}) where T
    Mθ = length(df.Mspan) ÷ 2
    println(io, "DiskFunction{$T}")
    println(io, "  lmax    : $(df.lmax)")
    println(io, "  Mθ      : $Mθ")
    println(io, "  is_real : $(df.is_real)")
    print(  io, "  slots   :")
    for (k, (i, j)) in enumerate(_IDX_TO_IJ)
        status = df._coeffs[k] === nothing ? "·" : "✓"
        print(io, "  $status $(_slot_label(i, j))")
    end
    println(io)
end

# ─── In-place constructor kernel ─────────────────────────────────────────────

"""
    _fill_derivative_slot!(df, f̂_tri, i, j)

Fill derivative slot `(i, j)` of `df` from the density coefficients `f̂_tri`.

∂ζ decrements the azimuthal frequency by 1, ∂ζ̄ increments it by 1, so for an
output column at frequency `m_out` the input frequency is `m_in = m_out + i - j`.
Frequencies outside `[-Mθ, Mθ]` are skipped (left zero).

Supported `(i, j)`: `(1,0)`, `(0,1)`, `(2,0)`, `(0,2)`.

# Arguments
- `df`     : `DiskFunction` whose slot `(i,j)` is already allocated
- `f̂_tri`  : `TriangularCoeffArray` of even-parity density coefficients
- `i, j`   : derivative orders

# Returns
- `df` (mutated in place)
"""
function _fill_derivative_slot!(df::DiskFunction{T}, f̂_tri::TriangularCoeffArray, i::Int, j::Int) where T
    lmax = df.lmax
    Mθ   = length(df.Mspan) ÷ 2
    Δm   = i - j
    k    = _ij_to_idx(i, j)
    nM   = length(df.Mspan)
    for (idx_out, m_out) in enumerate(df.Mspan)
        m_in = m_out + Δm
        abs(m_in) > Mθ && continue
        idx_in = _m_to_idx(m_in, nM)
        f̂ᵐ    = f̂_tri.data[idx_in]
        dest   = df._coeffs[k].data[idx_out]
        if     i == 1 && j == 0
            ∂ζΔ⁻¹_m_sparse!(dest, f̂ᵐ, lmax, m_in; aliasing=false)
        elseif i == 0 && j == 1
            ∂ζ̄Δ⁻¹_m_sparse!(dest, f̂ᵐ, lmax, m_in; aliasing=false)
        elseif i == 2 && j == 0
            ∂ζ∂ζΔ⁻¹_m_sparse!(dest, f̂ᵐ, lmax, m_in)
        elseif i == 0 && j == 2
            ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(dest, f̂ᵐ, lmax, m_in)
        else
            error("_fill_derivative_slot!: unsupported (i=$i, j=$j); use DiskFunction! for (0,0) and (1,1)")
        end
    end
    return df
end

"""
    DiskFunction!(df, f̂_tri; derivatives=())

In-place kernel: fills the (0,0) and (1,1) derivative slots of `df` from the
sparse per-frequency coefficient array `f̂_tri`, plus any slots listed in
`derivatives`.

# Arguments
- `df`          : pre-allocated `DiskFunction` with slots 1, 5, and any
                  requested derivative slots already allocated
- `f̂_tri`       : `TriangularCoeffArray` of even-parity coefficients of the density f
- `derivatives` : iterable of `(i, j)` pairs for extra slots; valid values are
                  `(1,0)`, `(0,1)`, `(2,0)`, `(0,2)`

# Returns
- `df` (mutated in place)
"""
function DiskFunction!(df::DiskFunction{T}, f̂_tri::TriangularCoeffArray; derivatives=()) where T
    lmax = df.lmax
    for (i, m) in enumerate(df.Mspan)
        f̂ᵐ = f̂_tri.data[i]
        tmp = zeros(Complex{T}, length(f̂ᵐ) + 1)
        Inverse_laplacian_coef_m_sparse!(tmp, f̂ᵐ, lmax, m)
        df._coeffs[1].data[i]  .= tmp[1:end-1]
        df.aliasing_factors[i]  = tmp[end]
        df._coeffs[5].data[i]  .= f̂ᵐ ./ 4
    end
    for (i, j) in derivatives
        (i == 0 && j == 0) && continue
        (i == 1 && j == 1) && continue
        _fill_derivative_slot!(df, f̂_tri, i, j)
    end
    return df
end

# ─── Constructor ─────────────────────────────────────────────────────────────

"""
    DiskFunction(f, D)

Construct a `DiskFunction` by solving Δu = f with homogeneous Dirichlet boundary
conditions, using the coefficient-space inverse Laplacian.

Coefficients are stored as `TriangularCoeffArray`: only even-parity modes
(where l+m is even) are retained, one vector per azimuthal frequency.

# Arguments
- `f` : density on the disk grid (AbstractMatrix)
- `D` : disk discretization

# Returns
- `DiskFunction` with (0,0) and (1,1) derivative slots populated; all others `nothing`
"""
function DiskFunction(f::AbstractMatrix, D::disk)
    T     = Float64
    Mspan = vec(Array(D.Mspan))
    f̂_tri = NodalToTriangularArray(f, D)

    coeffs    = Vector{Union{Nothing, TriangularCoeffArray{T}}}(nothing, 6)
    coeffs[1] = TriangularCoeffArray{T}(D.Mr, Mspan)   # u = Δ⁻¹f
    coeffs[5] = TriangularCoeffArray{T}(D.Mr, Mspan)   # ∂ζ∂ζ̄u = f/4

    aliasing = zeros(Complex{T}, length(Mspan))
    is_real  = eltype(f) <: Real

    df = DiskFunction{T}(D.Mr, Mspan, aliasing, coeffs, is_real)
    DiskFunction!(df, f̂_tri)
    return df
end

"""
    DiskFunction(f̂_tri, D; derivatives=(), is_real=false)

Construct a `DiskFunction` from pre-computed sparse PSH coefficients.

Populates the (0,0) and (1,1) slots by default. Additional derivative slots
can be requested via `derivatives`.

# Arguments
- `f̂_tri`       : `TriangularCoeffArray` of even-parity coefficients of the density f
- `D`            : disk discretization
- `derivatives`  : iterable of `(i, j)` pairs for extra slots to populate;
                   valid values: `(1,0)`, `(0,1)`, `(2,0)`, `(0,2)`
- `is_real`      : whether the represented function is real-valued (default `false`)

# Returns
- `DiskFunction` with (0,0), (1,1), and any requested derivative slots populated
"""
function DiskFunction(f̂_tri::TriangularCoeffArray, D::disk; derivatives=(), is_real=false)
    T     = Float64
    Mspan = vec(Array(D.Mspan))

    coeffs    = Vector{Union{Nothing, TriangularCoeffArray{T}}}(nothing, 6)
    coeffs[1] = TriangularCoeffArray{T}(D.Mr, Mspan)
    coeffs[5] = TriangularCoeffArray{T}(D.Mr, Mspan)
    for (i, j) in derivatives
        k = _ij_to_idx(i, j)
        coeffs[k] === nothing && (coeffs[k] = TriangularCoeffArray{T}(D.Mr, Mspan))
    end

    aliasing = zeros(Complex{T}, length(Mspan))
    df = DiskFunction{T}(D.Mr, Mspan, aliasing, coeffs, is_real)
    DiskFunction!(df, f̂_tri; derivatives=derivatives)
    return df
end

# ─── Grid-space derivatives ───────────────────────────────────────────────────

"""
    ∂ζ(df::DiskFunction, D)

Complex derivative ∂ζ of the Poisson solution, returned as nodal values on the disk grid.

# Arguments
- `df` : Poisson solution in coefficient space
- `D`  : disk discretization

# Returns
- grid function ∂ζu on `D.ζ`
"""
function ∂ζ(df::DiskFunction, D::disk)
    lmax           = df.lmax
    Mθ             = D.Mθ
    results_sparse = copy(df._coeffs[1])
    for (i, m) in enumerate(df.Mspan)
        m_out = m - 1
        abs(m_out) > Mθ && continue
        f̂ᵐ = 4 .* df._coeffs[5].data[i]
        ∂ζΔ⁻¹_m_sparse!(column(results_sparse, m_out), f̂ᵐ, lmax, m; aliasing=false)
    end
    return ipsh(TriangularArrayToPSH(results_sparse, D), D)
end

"""
    ∂ζ̄(df::DiskFunction, D)

Conjugate derivative ∂ζ̄ of the Poisson solution, returned as nodal values on the disk grid.

# Arguments
- `df` : Poisson solution in coefficient space
- `D`  : disk discretization

# Returns
- grid function ∂ζ̄u on `D.ζ`
"""
function ∂ζ̄(df::DiskFunction, D::disk)
    lmax           = df.lmax
    Mθ             = D.Mθ
    results_sparse = copy(df._coeffs[1])
    for (i, m) in enumerate(df.Mspan)
        m_out = m + 1
        abs(m_out) > Mθ && continue
        f̂ᵐ = 4 .* df._coeffs[5].data[i]
        ∂ζ̄Δ⁻¹_m_sparse!(column(results_sparse, m_out), f̂ᵐ, lmax, m; aliasing=false)
    end
    return ipsh(TriangularArrayToPSH(results_sparse, D), D)
end

"""
    evaluate(df, i, j, D)
    evaluate(df, i, j, D, r)
    evaluate(df, D)
    evaluate(df, D, r)

Return nodal values of ∂ζⁱ∂ζ̄ʲu by converting the pre-computed coefficient slot
`(i, j)` back to physical space.

Without `r`, evaluates on the native radial grid `D.r`. With `r`, evaluates at
the specified radial points (passed through to `ipsh`).

The `(i, j) = (0, 0)` shorthands `evaluate(df, D)` / `evaluate(df, D, r)` return
the solution u itself.

# Arguments
- `df`   : `DiskFunction` with slot `(i, j)` already populated
- `i, j` : derivative orders (`i + j ≤ 2`)
- `D`    : disk discretization
- `r`    : radial evaluation points (optional)

# Returns
- Matrix of nodal values
"""
function evaluate(df::DiskFunction, i::Int, j::Int, D::disk)
    k = _ij_to_idx(i, j)
    if df._coeffs[k] === nothing
        error("slot ($i,$j) = $(_slot_label(i,j)) has not been computed; " *
              "reconstruct with derivatives=[($i,$j)]")
    end
    return ipsh(TriangularArrayToPSH(df._coeffs[k], D), D)
end

function evaluate(df::DiskFunction, i::Int, j::Int, D::disk, r)
    k = _ij_to_idx(i, j)
    if df._coeffs[k] === nothing
        error("slot ($i,$j) = $(_slot_label(i,j)) has not been computed; " *
              "reconstruct with derivatives=[($i,$j)]")
    end
    return ipsh(TriangularArrayToPSH(df._coeffs[k], D), D, r)
end

evaluate(df::DiskFunction, D::disk)    = evaluate(df, 0, 0, D)
evaluate(df::DiskFunction, D::disk, r) = evaluate(df, 0, 0, D, r)

# ─── Scalar arithmetic ────────────────────────────────────────────────────────

"""
    lmul!(α, df)

In-place: scale all populated coefficient slots of `df` by `α` (`df *= α`).
Also scales `aliasing_factors`.

# Returns
- `df` (mutated in place)
"""
function LinearAlgebra.lmul!(α::Number, df::DiskFunction)
    for k in eachindex(df._coeffs)
        df._coeffs[k] === nothing && continue
        lmul!(α, df._coeffs[k])
    end
    df.aliasing_factors .*= α
    return df
end

LinearAlgebra.rmul!(df::DiskFunction, α::Number) = lmul!(α, df)

function Base.:*(α::Number, df::DiskFunction)
    result         = deepcopy(df)
    result.is_real = df.is_real && α isa Real
    lmul!(α, result)
    return result
end

Base.:*(df::DiskFunction, α::Number) = α * df
Base.:/(df::DiskFunction, α::Number) = (1/α) * df

# ─── Addition with HarmonicFunction ─────────────────────────────────────────

"""
    _harmonic_coeff_for_slot(h, i, j)

Compute the harmonic coefficients of ∂ζⁱ∂ζ̄ʲh as an FFT-ordered vector
(same layout as `h.û`).  Applies `i` rounds of `∂ζ_HarmonicFunction!`
followed by `j` rounds of `∂ζ̄_HarmonicFunction!`.

Since Δh = 0, the (1,1) case returns a zero vector.
"""
function _harmonic_coeff_for_slot(h::HarmonicFunction, i::Int, j::Int)
    û   = copy(h.û)
    tmp = similar(û)
    for _ in 1:i
        ∂ζ_HarmonicFunction!(tmp, û, h.Mspan)
        û, tmp = tmp, û
    end
    for _ in 1:j
        ∂ζ̄_HarmonicFunction!(tmp, û, h.Mspan)
        û, tmp = tmp, û
    end
    return û
end

"""
    add!(df, h)

In-place: add harmonic function `h` into every populated derivative slot of `df`.

For each non-`nothing` slot with derivative order (i,j), the l=|m| coefficient
(row 1) of each frequency column is incremented by the corresponding (i,j)-th
derivative of `h`.  Since Δh = 0 the (1,1) slot is unaffected.

# Arguments
- `df` : `DiskFunction` to mutate
- `h`  : `HarmonicFunction` to add

# Returns
- `df` (mutated in place)
"""
function add!(df::DiskFunction, h::HarmonicFunction)
    for (k, (i, j)) in enumerate(_IDX_TO_IJ)
        df._coeffs[k] === nothing && continue
        dû = _harmonic_coeff_for_slot(h, i, j)
        for col in eachindex(df.Mspan)
            df._coeffs[k].data[col][1] += dû[col]
        end
    end
    return df
end

"""
    df + h
    h + df

Out-of-place addition of a `HarmonicFunction` into a `DiskFunction`.
Returns a deep copy of `df` with `h` added to every populated derivative slot.
"""
function Base.:(+)(df::DiskFunction, h::HarmonicFunction)
    result          = deepcopy(df)
    result.is_real  = df.is_real && h.is_real
    add!(result, h)
    return result
end

Base.:(+)(h::HarmonicFunction, df::DiskFunction) = df + h

"""
    sub!(df, h)

In-place: subtract harmonic function `h` from every populated derivative slot of `df`.
Equivalent to `add!(df, -h)`, but avoids allocating a negated copy.

# Arguments
- `df` : `DiskFunction` to mutate
- `h`  : `HarmonicFunction` to subtract

# Returns
- `df` (mutated in place)
"""
function sub!(df::DiskFunction, h::HarmonicFunction)
    for (k, (i, j)) in enumerate(_IDX_TO_IJ)
        df._coeffs[k] === nothing && continue
        dû = _harmonic_coeff_for_slot(h, i, j)
        for col in eachindex(df.Mspan)
            df._coeffs[k].data[col][1] -= dû[col]
        end
    end
    return df
end

"""
    df - h
    h - df

Out-of-place subtraction of a `HarmonicFunction` and a `DiskFunction`.
"""
function Base.:(-)(df::DiskFunction, h::HarmonicFunction)
    result         = deepcopy(df)
    result.is_real = df.is_real && h.is_real
    sub!(result, h)
    return result
end

function Base.:(-)(h::HarmonicFunction, df::DiskFunction)
    result         = deepcopy(df)
    result.is_real = df.is_real && h.is_real
    for (k, (i, j)) in enumerate(_IDX_TO_IJ)
        result._coeffs[k] === nothing && continue
        dû = _harmonic_coeff_for_slot(h, i, j)
        for col in eachindex(df.Mspan)
            result._coeffs[k].data[col][1] = dû[col] - result._coeffs[k].data[col][1]
        end
    end
    return result
end
