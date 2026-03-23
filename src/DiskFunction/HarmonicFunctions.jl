"""
    HarmonicFunction{T<:Real}

A harmonic function on the unit disk (Δu = 0) with prescribed Dirichlet boundary
values, stored as a 1D coefficient vector in the PSH boundary normalization.

Indexed by azimuthal frequency `m` in the range `-Mθ:Mθ`.

# Fields
- `û::Vector{Complex{T}}` : per-frequency harmonic coefficients (length `2Mθ+1`)
- `Mspan::Vector{Int}`    : FFT-ordered frequency list `[0, 1, …, Mθ, -Mθ, …, -1]`
- `is_real::Bool`         : whether the represented function is real-valued
"""
struct HarmonicFunction{T<:Real} <: AbstractDiskFunction{T, 1}
    û       :: Vector{Complex{T}}
    Mspan   :: Vector{Int}
    is_real :: Bool
end

# ─── AbstractArray interface ──────────────────────────────────────────────────

Base.axes(h::HarmonicFunction) = let Mθ = length(h.û) ÷ 2; (-Mθ:Mθ,) end
Base.size(h::HarmonicFunction) = (length(h.û),)

@inline function Base.getindex(h::HarmonicFunction, m::Int)
    @boundscheck checkbounds(h, m)
    return h.û[_m_to_idx(m, length(h.û))]
end

@inline function Base.setindex!(h::HarmonicFunction, v, m::Int)
    @boundscheck checkbounds(h, m)
    h.û[_m_to_idx(m, length(h.û))] = v
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    HarmonicFunction(data, D; from_coefficients=false)

Construct a `HarmonicFunction` from a disk discretization `D` and either
boundary values or pre-computed harmonic coefficients.

# Arguments
- `data`              : input vector; interpreted according to `from_coefficients`
- `D`                 : disk discretization
- `from_coefficients` : if `false` (default), `data` are Dirichlet boundary values
                        on ∂D and the harmonic coefficients are solved for;
                        if `true`, `data` are the harmonic coefficients `û` directly
                        (length `2Mθ+1`, FFT-ordered)

# Returns
- `HarmonicFunction` with the harmonic extension matching `data` on ∂D
"""
function HarmonicFunction(data::AbstractVector, D::disk; from_coefficients=false)
    Mspan   = vec(Array(D.Mspan))
    û       = from_coefficients ? Vector{ComplexF64}(data) : SolveHarmonicFunction_coefficient(data, D)
    is_real = eltype(data) <: Real
    return HarmonicFunction{Float64}(û, Mspan, is_real)
end

function SolveHarmonicFunction_coefficient(g, D)
    ĝ = fft(g) / length(g)
    û_harmonic = zeros(ComplexF64, length(ĝ))
    return harmonic_coeff_from_dirichlet!(û_harmonic, ĝ, vec(Array(D.Mspan)))
end

# ─── Evaluation ───────────────────────────────────────────────────────────────

function EvaluateHarmonicFunction(û_harmonic, D)
    û_psh = zeros(ComplexF64, size(D.ζ))
    for i in eachindex(Array(D.Mspan))
        m = Array(D.Mspan)[i]
        if m < 0
            û_psh[abs(m)+1, end - (abs(m) - 1)] = û_harmonic[i]
        else
            û_psh[abs(m)+1, i] = û_harmonic[i]
        end
    end
    return ipsh(û_psh, D)
end

# ─── Arithmetic ───────────────────────────────────────────────────────────────

"""
    add!(h1, h2)

In-place: add `h2` into `h1` (`h1 += h2`). Mutates `h1.û`.

# Returns
- `h1` (mutated in place)
"""
function add!(h1::HarmonicFunction, h2::HarmonicFunction)
    h1.û .+= h2.û
    return h1
end

"""
    sub!(h1, h2)

In-place: subtract `h2` from `h1` (`h1 -= h2`). Mutates `h1.û`.

# Returns
- `h1` (mutated in place)
"""
function sub!(h1::HarmonicFunction, h2::HarmonicFunction)
    h1.û .-= h2.û
    return h1
end

function Base.:+(h1::HarmonicFunction{T}, h2::HarmonicFunction{T}) where T
    result = HarmonicFunction{T}(copy(h1.û), h1.Mspan, h1.is_real && h2.is_real)
    add!(result, h2)
    return result
end

function Base.:-(h1::HarmonicFunction{T}, h2::HarmonicFunction{T}) where T
    result = HarmonicFunction{T}(copy(h1.û), h1.Mspan, h1.is_real && h2.is_real)
    sub!(result, h2)
    return result
end

Base.:-(h::HarmonicFunction{T}) where T =
    HarmonicFunction{T}(-h.û, h.Mspan, h.is_real)

function Base.:*(α::Number, h::HarmonicFunction{T}) where T
    return HarmonicFunction{T}(α .* h.û, h.Mspan, h.is_real && α isa Real)
end

Base.:*(h::HarmonicFunction, α::Number) = α * h
Base.:/(h::HarmonicFunction, α::Number) = (1/α) * h

# ─── Per-frequency derivative operators ──────────────────────────────────────

∂ζ_HarmonicFunction!(dû_harmonic, û_harmonic, Mspan::Vector{Int}) =
    ∂ζ_harmonic!(dû_harmonic, û_harmonic, Mspan)

∂ζ_HarmonicFunction!(dû_harmonic, û_harmonic, D) =
    ∂ζ_harmonic!(dû_harmonic, û_harmonic, vec(Array(D.Mspan)))

∂ζ̄_HarmonicFunction!(dû_harmonic, û_harmonic, Mspan::Vector{Int}) =
    ∂ζ̄_harmonic!(dû_harmonic, û_harmonic, Mspan)

∂ζ̄_HarmonicFunction!(dû_harmonic, û_harmonic, D) =
    ∂ζ̄_harmonic!(dû_harmonic, û_harmonic, vec(Array(D.Mspan)))

function ∂²ζ_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    ∂ζ_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    ∂ζ_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    return nothing
end

function ∂²ζ̄_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    ∂ζ̄_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    ∂ζ̄_HarmonicFunction!(dû_harmonic, û_harmonic, D)
    return nothing
end

# ─── Grid-space derivatives ───────────────────────────────────────────────────

"""
    ∂ζ(h::HarmonicFunction, D)

Complex derivative ∂ζ of a harmonic function, returned as nodal values on the disk grid.

# Arguments
- `h` : harmonic function in coefficient space
- `D` : disk discretization

# Returns
- grid function ∂ζu on `D.ζ`
"""
function ∂ζ(h::HarmonicFunction, D::disk)
    dû = zeros(ComplexF64, length(h.û))
    ∂ζ_HarmonicFunction!(dû, h.û, D)
    return EvaluateHarmonicFunction(dû, D)
end

"""
    ∂ζ̄(h::HarmonicFunction, D)

Conjugate derivative ∂ζ̄ of a harmonic function, returned as nodal values on the disk grid.

# Arguments
- `h` : harmonic function in coefficient space
- `D` : disk discretization

# Returns
- grid function ∂ζ̄u on `D.ζ`
"""
function ∂ζ̄(h::HarmonicFunction, D::disk)
    dû = zeros(ComplexF64, length(h.û))
    ∂ζ̄_HarmonicFunction!(dû, h.û, D)
    return EvaluateHarmonicFunction(dû, D)
end
