#TODO: treat Nyquist term explicitly

"""
    harmonic_coeff_from_dirichlet!(û, ĝ, Mspan)

In-place: recover harmonic coefficients `û` from Fourier-transformed Dirichlet
boundary data `ĝ`.  Both vectors are FFT-ordered (length `2Mθ+1`).

For each frequency m: `û[i] = ĝ[i] / ylm(|m|, m, 1.0)`.

# Arguments
- `û`     : output harmonic coefficients (mutated)
- `ĝ`     : Fourier boundary data, FFT-ordered
- `Mspan` : FFT-ordered frequency list `[0, 1, …, Mθ, -Mθ, …, -1]`

# Returns
- `û` (mutated in place)
"""
function harmonic_coeff_from_dirichlet!(û, ĝ, Mspan::AbstractVector{Int})
    for (i, m) in enumerate(Mspan)
        û[i] = ĝ[i] / ylm(abs(m), m, 1.0)
    end
    return û
end

"""
    harmonic_coeff_from_dirichlet(ĝ, Mspan)

Out-of-place version of `harmonic_coeff_from_dirichlet!`.

# Returns
- new vector of harmonic coefficients, FFT-ordered
"""
harmonic_coeff_from_dirichlet(ĝ, Mspan::AbstractVector{Int}) =
    harmonic_coeff_from_dirichlet!(zeros(ComplexF64, length(ĝ)), ĝ, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    dirichlet_trace_harmonic!(ĝ, û, Mspan)

In-place: compute the Fourier-transformed Dirichlet boundary trace `ĝ` of a
harmonic function from its coefficients `û`.

For each frequency m: `ĝ[i] = û[i] * ylm(|m|, m, 1.0)`.

# Arguments
- `ĝ`     : output Fourier boundary data (mutated)
- `û`     : harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `ĝ` (mutated in place)
"""
function dirichlet_trace_harmonic!(ĝ, û, Mspan::AbstractVector{Int})
    for (i, m) in enumerate(Mspan)
        ĝ[i] = û[i] * ylm(abs(m), m, 1.0)
    end
    return ĝ
end

"""
    dirichlet_trace_harmonic(û, Mspan)

Out-of-place version of `dirichlet_trace_harmonic!`.

# Returns
- new vector of Fourier Dirichlet boundary data, FFT-ordered
"""
dirichlet_trace_harmonic(û, Mspan::AbstractVector{Int}) =
    dirichlet_trace_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    ζ_∂ζ_harmonic!(res, û, Mspan)

In-place: compute the harmonic coefficients of `ζ ∂ζ h`.

This is a same-frequency operator: frequency m maps to m (no shift).
Only modes with m ≥ 1 contribute; all other entries are zero.

For m ≥ 1: `res[i] = m * û[i]`.

# Arguments
- `res`   : output harmonic coefficients (mutated, zeroed first)
- `û`     : input harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `res` (mutated in place)
"""
function ζ_∂ζ_harmonic!(res, û, Mspan::AbstractVector{Int})
    fill!(res, 0)
    for (i, m) in enumerate(Mspan)
        m < 1 && continue
        res[i] = m * û[i]
    end
    return res
end

"""
    ζ_∂ζ_harmonic(û, Mspan)

Out-of-place version of `ζ_∂ζ_harmonic!`.
"""
ζ_∂ζ_harmonic(û, Mspan::AbstractVector{Int}) =
    ζ_∂ζ_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    ζ̄_∂ζ̄_harmonic!(res, û, Mspan)

In-place: compute the harmonic coefficients of `ζ̄ ∂ζ̄ h`.

This is a same-frequency operator: frequency m maps to m (no shift).
Only modes with m ≤ −1 contribute; all other entries are zero.

For m ≤ −1: `res[i] = |m| * û[i]`.

# Arguments
- `res`   : output harmonic coefficients (mutated, zeroed first)
- `û`     : input harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `res` (mutated in place)
"""
function ζ̄_∂ζ̄_harmonic!(res, û, Mspan::AbstractVector{Int})
    fill!(res, 0)
    for (i, m) in enumerate(Mspan)
        m > -1 && continue
        res[i] = abs(m) * û[i]
    end
    return res
end

"""
    ζ̄_∂ζ̄_harmonic(û, Mspan)

Out-of-place version of `ζ̄_∂ζ̄_harmonic!`.
"""
ζ̄_∂ζ̄_harmonic(û, Mspan::AbstractVector{Int}) =
    ζ̄_∂ζ̄_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    neumann_trace_harmonic!(n̂, û, Mspan)

In-place: compute the Fourier-transformed Neumann trace `∂ᵣh|_{r=1}` of a
harmonic function from its coefficients `û`.

Uses the identity `r∂ᵣ = ζ∂ζ + ζ̄∂ζ̄`: computes `(ζ∂ζ + ζ̄∂ζ̄)h` in
coefficient space and then evaluates its Dirichlet trace.

# Arguments
- `n̂`     : output Fourier Neumann data (mutated)
- `û`     : harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `n̂` (mutated in place)
"""
function neumann_trace_harmonic!(n̂, û, Mspan::AbstractVector{Int})
    tmp = zeros(ComplexF64, length(û))
    ζ_∂ζ_harmonic!(tmp, û, Mspan)
    ζ̄_part = zeros(ComplexF64, length(û))
    ζ̄_∂ζ̄_harmonic!(ζ̄_part, û, Mspan)
    tmp .+= ζ̄_part
    return dirichlet_trace_harmonic!(n̂, tmp, Mspan)
end

"""
    neumann_trace_harmonic(û, Mspan)

Out-of-place version of `neumann_trace_harmonic!`.
"""
neumann_trace_harmonic(û, Mspan::AbstractVector{Int}) =
    neumann_trace_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    ∂ζ_harmonic!(dû, û, Mspan)

In-place: compute the harmonic-coefficient vector of ∂ζh.

∂ζ shifts the azimuthal frequency by −1: contributions at frequency m land at
frequency m−1.  Only modes with m ≥ 1 contribute; all other entries are zero.

For m ≥ 1:
    dû[idx(m-1)] = m * û[idx(m)] * ylm(m, m, 1.0) / ylm(m-1, m-1, 1.0)

# Arguments
- `dû`    : output harmonic coefficients at shifted frequencies (mutated, zeroed first)
- `û`     : input harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `dû` (mutated in place)
"""
function ∂ζ_harmonic!(dû, û, Mspan::AbstractVector{Int})
    fill!(dû, 0)
    for (i, m) in enumerate(Mspan)
        m < 1 && continue
        dû[i-1] = m * û[i] * ylm(abs(m), m, 1.0) / ylm(abs(m-1), m-1, 1.0)
    end
    return dû
end

"""
    ∂ζ_harmonic(û, Mspan)

Out-of-place version of `∂ζ_harmonic!`.

# Returns
- new vector of harmonic coefficients of ∂ζh, FFT-ordered
"""
∂ζ_harmonic(û, Mspan::AbstractVector{Int}) =
    ∂ζ_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)

# ──────────────────────────────────────────────────────────────────────────────

"""
    ∂ζ̄_harmonic!(dû, û, Mspan)

In-place: compute the harmonic-coefficient vector of ∂ζ̄h.

∂ζ̄ shifts the azimuthal frequency by +1: contributions at frequency m land at
frequency m+1.  Only modes with m ≤ −1 contribute; all other entries are zero.

For m ≤ −1:
    dû[idx(m+1)] = |m| * û[idx(m)] * ylm(|m|, m, 1.0) / ylm(|m+1|, m+1, 1.0)

# Arguments
- `dû`    : output harmonic coefficients at shifted frequencies (mutated, zeroed first)
- `û`     : input harmonic coefficients, FFT-ordered
- `Mspan` : FFT-ordered frequency list

# Returns
- `dû` (mutated in place)
"""
function ∂ζ̄_harmonic!(dû, û, Mspan::AbstractVector{Int})
    fill!(dû, 0)
    for (i, m) in enumerate(Mspan)
        m > -1 && continue
        if m == -1
            dû[1] = û[i] * ylm(1, -1, 1.0) / ylm(0, 0, 1.0)
        else
            dû[i+1] = abs(m) * û[i] * ylm(abs(m), m, 1.0) / ylm(abs(m+1), m+1, 1.0)
        end
    end
    return dû
end

"""
    ∂ζ̄_harmonic(û, Mspan)

Out-of-place version of `∂ζ̄_harmonic!`.

# Returns
- new vector of harmonic coefficients of ∂ζ̄h, FFT-ordered
"""
∂ζ̄_harmonic(û, Mspan::AbstractVector{Int}) =
    ∂ζ̄_harmonic!(zeros(ComplexF64, length(û)), û, Mspan)
