# apply_triangular.jl
# Triangular-array wrappers for the per-frequency operators in apply_sparse.jl.
# Each function applies Xᵐ across all frequencies stored in a TriangularCoeffArray.
#
# Naming convention: drop the supra-index (Ĝᵐ → Ĝ, ∂Ĝᵐ∂ζ → ∂Ĝ∂ζ, etc.).
#
# Same-frequency operators (res has the same Mspan as f̂):
#   Ĝ, ζ_∂Ĝ∂ζ, ζ̄_∂Ĝ∂ζ̄, ∂²Ĝ∂ζ∂ζ̄, r_∂Ĝ∂r
#
# Frequency-shifting operators (res.Mspan = f̂.Mspan .± shift):
#   ∂Ĝ∂ζ  (shift −1)   ∂Ĝ∂ζ̄  (shift +1)
#   ∂²Ĝ∂ζ² (shift −2)  ∂²Ĝ∂ζ̄² (shift +2)
#
# For the in-place versions of frequency-shifting operators the caller must
# supply a pre-allocated `res` whose Mspan and column sizes match the shifted
# output frequencies.

# ── Inverse Laplacian ─────────────────────────────────────────────────────────

"""
    Ĝ!(res, f̂, lmax)

Apply 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place, storing results in `res`.

Calls [`Ĝᵐ!`](@ref) for each frequency in `f̂.Mspan`. `res` the Mspan of res
must constain the Mspan of f̂. res.lmax ≥ f̂.lmax

# Arguments
- `res`  : output `TriangularCoeffArray`; same structure as `f̂`
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function Ĝ!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    lmax = f̂.lmax
    if lmax > res.lmax
        error("Error applying Ĝ!: res has a smaller lmax than f̂")
    end
    for (i, m) in enumerate(f̂.Mspan)
        Ĝᵐ!(mode_coefficients(res, m), mode_coefficients(f̂, m), lmax, m)
    end
    return res
end

"""
    Ĝ(f̂, lmax)

Out-of-place version of [`Ĝ!`](@ref). Allocates the result and delegates to
the in-place implementation.
"""
function Ĝ(f̂::TriangularCoeffArray)
    res = zero(f̂)
    Ĝ!(res, f̂)
    return res
end

# ── First derivatives ─────────────────────────────────────────────────────────

"""
    ∂Ĝ∂ζ!(res, f̂)

Apply ∂/∂ζ 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`∂Ĝᵐ∂ζ!`](@ref) for each frequency `m` in `f̂.Mspan`. The i-th column
of `res` holds the output at frequency `f̂.Mspan[i] - 1`; `res` must be
pre-allocated with `res.Mspan = f̂.Mspan .- 1` and column sizes
`∂ζ_indexing_sparse(lmax, m; aliasing=false)` for each input frequency `m`.

# Arguments
- `res`  : output `TriangularCoeffArray` at shifted frequencies
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ∂Ĝ∂ζ!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    lmax_res = res.lmax
    lmax_f = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        if size_current_m(lmax_res, m-1) > 0
            ∂Ĝᵐ∂ζ!(mode_coefficients(res, m-1), mode_coefficients(f̂, m), lmax_f, m)
        end
    end
    return res
end

"""
    ∂Ĝ∂ζ(f̂, lmax)

Out-of-place version of [`∂Ĝ∂ζ!`](@ref). Allocates a `TriangularCoeffArray`
with `Mspan = f̂.Mspan .- 1` and delegates to the in-place implementation.
"""

function ∂Ĝ∂ζ(f̂::TriangularCoeffArray)

    res = zero(f̂)
    ∂Ĝ∂ζ!(res, f̂)
    return res

end

"""
    ∂Ĝ∂ζ̄!(res, f̂, lmax)

Apply ∂/∂ζ̄ 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`∂Ĝᵐ∂ζ̄!`](@ref) for each frequency `m` in `f̂.Mspan`. The i-th column
of `res` holds the output at frequency `f̂.Mspan[i] + 1`; `res` must be
pre-allocated with `res.Mspan = f̂.Mspan .+ 1` and column sizes
`∂ζ̄_indexing_sparse(lmax, m; aliasing=false)` for each input frequency `m`.

# Arguments
- `res`  : output `TriangularCoeffArray` at shifted frequencies
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ∂Ĝ∂ζ̄!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    
    lmax_res = res.lmax
    lmax_f = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        if size_current_m(lmax_res, m+1) > 0
            ∂Ĝᵐ∂ζ̄!(mode_coefficients(res, m+1), mode_coefficients(f̂, m), lmax_f, m)
        end
    end
    return res
    
end

"""
    ∂Ĝ∂ζ̄(f̂, lmax)

Out-of-place version of [`∂Ĝ∂ζ̄!`](@ref). Allocates a `TriangularCoeffArray`
with `Mspan = f̂.Mspan .+ 1` and delegates to the in-place implementation.
"""
function ∂Ĝ∂ζ̄(f̂::TriangularCoeffArray)

    res = zero(f̂)
    ∂Ĝ∂ζ̄!(res, f̂)
    return res
end

# ── ζ-weighted first derivatives ──────────────────────────────────────────────

"""
    ζ_∂Ĝ∂ζ!(res, f̂, lmax)

Apply ζ * ∂/∂ζ 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`ζ_∂Ĝᵐ∂ζ!`](@ref) for each frequency in `f̂.Mspan`. `res` must have
the same Mspan and per-frequency sizes as `f̂`.

# Arguments
- `res`  : output `TriangularCoeffArray`; same structure as `f̂`
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ζ_∂Ĝ∂ζ!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    
    lmax = f̂.lmax
    if lmax > res.lmax
        error("Error applying Ĝ!: res has a smaller lmax than f̂")
    end
    for (i, m) in enumerate(f̂.Mspan)
        ζ_∂Ĝᵐ∂ζ!(mode_coefficients(res, m), mode_coefficients(f̂, m), lmax, m)
    end
    return res
    
end

"""
    ζ_∂Ĝ∂ζ(f̂, lmax)

Out-of-place version of [`ζ_∂Ĝ∂ζ!`](@ref). Allocates the result and delegates
to the in-place implementation.
"""
function ζ_∂Ĝ∂ζ(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return ζ_∂Ĝ∂ζ!(res, f̂)
end

"""
    ζ̄_∂Ĝ∂ζ̄!(res, f̂, lmax)

Apply ζ̄ * ∂/∂ζ̄ 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`ζ̄_∂Ĝᵐ∂ζ̄!`](@ref) for each frequency in `f̂.Mspan`. `res` must have
the same Mspan and per-frequency sizes as `f̂`.

# Arguments
- `res`  : output `TriangularCoeffArray`; same structure as `f̂`
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ζ̄_∂Ĝ∂ζ̄!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    
    lmax = f̂.lmax
    if lmax > res.lmax
        error("Error applying Ĝ!: res has a smaller lmax than f̂")
    end
    for (i, m) in enumerate(f̂.Mspan)
        ζ̄_∂Ĝᵐ∂ζ̄!(mode_coefficients(res, m), mode_coefficients(f̂, m), lmax, m)
    end
    return res

end

"""
    ζ̄_∂Ĝ∂ζ̄(f̂, lmax)

Out-of-place version of [`ζ̄_∂Ĝ∂ζ̄!`](@ref). Allocates the result and delegates
to the in-place implementation.
"""
function ζ̄_∂Ĝ∂ζ̄(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return ζ̄_∂Ĝ∂ζ̄!(res, f̂)
end

# ── Second derivatives ────────────────────────────────────────────────────────

"""
    ∂²Ĝ∂ζ̄²!(res, f̂, lmax)

Apply (∂/∂ζ̄)² 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`∂²Ĝᵐ∂ζ̄²!`](@ref) for each frequency `m` in `f̂.Mspan`. The i-th
column of `res` holds the output at frequency `f̂.Mspan[i] + 2`; `res` must
be pre-allocated with `res.Mspan = f̂.Mspan .+ 2` and column sizes
`∂ζ̄∂ζ̄_indexing_sparse(lmax, m; aliasing=false)` for each input frequency `m`.

# Arguments
- `res`  : output `TriangularCoeffArray` at shifted frequencies
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ∂²Ĝ∂ζ̄²!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    lmax_res = res.lmax
    lmax_f = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        if size_current_m(lmax_res, m+2) > 0
            ∂²Ĝᵐ∂ζ̄²!(mode_coefficients(res, m+2), mode_coefficients(f̂, m), lmax_f, m)
        end
    end
    return res
end

"""
    ∂²Ĝ∂ζ̄²(f̂, lmax)

Out-of-place version of [`∂²Ĝ∂ζ̄²!`](@ref). Allocates a `TriangularCoeffArray`
with `Mspan = f̂.Mspan .+ 2` and delegates to the in-place implementation.
"""
function ∂²Ĝ∂ζ̄²(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return ∂²Ĝ∂ζ̄²!(res, f̂)
end

"""
    ∂²Ĝ∂ζ²!(res, f̂, lmax)

Apply (∂/∂ζ)² 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`∂²Ĝᵐ∂ζ²!`](@ref) for each frequency `m` in `f̂.Mspan`. The i-th
column of `res` holds the output at frequency `f̂.Mspan[i] - 2`; `res` must
be pre-allocated with `res.Mspan = f̂.Mspan .- 2` and column sizes
`∂ζ∂ζ_indexing_sparse(lmax, m; aliasing=false)` for each input frequency `m`.

# Arguments
- `res`  : output `TriangularCoeffArray` at shifted frequencies
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ∂²Ĝ∂ζ²!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)

    lmax_res = res.lmax
    lmax_f = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        if size_current_m(lmax_res, m-2) > 0
            ∂²Ĝᵐ∂ζ²!(mode_coefficients(res, m-2), mode_coefficients(f̂, m), lmax_f, m)
        end
    end
    return res
    
end

"""
    ∂²Ĝ∂ζ²(f̂, lmax)

Out-of-place version of [`∂²Ĝ∂ζ²!`](@ref). Allocates a `TriangularCoeffArray`
with `Mspan = f̂.Mspan .- 2` and delegates to the in-place implementation.
"""
function ∂²Ĝ∂ζ²(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return ∂²Ĝ∂ζ²!(res, f̂)
end

"""
    ∂²Ĝ∂ζ∂ζ̄!(res, f̂, lmax)

Apply ∂²/∂ζ∂ζ̄ 𝒮𝒩⁻¹ to all frequencies of `f̂` in-place.

Calls [`∂²Ĝᵐ∂ζ∂ζ̄!`](@ref) for each frequency in `f̂.Mspan`. `res` must have
the same Mspan and per-frequency sizes as `f̂`.

# Arguments
- `res`  : output `TriangularCoeffArray`; same structure as `f̂`
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function ∂²Ĝ∂ζ∂ζ̄!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    lmax = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        ∂²Ĝᵐ∂ζ∂ζ̄!(mode_coefficients(res, m), mode_coefficients(f̂, m), lmax, m)
    end
    return res
end

"""
    ∂²Ĝ∂ζ∂ζ̄(f̂, lmax)

Out-of-place version of [`∂²Ĝ∂ζ∂ζ̄!`](@ref). Allocates the result and
delegates to the in-place implementation.
"""
function ∂²Ĝ∂ζ∂ζ̄(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return ∂²Ĝ∂ζ∂ζ̄!(res, f̂)
end

# ── Radial gradient ───────────────────────────────────────────────────────────

"""
    r_∂Ĝ∂r!(res, f̂, lmax)

Apply r·∇Δ⁻¹ to all frequencies of `f̂` in-place.

Calls [`r_∂Ĝᵐ∂r!`](@ref) for each frequency in `f̂.Mspan`. `res` must have
the same Mspan and per-frequency sizes as `f̂`.

# Arguments
- `res`  : output `TriangularCoeffArray`; same structure as `f̂`
- `f̂`   : input `TriangularCoeffArray` of PSH coefficients
- `lmax` : maximum spherical-harmonic degree

# Returns
- `res`
"""
function r_∂Ĝ∂r!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)
    lmax = f̂.lmax
    for (i, m) in enumerate(f̂.Mspan)
        r_∂Ĝᵐ∂r!(mode_coefficients(res, m), mode_coefficients(f̂, m), lmax, m)
    end
    return res
end

"""
    r_∂Ĝ∂r(f̂, lmax)

Out-of-place version of [`r_∂Ĝ∂r!`](@ref). Allocates the result and delegates
to the in-place implementation.
"""
function r_∂Ĝ∂r(f̂::TriangularCoeffArray)
    res = zero(f̂)
    return r_∂Ĝ∂r!(res, f̂)
end
