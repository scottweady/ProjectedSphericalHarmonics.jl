function _check_inplace_length(opname, res, minlen)
    if length(res) < minlen
        throw(ArgumentError("`$opname`: `res` length $(length(res)) is too small; expected at least $minlen"))
    end
end

"""
    Inverse_laplacian_coef_m_sparse!(res, f̂ᵐ, lmax, m)

Apply Δ⁻¹ to the frequency-`m` spherical harmonic coefficients in-place, using the sparse
coefficient layout (only degrees `l` with `l + m` even are stored).

The sparse index `i` corresponds to degree `l = |m| + 2*(i-1)`. For `m < 0`, conjugate
symmetry is used: the result is `conj(Δ⁻¹(conj(f̂ᵐ)))` evaluated at frequency `-m`.

# Arguments
- `res`  : output vector; must have length `≥ length(f̂ᵐ)`
- `f̂ᵐ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function Inverse_laplacian_coef_m_sparse!(res, f̂ᵐ, lmax, m)
    fill!(res, 0)

    if m < 0
        tmp = similar(res)
        Inverse_laplacian_coef_m_sparse!(tmp, conj.(f̂ᵐ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    _check_inplace_length("Inverse_laplacian_coef_m_sparse!", res, length(f̂ᵐ))
    aliased_output = length(res) > length(f̂ᵐ)

    if isempty(f̂ᵐ)
        return res
    end

    res[1] += -1 / ((2m + 3) * (2m + 1)) * f̂ᵐ[1]
    if length(res) >= 2
        res[2] += -Nlm(m, m, m + 2, m) / ((2m + 3) * (2m + 1) * (2m + 2)) * f̂ᵐ[1]
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        if i > length(f̂ᵐ) || i > length(res)
            break
        end
        res[i] += -2 / ((2l - 1) * (2l + 3)) * f̂ᵐ[i]
        if i - 1 >= 1
            res[i - 1] += -(l + m) / ((2l + 1) * (2l - 1) * (l - m - 1)) * Nlm(l, m, l - 2, m) * f̂ᵐ[i]
        end
        if i + 1 <= length(res) && ((l < lmax && l + 1 < lmax) || aliased_output)
            res[i + 1] += -(l - m + 1) / ((2l + 1) * (2l + 3) * (l + m + 2)) * Nlm(l, m, l + 2, m) * f̂ᵐ[i]
        end
    end

    return res
end


#First Derivative

"""
    ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m; aliasing=false)

Apply ∂ζΔ⁻¹ to the frequency-`m` sparse coefficients in-place.

∂ζ decrements the azimuthal frequency by 1, so the output represents coefficients at
frequency `m - 1`. For `m < 0`, the operator is mapped to `conj(∂ζ̄Δ⁻¹(conj(μ̂ₘ)))` at
frequency `-m` via conjugate symmetry.

# Arguments
- `res`      : output vector at frequency `m - 1`; must have length `≥ ∂ζ_indexing_sparse(lmax, m; aliasing)`
- `μ̂ₘ`      : sparse coefficient vector for frequency `m`
- `lmax`     : maximum spherical-harmonic degree
- `m`        : azimuthal frequency (any sign)
- `aliasing` : if `true`, allow extra output entries for aliased modes (default `false`)

# Returns
- `res`
"""
function ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m; aliasing = false)
    fill!(res, 0)

    base_len = ∂ζ_indexing_sparse(lmax, m; aliasing = aliasing)
    _check_inplace_length("∂ζΔ⁻¹_m_sparse!", res, base_len)
    aliased_output = length(res) > base_len

    if m < 0
        tmp = similar(res)
        ∂ζ̄Δ⁻¹_m_sparse!(tmp, conj.(μ̂ₘ), lmax, -m; aliasing = aliasing)
        res .= conj.(tmp)
        return res
    end

    if isempty(μ̂ₘ)
        return res
    end



    if m > 0 && length(res) >= 2
        res[2] += 1 / 2 / (2m + 1) * _order_conversion_factor(m + 1, m - 1) * Nlm(m, m, m + 1, m - 1) * μ̂ₘ[1]
    

        for l in m+2:2:lmax
            i = (l - m) ÷ 2 + 1
            if i > length(μ̂ₘ) || i > length(res)
                break
            end
            res[i] += (l + m) * 1 / 2 / (2l + 1) * _order_conversion_factor(l - 1, m - 1) * Nlm(l, m, l - 1, m - 1) * μ̂ₘ[i]
            if i + 1 <= length(res) && ((l < lmax && l + 1 < lmax) || aliased_output)
                res[i+1] += (l - m + 1) * 1 / 2 / (2l + 1) * _order_conversion_factor(l + 1, m - 1) * Nlm(l, m, l + 1, m - 1) * μ̂ₘ[i]
            end
        end

    


    elseif m == 0

        res[1] += 1 / 2 / (2m + 1) * _order_conversion_factor(m + 1, m - 1) * Nlm(m, m, m + 1, m - 1) * μ̂ₘ[1]
    

        for l in m+2:2:lmax
            i = (l - m) ÷ 2 + 1
            if i > length(μ̂ₘ) || i > length(res)
                break
            end
            res[i-1] += (l + m) * 1 / 2 / (2l + 1) * _order_conversion_factor(l - 1, m - 1) * Nlm(l, m, l - 1, m - 1) * μ̂ₘ[i]
            if i + 1 <= length(res) && ((l < lmax && l + 1 < lmax) || aliased_output)
                res[i] += (l - m + 1) * 1 / 2 / (2l + 1) * _order_conversion_factor(l + 1, m - 1) * Nlm(l, m, l + 1, m - 1) * μ̂ₘ[i]
            end
        end



    end

    return res
    
end

"""
    ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m; aliasing=false)

Apply ∂ζ̄Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

∂ζ̄ increments the azimuthal frequency by 1, so the output represents coefficients at
frequency `m + 1`. For `m < 0`, the operator is mapped to `conj(∂ζΔ⁻¹(conj(μ̂ₘ)))` at
frequency `-m` via conjugate symmetry.

# Arguments
- `res`      : output vector at frequency `m + 1`; must have length `≥ ∂ζ̄_indexing_sparse(lmax, m; aliasing)`
- `μ̂ₘ`      : sparse coefficient vector for frequency `m`
- `lmax`     : maximum spherical-harmonic degree
- `m`        : azimuthal frequency (any sign)
- `aliasing` : if `true`, allow extra output entries for aliased modes (default `false`)

# Returns
- `res`
"""
function ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m; aliasing = false)
    fill!(res, 0)

    base_len = ∂ζ̄_indexing_sparse(lmax, m; aliasing = aliasing)
    _check_inplace_length("∂ζ̄Δ⁻¹_m_sparse!", res, base_len)
    aliased_output = length(res) > base_len

    if m < 0
        tmp = similar(res)
        ∂ζΔ⁻¹_m_sparse!(tmp, conj.(μ̂ₘ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    if isempty(μ̂ₘ)
        return res
    end

    if  length(res) >= 2
        res[1] += -1 / 4 / (2m + 1) / (m + 1) * Nlm(m, m, m + 1, m + 1) * μ̂ₘ[1]
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        if i > length(μ̂ₘ) || i > length(res)
            break
        end
        res[i-1] += -1 / (l - m - 1) * 1 / 2 / (2l + 1) * Nlm(l, m, l - 1, m + 1) * μ̂ₘ[i]
        if i <= length(res)
            res[i] += -1 / 2 * 1 / (2l + 1) / (l + m + 2) * Nlm(l, m, l + 1, m + 1) * μ̂ₘ[i]
        end
    end

    return res
end

"""
    ζ∂ζΔ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)

Apply ζ∂ζΔ⁻¹ to the frequency-`m` sparse coefficients in-place.

The output is at the same frequency `m` as the input. For `m < 0`, the operator is mapped
to `conj(ζ̄∂ζ̄Δ⁻¹(conj(f̂ᵐ)))` at frequency `-m` via conjugate symmetry.

# Arguments
- `res`  : output vector at frequency `m`; must have length `≥ length(f̂ᵐ)`
- `f̂ᵐ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function ζ∂ζΔ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)
    fill!(res, 0)
    _check_inplace_length("ζ∂ζΔ⁻¹_m_sparse!", res, length(f̂ᵐ))
    aliased_output = length(res) > length(f̂ᵐ)

    if m < 0
        tmp = similar(res)
        ζ̄∂ζ̄Δ⁻¹_m_sparse!(tmp, conj.(f̂ᵐ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    if isempty(f̂ᵐ)
        return res
    end

    move_index = false

    res[1] = 1 / (2 * (2m + 1) * (2m + 3)) * f̂ᵐ[1]
    if length(res) >= 2
        coeff = -(1 / (2 * (2m + 1) * (2m + 3))) * Nlm(m, m, m + 2, m)
        res[2- move_index] = coeff * f̂ᵐ[1]
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        if i > length(f̂ᵐ) || i > length(res)
            break
        end
        diag_coeff = (((l - m + 1) / (2l + 3)) - ((l + m) / (2l - 1))) / (2 * (2l + 1))
        sub_coeff = ((l + m) / (2 * (2l + 1) * (2l - 1))) * Nlm(l, m, l - 2, m)
        res[i- move_index] += diag_coeff * f̂ᵐ[i]
        if i - 1 >= 1
            res[i - 1 - move_index] += sub_coeff * f̂ᵐ[i]
        end
        if i + 1 <= length(res) && ((l < lmax && l + 1 < lmax) || aliased_output)
            super_coeff = -((l - m + 1) / (2 * (2l + 1) * (2l + 3))) * Nlm(l, m, l + 2, m)
            res[i + 1 - move_index] += super_coeff * f̂ᵐ[i]
        end
    end

    return res
end

"""
    ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)

Apply ζ̄∂ζ̄Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

The output is at the same frequency `m` as the input. For `m < 0`, the operator is mapped
to `conj(ζ∂ζΔ⁻¹(conj(f̂ᵐ)))` at frequency `-m` via conjugate symmetry.

# Arguments
- `res`  : output vector at frequency `m`; must have length `≥ length(f̂ᵐ)`
- `f̂ᵐ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)
    fill!(res, 0)
    _check_inplace_length("ζ̄∂ζ̄Δ⁻¹_m_sparse!", res, length(f̂ᵐ))
    aliased_output = length(res) > length(f̂ᵐ)

    if m < 0
        tmp = similar(res)
        ζ∂ζΔ⁻¹_m_sparse!(tmp, conj.(f̂ᵐ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    if isempty(f̂ᵐ)
        return res
    end

    move_index = (m==0)

    res[1] = 1 / (2 * (2m + 3)) * f̂ᵐ[1]
    if length(res) >= 2
        coeff = -(1 / (2 * (2m + 1) * (2m + 3) * (m + 1))) * Nlm(m, m, m + 2, m)
        res[2] = coeff * f̂ᵐ[1]
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        if i > length(f̂ᵐ) || i > length(res)
            break
        end
        diag_coeff = (((l + m + 1) / (2l + 3)) - ((l - m) / (2l - 1))) / (2 * (2l + 1))
        sub_coeff = ((l + m) * (l + m - 1) / (2 * (2l + 1) * (2l - 1) * (l - m - 1))) * Nlm(l, m, l - 2, m)
        res[i] += diag_coeff * f̂ᵐ[i]
        if i - 1 >= 1
            res[i - 1] += sub_coeff * f̂ᵐ[i]
        end
        if i + 1 <= length(res) && ((l < lmax && l + 1 < lmax) || aliased_output)
            super_coeff = -((l - m + 1) * (l - m + 2) / (2 * (2l + 1) * (2l + 3) * (l + m + 2))) * Nlm(l, m, l + 2, m)
            res[i + 1] += super_coeff * f̂ᵐ[i]
        end
    end

    return res
end

function _order_conversion_factor(l, q)
    if q >= 0
        return 1.0
    end

    k = abs(q)
    return (-1)^k * exp(loggamma(l - k + 1) - loggamma(l + k + 1))
end

function _normalized_second_derivative_ratio(l, m, p, q)
    return _order_conversion_factor(p, q) * Nlm(l, m, p, q)
end

"""
    ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)

Apply ∂ζ̄²Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

Two applications of ∂ζ̄ increment the azimuthal frequency by 2, so the output represents
coefficients at frequency `m + 2`. For `m < 0`, the operator is mapped to
`conj(∂ζ²Δ⁻¹(conj(μ̂ₘ)))` at frequency `-m`.

# Arguments
- `res`  : output vector at frequency `m + 2`; must have length `≥ size_current_m(lmax, m + 2)`
- `μ̂ₘ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
    fill!(res, 0)

    base_len = size_current_m(lmax, m + 2)
    _check_inplace_length("∂ζ̄∂ζ̄Δ⁻¹_m_sparse!", res, base_len)

    if m < 0
        tmp = similar(res)
        ∂ζ∂ζΔ⁻¹_m_sparse!(tmp, conj.(μ̂ₘ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    if length(μ̂ₘ) <= 1
        return res
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        j = i - 1
        if i > length(μ̂ₘ) || j > length(res)
            break
        end
        coeff = -1 / (4 * (l + m + 2) * (l - m - 1)) * _normalized_second_derivative_ratio(l, m, l, m + 2)
        res[j] += coeff * μ̂ₘ[i]
    end

    return res
end

"""
    ∂ζ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)

Apply ∂ζ²Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

Two applications of ∂ζ decrement the azimuthal frequency by 2, so the output represents
coefficients at frequency `m - 2`. For `m < 0`, the operator is mapped to
`conj(∂ζ̄²Δ⁻¹(conj(μ̂ₘ)))` at frequency `-m`.

# Arguments
- `res`  : output vector at frequency `m - 2`; must have length `≥ ∂ζ∂ζ_indexing_sparse(lmax, m)`
- `μ̂ₘ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function ∂ζ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
    fill!(res, 0)

    if m < 0
        tmp = similar(res)
        ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(tmp, conj.(μ̂ₘ), lmax, -m)
        res .= conj.(tmp)
        return res
    end

    target_m = m - 2
    base_len = size_current_m(lmax, target_m)
    _check_inplace_length("∂ζ∂ζΔ⁻¹_m_sparse!", res, base_len)

    if m == 0
        for l in 2:2:lmax
            i = l ÷ 2 + 1
            j = l ÷ 2
            if i > length(μ̂ₘ) || j > length(res)
                break
            end
            coeff = -(l * (l + 1) / 4) * _normalized_second_derivative_ratio(l, 0, l, -2)
            res[j] += coeff * μ̂ₘ[i]
        end
        return res
    end

    if m == 1 && !isempty(μ̂ₘ)
        res[1] += -(1 / 2) * _normalized_second_derivative_ratio(1, 1, 1, -1) * μ̂ₘ[1]
    end

    if m >= 2 && !isempty(μ̂ₘ)
        leading_coeff_same_l = -(m / 2) * Nlm(m, m, m, m - 2)
        leading_coeff_lower_l = (m * (2m - 3) * (m - 1) / (2m + 1)) * Nlm(m, m, m - 2, m - 2)
        if length(res) >= 2
            res[2] += leading_coeff_same_l * μ̂ₘ[1]
        end
        res[1] += leading_coeff_lower_l * μ̂ₘ[1]
    end

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        j = (l - abs(target_m)) ÷ 2 + 1
        if i > length(μ̂ₘ) || j > length(res)
            break
        end
        coeff = -((l + m) * (l - m + 1) / 4) * _normalized_second_derivative_ratio(l, m, l, m - 2)
        res[j] += coeff * μ̂ₘ[i]
    end

    return res
end

"""
    ∂ζ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)

Apply ∂ζ∂ζ̄Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

The mixed partial ∂ζ∂ζ̄ preserves the azimuthal frequency, so the output is at the same
frequency `m`. Due to the identity ∂ζ∂ζ̄Δ⁻¹ = 1/4 on the relevant function space, this
simply scales the input by 1/4.

# Arguments
- `res`  : output vector at frequency `m`; must have length `≥ size_current_m(lmax, m)`
- `μ̂ₘ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function ∂ζ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
    fill!(res, 0)
    base_len = size_current_m(lmax, m)
    _check_inplace_length("∂ζ∂ζ̄Δ⁻¹_m_sparse!", res, base_len)
    res[1:length(μ̂ₘ)] .= μ̂ₘ ./ 4
    return res
end

"""
    r_dot_∇Δ⁻¹!(res, f̂ᵐ, lmax, m)

Apply r·∇Δ⁻¹ to the frequency-`m` sparse coefficients in-place.

Computed as `ζ∂ζΔ⁻¹ + ζ̄∂ζ̄Δ⁻¹`, both of which preserve the azimuthal frequency, so the
output is at frequency `m`.

# Arguments
- `res`  : output vector at frequency `m`; must have length `≥ length(f̂ᵐ)`
- `f̂ᵐ`  : sparse coefficient vector for frequency `m`
- `lmax` : maximum spherical-harmonic degree
- `m`    : azimuthal frequency (any sign)

# Returns
- `res`
"""
function r_dot_∇Δ⁻¹!(res, f̂ᵐ, lmax, m)
    tmp = similar(res)
    ζ∂ζΔ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)
    ζ̄∂ζ̄Δ⁻¹_m_sparse!(tmp, f̂ᵐ, lmax, m)
    res .+= tmp
    return res
end

# Out-of-place wrappers.

"""
    Inverse_laplacian_coef_m_sparse(f̂ᵐ, lmax, m; aliasing=true)

Out-of-place version of [`Inverse_laplacian_coef_m_sparse!`](@ref). Allocates the result
vector and delegates to the in-place implementation.
"""
function Inverse_laplacian_coef_m_sparse(f̂ᵐ, lmax, m; aliasing=true)
    res = zeros(ComplexF64, length(f̂ᵐ) + aliasing)
    return Inverse_laplacian_coef_m_sparse!(res, f̂ᵐ, lmax, m)
end

"""
    ∂ζΔ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)

Out-of-place version of [`∂ζΔ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ∂ζΔ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)
    n = ∂ζ_indexing_sparse(lmax, m; aliasing = aliasing)
    res = zeros(ComplexF64, n)
    return ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
end

"""
    ∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)

Out-of-place version of [`∂ζ̄Δ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)
    n = ∂ζ̄_indexing_sparse(lmax, m; aliasing = aliasing)
    res = zeros(ComplexF64, n)
    return ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
end

"""
    ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=true)

Out-of-place version of [`ζ∂ζΔ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=true)
    res = zeros(ComplexF64, length(f̂ᵐ) + aliasing)
    return ζ∂ζΔ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)
end

"""
    ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=true)

Out-of-place version of [`ζ̄∂ζ̄Δ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, lmax, m; aliasing=true)
    res = zeros(ComplexF64, length(f̂ᵐ) + aliasing)
    return ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, f̂ᵐ, lmax, m)
end

"""
    ∂ζ̄∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)

Out-of-place version of [`∂ζ̄∂ζ̄Δ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ∂ζ̄∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)
    n = size_current_m(lmax, m + 2; aliasing = aliasing)
    res = zeros(ComplexF64, n)
    return ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
end

"""
    ∂ζ∂ζΔ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)

Out-of-place version of [`∂ζ∂ζΔ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ∂ζ∂ζΔ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)
    n = ∂ζ∂ζ_indexing_sparse(lmax, m; aliasing = aliasing)
    res = zeros(ComplexF64, n)
    return ∂ζ∂ζΔ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
end

"""
    ∂ζ∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)

Out-of-place version of [`∂ζ∂ζ̄Δ⁻¹_m_sparse!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function ∂ζ∂ζ̄Δ⁻¹_m_sparse(μ̂ₘ, lmax, m; aliasing=true)
    n = size_current_m(lmax, m; aliasing = aliasing)
    res = zeros(ComplexF64, n)
    return ∂ζ∂ζ̄Δ⁻¹_m_sparse!(res, μ̂ₘ, lmax, m)
end

"""
    r_dot_∇Δ⁻¹(f̂ᵐ, lmax, m; aliasing=true)

Out-of-place version of [`r_dot_∇Δ⁻¹!`](@ref). Allocates the result vector and
delegates to the in-place implementation.
"""
function r_dot_∇Δ⁻¹(f̂ᵐ, lmax, m; aliasing=true)
    res = zeros(ComplexF64, length(f̂ᵐ) + aliasing)
    return r_dot_∇Δ⁻¹!(res, f̂ᵐ, lmax, m)
end
