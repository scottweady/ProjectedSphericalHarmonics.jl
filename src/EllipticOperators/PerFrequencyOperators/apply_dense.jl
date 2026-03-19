"""
    Inverse_laplacian_coef_m_positive(f̂ᵐ, lmax, m; aliasing=true)

Apply Δ⁻¹ to the frequency-`m` spherical harmonic coefficients using the dense layout,
for `m ≥ 0`.

The input `f̂ᵐ` holds coefficients for degrees `l = m, m+1, …, lmax`; only entries with
`l + m` even are non-zero (odd entries are ignored). The output has the same indexing
convention; with `aliasing=true` two extra entries are appended to capture the degree
elevation introduced by Δ⁻¹.

# Arguments
- `f̂ᵐ`      : dense coefficient vector for frequency `m`, length `lmax - m + 1`
- `lmax`    : maximum spherical-harmonic degree in `f̂ᵐ`
- `m`       : azimuthal frequency (`m ≥ 0`)
- `aliasing`: if `true`, allocate 2 extra output entries for aliased modes (default `true`)

# Returns
- `Δ⁻¹f̂ᵐ` : dense coefficient vector of length `length(f̂ᵐ) + 2*aliasing`
"""
function Inverse_laplacian_coef_m_positive(f̂ᵐ, lmax, m; aliasing=true)

    # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
    # We recall that l+m has to be even, otherwise is 0
    # We omit the negative m, since the coefficients are there conjugate

    Δ⁻¹f̂ᵐ = zeros(ComplexF64, length(f̂ᵐ) + 2 * aliasing)

    Δ⁻¹f̂ᵐ[1] += -1 / ((2m + 3) * (2m + 1)) * f̂ᵐ[1]
    if length(Δ⁻¹f̂ᵐ) > 2
        Δ⁻¹f̂ᵐ[3] += -Nlm(m, m, m + 2, m) / ((2m + 3) * (2m + 1) * (2m + 2)) * f̂ᵐ[1]
    end

    for l in m+2:lmax
        i = (l - m) + 1

        if (l + m) % 2 == 0
            Δ⁻¹f̂ᵐ[i] += -2 / ((2l - 1) * (2l + 3)) * f̂ᵐ[i]
            Δ⁻¹f̂ᵐ[i - 2] += -(l + m) / ((2l + 1) * (2l - 1) * (l - m - 1)) * Nlm(l, m, l - 2, m) * f̂ᵐ[i]
            if l < lmax && l + 1 < lmax
                Δ⁻¹f̂ᵐ[i + 2] += -(l - m + 1) / ((2l + 1) * (2l + 3) * (l + m + 2)) * Nlm(l, m, l + 2, m) * f̂ᵐ[i]
            elseif l == lmax && aliasing || (l == lmax - 1 && aliasing)
                Δ⁻¹f̂ᵐ[i + 2] += -(l - m + 1) / ((2l + 1) * (2l + 3) * (l + m + 2)) * Nlm(l, m, l + 2, m) * f̂ᵐ[i]
            end
        end
    end

    return Δ⁻¹f̂ᵐ
end

"""
    Inverse_laplacian_coef_m(f̂ᵐ, lmax, m; aliasing=true)

Apply Δ⁻¹ to the frequency-`m` spherical harmonic coefficients using the dense layout.
Delegates to [`Inverse_laplacian_coef_m_positive`](@ref) for `m ≥ 0`; for `m < 0` uses
conjugate symmetry: conjugates the input, calls the `|m|` version, then conjugates the
result.

# Arguments
- `f̂ᵐ`      : dense coefficient vector for frequency `m`, length `lmax - |m| + 1`
- `lmax`    : maximum spherical-harmonic degree in `f̂ᵐ`
- `m`       : azimuthal frequency (any sign)
- `aliasing`: if `true`, allocate 2 extra output entries for aliased modes (default `true`)

# Returns
- `Δ⁻¹f̂ᵐ` : dense coefficient vector of length `length(f̂ᵐ) + 2*aliasing`
"""
function Inverse_laplacian_coef_m(f̂ᵐ, lmax, m; aliasing=true)

    # f̂ᵐ are the coefficients on the ylm basis, i.e. f̂ᵐ = [f̂ᵐₘ, f̂ᵐₘ₊₁, ..., f̂ᵐₗ_ₘₐₓ]
    # We recall that l+m has to be even, otherwise is 0
    # We omit the negative m, since the coefficients are there conjugate

    if m < 0
        return conj.(Inverse_laplacian_coef_m_positive(conj.(f̂ᵐ), lmax, -m; aliasing=aliasing))
    end

    return Inverse_laplacian_coef_m_positive(f̂ᵐ, lmax, m; aliasing=aliasing)
end
