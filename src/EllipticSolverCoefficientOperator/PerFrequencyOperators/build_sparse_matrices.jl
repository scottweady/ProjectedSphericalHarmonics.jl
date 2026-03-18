using BandedMatrices
using FastAlmostBandedMatrices

"""
    inverse_laplacian_matrix_sparse(lmax, m; rectangular=false)

Build the banded matrix representing Δ⁻¹ acting on the sparse (l+m even) coefficients at
frequency `m`.

The columns correspond to input degrees `l = |m|, |m|+2, …, lmax` and the rows to output
degrees in the same range. For `m < 0` the matrix equals that of `|m|` (Δ⁻¹ commutes with
conjugation).

# Arguments
- `lmax`        : maximum spherical-harmonic degree
- `m`           : azimuthal frequency (any sign)
- `rectangular` : if `true`, add one extra row to capture the aliased output degree (default `false`)

# Returns
- `A` : `BandedMatrix` of size `(n + rectangular, n)` where `n = floor((lmax + 2 - |m|) / 2)`
"""
function inverse_laplacian_matrix_sparse(lmax, m; rectangular = false)
    # The matrix has dimensions (ceil((lmax-m)/2), ceil((lmax-m)/2)) since we only consider coefficients with l+m even.

    if m < 0
        return inverse_laplacian_matrix_sparse(lmax, -m; rectangular = rectangular)
    end

    n = floor(Int, (lmax + 2 - m) / 2)
    A = BandedMatrix(BandedMatrices.FillArrays.Zeros(n + rectangular, n), (1, 1))

    A[1, 1] = -1 / ((2m + 3) * (2m + 1))
    A[2, 1] = -Nlm(m, m, m + 2, m) / ((2m + 3) * (2m + 1) * (2m + 2))

    for l in m+2:2:lmax
        i = (l - m) ÷ 2 + 1
        A[i, i] = -2 / ((2l - 1) * (2l + 3))
        A[i - 1, i] = -(l + m) / ((2l + 1) * (2l - 1) * (l - m - 1)) * Nlm(l, m, l - 2, m)

        if l < lmax && l + 1 < lmax || rectangular
            A[i + 1, i] = -(l - m + 1) / ((2l + 1) * (2l + 3) * (l + m + 2)) * Nlm(l, m, l + 2, m)
        end
    end

    return A
end

"""
    ζ∂ζΔ⁻¹_matrix_sparse(lmax, m; rectangular=false)

Build the banded matrix representing ζ∂ζΔ⁻¹ acting on the sparse (l+|m| even)
coefficients at frequency `m`.

For `m ≥ 0` the coefficients of ζ∂ζΔ⁻¹f are computed; for `m < 0` the conjugate operator
ζ̄∂ζ̄Δ⁻¹ at frequency `|m|` is used (the matrix entries differ between the two cases).
The output frequency equals the input frequency in both cases.

# Arguments
- `lmax`        : maximum spherical-harmonic degree
- `m`           : azimuthal frequency (any sign)
- `rectangular` : if `true`, add one extra row to capture the aliased output degree (default `false`)

# Returns
- `A` : `BandedMatrix` of size `(n + rectangular, n)` where `n = floor((lmax + 2 - |m|) / 2)`
"""
function ζ∂ζΔ⁻¹_matrix_sparse(lmax, m; rectangular = false)
    # Banded matrix for ζ_∂Ĝᵐ∂ζ on the sparse same-mode coefficients.

    p = abs(m)
    n = floor(Int, (lmax + 2 - p) / 2)
    A = BandedMatrix(zeros(ComplexF64, n + rectangular, n), (1, 1))

    if m >= 0
        A[1, 1] = 1 / (2 * (2p + 1) * (2p + 3))
        if size(A, 1) >= 2
            A[2, 1] = -(1 / (2 * (2p + 1) * (2p + 3))) * Nlm(p, p, p + 2, p)
        end

        for l in p+2:2:lmax
            i = (l - p) ÷ 2 + 1
            A[i, i] = (((l - p + 1) / (2l + 3)) - ((l + p) / (2l - 1))) / (2 * (2l + 1))
            A[i - 1, i] = ((l + p) / (2 * (2l + 1) * (2l - 1))) * Nlm(l, p, l - 2, p)

            if ((l < lmax && l + 1 < lmax) || rectangular) && i + 1 <= size(A, 1)
                A[i + 1, i] = -((l - p + 1) / (2 * (2l + 1) * (2l + 3))) * Nlm(l, p, l + 2, p)
            end
        end
    else
        A[1, 1] = 1 / (2 * (2p + 3))
        if size(A, 1) >= 2
            A[2, 1] = -(1 / (2 * (2p + 1) * (2p + 3) * (p + 1))) * Nlm(p, p, p + 2, p)
        end

        for l in p+2:2:lmax
            i = (l - p) ÷ 2 + 1
            A[i, i] = (((l + p + 1) / (2l + 3)) - ((l - p) / (2l - 1))) / (2 * (2l + 1))
            A[i - 1, i] = ((l + p) * (l + p - 1) / (2 * (2l + 1) * (2l - 1) * (l - p - 1))) * Nlm(l, p, l - 2, p)

            if ((l < lmax && l + 1 < lmax) || rectangular) && i + 1 <= size(A, 1)
                A[i + 1, i] = -((l - p + 1) * (l - p + 2) / (2 * (2l + 1) * (2l + 3) * (l + p + 2))) * Nlm(l, p, l + 2, p)
            end
        end
    end

    return A
end

"""
    r_dot_∇Δ⁻¹_matrix_sparse(lmax, m; rectangular=false)

Build the banded matrix representing r·∇Δ⁻¹ acting on the sparse (l+|m| even) coefficients
at frequency `m`.

Computed as the sum of [`ζ∂ζΔ⁻¹_matrix_sparse`](@ref) at `m` and at `-m`, which
corresponds to ζ∂ζΔ⁻¹ + ζ̄∂ζ̄Δ⁻¹.

# Arguments
- `lmax`        : maximum spherical-harmonic degree
- `m`           : azimuthal frequency (any sign)
- `rectangular` : if `true`, add one extra row to capture the aliased output degree (default `false`)

# Returns
- `A` : `BandedMatrix` of size `(n + rectangular, n)` where `n = floor((lmax + 2 - |m|) / 2)`
"""
function r_dot_∇Δ⁻¹_matrix_sparse(lmax, m; rectangular = false)
    return ζ∂ζΔ⁻¹_matrix_sparse(lmax, m; rectangular = rectangular) +
           ζ∂ζΔ⁻¹_matrix_sparse(lmax, -m; rectangular = rectangular)
end
