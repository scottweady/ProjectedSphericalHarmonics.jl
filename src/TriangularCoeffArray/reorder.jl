# reorder.jl
# Column-reordering functions for TriangularCoeffArray between FFT ordering
# and natural (centered) ordering, using fftshift / ifftshift from AbstractFFTs.

using FFTW: fftshift, ifftshift

"""
    circshift_fft_to_natural(A) -> TriangularCoeffArray

Reorder columns from FFT ordering to natural (centered) ordering using `fftshift`.

FFT ordering produced by `fftfreq(MT, MT)`:
- odd  MT, M = MT÷2: `[0, 1, …, M, -M, …, -1]`
- even MT, M = MT÷2: `[0, 1, …, M-1, -M, …, -1]`

After `fftshift` (natural ordering):
- odd  MT: `[-M, …, 0, …, M]`
- even MT: `[-M, …, 0, …, M-1]`  (Nyquist at -M)

The result has `ordering=:natural`; parity is preserved.
Only defined for `ordering=:fft` arrays.
"""
function circshift_fft_to_natural(A::TriangularCoeffArray{T,N,P,:fft}) where {T,N,P}
    new_Mspan = fftshift(A.Mspan)
    new_data  = [copy(v) for v in fftshift(A.data)]
    return TriangularCoeffArray(new_Mspan, new_data; parity = P, ordering = :natural)
end

"""
    circshift_natural_to_fft(A) -> TriangularCoeffArray

Reorder columns from natural (centered) ordering back to FFT ordering using
`ifftshift`. Inverse of `circshift_fft_to_natural`.

The result has `ordering=:fft`; parity is preserved.
Only defined for `ordering=:natural` arrays.
"""
function circshift_natural_to_fft(A::TriangularCoeffArray{T,N,P,:natural}) where {T,N,P}
    new_Mspan = ifftshift(A.Mspan)
    new_data  = [copy(v) for v in ifftshift(A.data)]
    return TriangularCoeffArray(new_Mspan, new_data; parity = P, ordering = :fft)
end
