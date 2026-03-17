"""
    AbstractDiskFunction{T<:Real, N}

Abstract supertype for functions on the disk stored in PSH coefficient space.

`T` is the underlying real type; coefficients are always `Complex{T}`.
`N` is the number of array dimensions (1 for `HarmonicFunction`, 4 for `DiskFunction`).
"""
abstract type AbstractDiskFunction{T<:Real, N} <: AbstractArray{Complex{T}, N} end

# ─── Shared helpers ───────────────────────────────────────────────────────────

"""
    _m_to_idx(m, n)

Convert azimuthal frequency `m` to a 1-based storage index in an FFT-ordered
coefficient vector of length `n` (ordering: [0, 1, …, Mθ, -Mθ, …, -1]).
"""
@inline _m_to_idx(m::Int, n::Int) = m ≥ 0 ? m + 1 : n + m + 1

"""
    _ij_to_idx(i, j)

Map derivative order `(i, j)` to a linear slot index in `[1, 6]`.
Throws `BoundsError` if `i < 0`, `j < 0`, or `i + j > 2`.

Slot mapping:
  1 ↔ (0,0),  2 ↔ (1,0),  3 ↔ (0,1),  4 ↔ (2,0),  5 ↔ (1,1),  6 ↔ (0,2)
"""
function _ij_to_idx(i::Int, j::Int)
    (i < 0 || j < 0 || i + j > 2) &&
        throw(BoundsError("invalid derivative order (i=$i, j=$j): need i≥0, j≥0, i+j≤2"))
    i == 0 && j == 0 && return 1
    i == 1 && j == 0 && return 2
    i == 0 && j == 1 && return 3
    i == 2 && j == 0 && return 4
    i == 1 && j == 1 && return 5
    return 6   # (0, 2)
end

"""Inverse of `_ij_to_idx`: maps slot index 1–6 back to `(i, j)`."""
const _IDX_TO_IJ = NTuple{2,Int}[(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]
