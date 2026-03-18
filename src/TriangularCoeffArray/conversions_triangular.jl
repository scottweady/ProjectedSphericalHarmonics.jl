# conversions_triangular.jl
# Interop between TriangularCoeffArray and the rest of the library:
# forward transform from nodal values (NodalToTriangularArray) and
# reconstruction of the full PSH coefficient matrix (TriangularArrayToPSH).

# ─── Nodal conversion ────────────────────────────────────────────────────────

"""
    NodalToTriangularArray(u, D; parity=:even)

Transform nodal values `u` on the disk grid into a `TriangularCoeffArray` by
computing the PSH transform and retaining only the coefficients of the
specified `parity` for each azimuthal frequency.

# Arguments
- `u`      : nodal values on the disk grid (AbstractMatrix, size `(Nr, Nθ)`)
- `D`      : disk discretization
- `parity` : `:even` (default) stores `l+m` even modes; `:odd` stores `l+m` odd modes

# Returns
- `TriangularCoeffArray` with `ordering=:fft` whose column for frequency `m`
  holds the radial coefficients of the requested parity
"""
#TODO: change to grid_to_triang
function NodalToTriangularArray(u::AbstractMatrix, D::disk; parity::Symbol = :even)
    T     = Float64
    û     = psh(u, D; parity = parity)
    Mspan = vec(Array(D.Mspan))
    if parity == :even
        data = [Vector{Complex{T}}(û[D.even[:,i], i]) for i in eachindex(Mspan)]
    else
        data = [Vector{Complex{T}}(û[D.odd[:,i], i]) for i in eachindex(Mspan)]
    end
    return TriangularCoeffArray(Mspan, data; parity = parity, ordering = :fft)
end

"""
    TriangularArrayToPSH(û_tri, D)

Convert a `TriangularCoeffArray` back into the full PSH coefficient matrix
expected by `ipsh`.

Each column `i` of the output matrix corresponds to frequency `D.Mspan[i]`,
and the rows of the appropriate parity (selected by `D.even[:,i]` or its
complement) are filled from `mode_coefficients(û_tri, D.Mspan[i])`; all
other entries remain zero.

# Arguments
- `û_tri` : `TriangularCoeffArray` holding per-parity coefficients per frequency
- `D`     : disk discretization

# Returns
- PSH coefficient matrix of size `(D.Mr + 1, length(D.Mspan))`
"""
#TODO: change to triangular_to_square
function TriangularArrayToPSH(û_tri::TriangularCoeffArray, D::disk)
    P     = parity(û_tri)
    Mspan = vec(Array(D.Mspan))
    û_psh = zeros(ComplexF64, D.Mr + 1, length(Mspan))
    for (i, m) in enumerate(Mspan)
        if P == :even
            û_psh[D.even[:,i], i] .= mode_coefficients(û_tri, m)
        else
            û_psh[D.odd[:,i], i] .= mode_coefficients(û_tri, m)
        end
    end
    return û_psh
end
