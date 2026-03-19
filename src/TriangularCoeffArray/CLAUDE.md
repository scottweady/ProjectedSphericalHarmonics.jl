# CLAUDE.md — TriangularCoeffArray

## Struct

`TriangularCoeffArray{T,N,P,O}` stores PSH coefficients in the sparse triangular layout. Only O parity modes (`(l+m)` O, `l ≥ |m|`) are kept.

Key fields: `lmax::Int`, `Mspan::Vector{Int}` (FFT-ordered: `[0,1,...,Mθ,-Mθ,...,-1]`), `data::Vector{Vector{Complex{T}}}`.

## Accessing coefficients

```julia
mode_coefficients(arr, m)          # vector of length size_current_m(lmax, m)
size_current_m(lmax, m)            # = floor((lmax + 2 - |m|) / 2)
```

Index `j` in `mode_coefficients(arr, m)` → degree `l = |m| + 2*(j-1)`.
Row 1 (`j=1`) is the lowest degree `l = |m|` — the harmonic mode for that frequency.

## Construction

```julia
TriangularCoeffArray{Float64}(lmax, Mspan)                    # zero-filled, even parity
TriangularCoeffArray{Float64}(lmax, Mspan; parity=:odd)
zero(arr)    # same structure, zero-filled
similar(arr) # same structure, uninitialized
copy(arr)    # deep copy
```

## Transforms

```julia
psh!( û_tri, u, D) # nodal matrix → TriangularCoeffArray (forward PSH) (in place)
psh_triangular(u, D)              # nodal matrix → TriangularCoeffArray (forward PSH)
ipsh!(u, û_tri, D)                # TriangularCoeffArray → nodal matrix (in-place)
NodalToTriangularArray(u, D)      # alias for psh_triangular
TriangularArrayToPSH(û_tri, D)    # → dense PSH matrix (for old ipsh(psh_matrix, D) API)
```

## Arithmetic

`+`, `-`, `*`, `/` are defined on `TriangularCoeffArray`. `lmul!(α, arr)` scales in-place.

## Operators on TriangularCoeffArray

All operators are in `FhatOperatorsTriangularArrays/` — see that subdirectory's `CLAUDE.md` for the full table. They operate on the **density** of the inverse Laplacian (i.e., given `f̂`, computes derivatives of `Gf`, where `Gf` is a particular solution of  `Δ(Gf) = f`).

Key note: frequency-shifting operators (`∂Ĝ∂ζ`, `∂Ĝ∂ζ̄`, etc.) scatter output into the column at the shifted frequency. Read result via `mode_coefficients(res, m ± shift)`, **not** `res.data[i]`.
