# TriangularCoeffArray

A custom 1D array type for storing PSH (Projected Spherical Harmonic) coefficient
data in its natural triangular sparsity structure.

## Motivation

The PSH coefficient space for a function on the disk has a triangular structure:
for each azimuthal frequency `m`, only the radial degrees `l` satisfying
`l ≥ |m|` and `l + m` even are active. Storing the full `(Mr+1) × Nθ` matrix
wastes roughly half the memory and forces every operator to skip the odd-parity
entries explicitly. `TriangularCoeffArray` packs only the active coefficients,
one contiguous inner vector per frequency, while exposing a flat 1D interface so
that Krylov solvers (KrylovKit / GMRES) can treat it as an ordinary vector.

## Layout

```
Mspan = [0, 1, 2, -2, -1]          # FFT-ordered frequency list (Mθ = 2)

data[1]  →  û[l+m even, m= 0]
data[2]  →  û[l+m even, m= 1]
data[3]  →  û[l+m even, m= 2]
data[4]  →  û[l+m even, m=-2]
data[5]  →  û[l+m even, m=-1]
```

The flat 1D index runs through `data[1]`, then `data[2]`, etc.
Cumulative offsets (`_offsets`) allow O(log n) conversion between
a flat index and a `(column, row)` pair.

## Files

| File | Contents |
|------|----------|
| `struct.jl` | Struct definition, registry, constructors, `similar`/`copy`/`fill!` |
| `array_interface.jl` | `AbstractArray` interface, `column`, `ncolumns` |
| `display.jl` | `show` methods with terminal-aware truncation |
| `arithmetic.jl` | `+`/`-`/`*`/`/`, Krylov primitives, broadcasting |
| `conversions_triangular.jl` | `NodalToTriangularArray`, `TriangularArrayToPSH` |
