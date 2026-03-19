# CLAUDE.md — DiskFunction

> **Do not read files in this folder unless explicitly asked to work on DiskFunction or HarmonicFunction.**
> Key public API and usage patterns are summarised in the top-level `CLAUDE.md` under "Key data structures".

This folder is intended to eventually become its own package.

---

## Files

| File | Contents |
|------|----------|
| `AbstractDiskFunction.jl` | Abstract supertype `AbstractDiskFunction{T,N}` and shared index helpers |
| `HarmonicFunctions.jl` | `HarmonicFunction` struct: harmonic functions `Δh=0`, derivative operators |
| `DiskFunction.jl` | `DiskFunction` struct: Poisson solution `Δu=f`, lazy derivative slots |

---

## AbstractDiskFunction

`AbstractDiskFunction{T<:Real, N} <: AbstractArray{Complex{T}, N}`

Shared helpers defined here:

| Helper | Purpose |
|--------|---------|
| `_m_to_idx(m, n)` | FFT-ordered frequency `m` → 1-based storage index |
| `_ij_to_idx(i, j)` | Derivative order `(i,j)` → slot index `1–6` |
| `_IDX_TO_IJ` | Inverse map: slot index → `(i,j)` |

Slot mapping: `1↔(0,0)`, `2↔(1,0)`, `3↔(0,1)`, `4↔(2,0)`, `5↔(1,1)`, `6↔(0,2)`.

---

## HarmonicFunction

`HarmonicFunction{T}` represents a harmonic function `h` (`Δh = 0`) on the disk.

Stores `û::Vector{ComplexF64}` in FFT order (length `2Mθ+1`, same layout as `D.Mspan`).
Normalization: `û[i]` is the coefficient at frequency `Mspan[i]` such that
`h(1, θ) = Σᵢ û[i] * ylm(|m|, m, 1.0) * e^{imθ}`.

### Constructors

```julia
HarmonicFunction(g, D)                          # g = boundary values (length Nθ)
HarmonicFunction(û, D; from_coefficients=true)  # û already in PSH normalization
```

### Coefficient-space derivatives

`∂ζ` shifts `m → m-1`; `∂ζ̄` shifts `m → m+1`.

```julia
∂ζ_HarmonicFunction!(dû, û, Mspan)
∂ζ̄_HarmonicFunction!(dû, û, Mspan)
```

### Grid-space derivatives

```julia
∂ζ(h::HarmonicFunction, D)    # nodal values of ∂ζh
∂ζ̄(h::HarmonicFunction, D)   # nodal values of ∂ζ̄h
```

### Adding harmonic derivative into a TriangularCoeffArray

The harmonic derivative contributes to the **first row** (`j=1`) of each frequency column:

```julia
∂ζĥ = zeros(ComplexF64, length(Mspan))
∂ζ_HarmonicFunction!(∂ζĥ, ĥ_vec, Mspan)
for (i, m) in enumerate(Mspan)
    size_current_m(lmax, m) > 0 || continue
    mode_coefficients(∂ζu_tri, m)[1] += ∂ζĥ[i]
end
```

---

## DiskFunction

`DiskFunction{T}` is the solution to `Δu = f` on the unit disk with homogeneous
Dirichlet boundary conditions, stored in PSH coefficient space.

Indexed as `df[l, m, i, j]` where `(i, j)` selects the derivative `∂ζⁱ∂ζ̄ʲu`.
Only even-parity modes (`(l+m) % 2 == 0`, `abs(m) ≤ l`) are valid.
Derivative slots beyond `(0,0)` and `(1,1)` are **lazily populated**.

### Constructors

```julia
DiskFunction(f, D)                                    # from nodal density
DiskFunction(f̂_tri, D; derivatives=(), is_real=false) # from TriangularCoeffArray
DiskFunction!(df, f̂_tri; derivatives=())              # in-place (canonical)
```

`derivatives` is an iterable of `(i, j)` pairs: valid extra slots are
`(1,0)`, `(0,1)`, `(2,0)`, `(0,2)`.

### Derivative slots

| k | (i,j) | meaning      | populated by default |
|---|-------|--------------|----------------------|
| 1 | (0,0) | u = Δ⁻¹f     | yes |
| 2 | (1,0) | ∂ζu          | only if in `derivatives=` |
| 3 | (0,1) | ∂ζ̄u          | only if in `derivatives=` |
| 4 | (2,0) | ∂ζ²u         | only if in `derivatives=` |
| 5 | (1,1) | ∂ζ∂ζ̄u = f/4 | yes |
| 6 | (0,2) | ∂ζ̄²u        | only if in `derivatives=` |

### Evaluating derivatives

```julia
evaluate(df, i, j, D)      # nodal values of ∂ζⁱ∂ζ̄ʲu on D.r
evaluate(df, i, j, D, r)   # at custom radial points r
evaluate(df, D)             # shorthand for (i,j) = (0,0)
```

Grid-space derivatives via the operator approach (no pre-stored slot required):

```julia
∂ζ(df, D)    # nodal ∂ζu
∂ζ̄(df, D)   # nodal ∂ζ̄u
```

### Arithmetic with HarmonicFunction

```julia
add!(df, h)   # df += h  (in-place; updates all populated slots)
sub!(df, h)   # df -= h  (in-place)
df + h        # out-of-place
df - h
```

Since `Δh = 0`, the `(1,1)` slot of `df` is unaffected by `add!`/`sub!`.
