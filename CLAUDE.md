# CLAUDE.md — ProjectedSphericalHarmonics.jl

## Project context

Research code for projected spherical harmonics on the disk, with the goal of becoming a usable public library. There is one co-contributor in addition to the primary author.

---

## Code conventions

## To know

- When possible, use triangular arrays and the sparse representation of the operators.

### In-place vs out-of-place

- The **in-place** (`!`) version is always the canonical implementation.
- The out-of-place version allocates `res` and delegates to the in-place version — it should never contain independent logic.

### Argument order

Per-frequency operators must follow this canonical order:

```julia
operator(f̂ᵐ, lmax, m)          # out-of-place
operator!(res, f̂ᵐ, lmax, m)    # in-place
```

If you encounter existing functions with inconsistent argument order (e.g. `(f̂ᵐ, m, lmax)`), fix them.

### Naming

- Struct follow the julia style, Pascal case
- functions are all lower case, with underscores (snake case)
- Prefer to not use unicode for internal function or struct naming. Unicode for names should be reserved for public API's
- Prefer unicode for mathematical symbols (∂ζ, Δ⁻¹, f̂ᵐ, ζ̄, etc.) — do not fall back to ASCII equivalents.
- Internal helper names may use shorter forms (e.g. `_check_inplace_length`) but must still prefer Unicode for mathematical symbols.

### Negative frequencies

The default pattern for `m < 0` is conjugate symmetry: conjugate the input, delegate to the positive-`m` version, then conjugate the result. However, some operators require a distinct implementation for `m < 0` — do not blindly apply the conjugate pattern without verifying it is mathematically valid for the specific operator.

### Documentation

Most functions should have a Julia-standard docstring:

```julia
"""
    function_name(args)

One-line description.

# Arguments
- `arg` : description

# Returns
- description
"""
```

Undocumented functions are acceptable only for small internal helpers prefixed with `_`.

---

## Codebase architecture

Detailed documentation lives in subdirectory `CLAUDE.md` files. This section is a quick-reference summary only.

### `disk` struct fields

`D.Mr` (= lmax), `D.Mθ`, `D.Mspan` (FFT-ordered: `[0,1,...,Mθ,-Mθ,...,-1]`), `D.r`, `D.θ`, `D.ζ` (grid points), `D.dw` (quadrature weights).

### Key data structures

| Struct | What it represents | Deep docs |
|--------|--------------------|-----------|
| `TriangularCoeffArray` | PSH coefficients in sparse triangular layout; even-parity modes only | `src/TriangularCoeffArray/CLAUDE.md` |
| `HarmonicFunction` | Harmonic function `Δh=0`; stores per-frequency coefficients `û[i]` | `src/DiskFunction/CLAUDE.md` |
| `DiskFunction` | Solution to `Δu=f` with zero Dirichlet BC; lazy derivative slots `(i,j)` | `src/DiskFunction/CLAUDE.md` |

### Key patterns

- **Transforms**: `psh_triangular(u, D)` → `TriangularCoeffArray`; `ipsh!(u, û_tri, D)` → nodal. See `src/TriangularCoeffArray/CLAUDE.md`.
- **Operators** (`Ĝ`, `∂Ĝ∂ζ`, `∂Ĝ∂ζ̄`, etc.): act on the Poisson **density** `f̂` to compute derivatives of `Δ⁻¹f`. See `src/EllipticOperators/FhatOperatorsTriangularArrays/CLAUDE.md`.
- **Helmholtz solve** `(α−Δ)u=f`: per-frequency, precompute `helmholtz_matrix(lmax, m, α)`. Full workflow in `src/EllipticOperators/CLAUDE.md`.
- **Exact derivatives** from a Helmholtz step: apply `∂Ĝ∂ζ` to the stored density `ρ_m = Solution_vector[i][2:end]`. Add harmonic correction at row 1 of each column. See `src/EllipticOperators/CLAUDE.md`.

---

## Testing

### Running all tests

```bash
julia --project tests/runtests.jl
```

### Running tests for a specific module

```bash
julia --project tests/EllipticOperators/per_frequency_operators.jl
julia --project tests/EllipticOperators/full_fhat_operators.jl
julia --project tests/EllipticOperators/full_fhat_operators_triangular.jl
julia --project tests/EllipticOperators/boundary_operators.jl
julia --project tests/TriangularArrays/runtests.jl
```

### Test file organization (by source module)

```
tests/
├── runtests.jl
├── EllipticOperators/
│   ├── per_frequency_operators.jl          — PerFrequencyOperators/ + in-place tests
│   ├── full_fhat_operators.jl              — FullFhatOperators/ (dense-matrix wrappers)
│   ├── full_fhat_operators_triangular.jl   — FhatOperatorsTriangularArrays/
│   └── boundary_operators.jl              — BoundaryConditionOperators/
├── Functions/
│   ├── disk_function_test.jl
│   ├── derivatives_test.jl
│   └── harmonic_function_test.jl
├── TriangularArrays/                       — TriangularCoeffArray/ + transforms
│   └── runtests.jl  (+ sub-test files)
└── legacy/                                 — retired files; do not add new tests here
    ├── operators.jl
    ├── testing_indexing_functions.jl
    └── CheckingRecurions/
```

Each test file must be runnable as a standalone script (not only via `runtests.jl`).

### Test requirements

- Every operator must be tested at **at least two discretization sizes**: one with `MR` odd and one with `MR` even.
- Numerical comparisons use `rtol = 1e-12`.
- Tests are written per frequency (not as full-matrix tests only).

### Legacy test files

All legacy files have been moved to `tests/legacy/`. Do not add new tests there.
