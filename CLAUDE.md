# CLAUDE.md — ProjectedSphericalHarmonics.jl

## Project context

Research code for projected spherical harmonics on the disk, with the goal of becoming a usable public library. There is one co-contributor in addition to the primary author.

---

## Code conventions

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

- Always use Unicode names (∂ζ, Δ⁻¹, f̂ᵐ, ζ̄, etc.) — never fall back to ASCII equivalents.
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

## Testing

### Running all tests

```bash
julia --project tests/runtests.jl
```

### Running tests for a specific module

```bash
julia --project tests/per_frequency_operators.jl
julia --project tests/full_fhat_operators.jl
julia --project tests/boundary_operators.jl
```

### Test file organization (Option A — by source module)

| Test file | Covers |
|-----------|--------|
| `tests/per_frequency_operators.jl` | `src/EllipticSolverCoefficientOperator/PerFrequencyOperators/` |
| `tests/full_fhat_operators.jl` | `src/EllipticSolverCoefficientOperator/FullFhatOperators/` |
| `tests/boundary_operators.jl` | `src/EllipticSolverCoefficientOperator/BoundaryConditionOperators/` |

Each test file must be runnable as a standalone script (not only via `runtests.jl`).

### Test requirements

- Every operator must be tested at **at least two discretization sizes**: one with `MR` odd and one with `MR` even.
- Numerical comparisons use `rtol = 1e-12`.
- Tests are written per frequency (not as full-matrix tests only).

### Legacy test files

The files `testing_inverse_laplace_*.jl`, `Inverse_laplacian_coef.jl`, `operators.jl`, `null_space.jl`, and `testing_indexing_functions.jl` are to be retired once their content is absorbed into the proper test files above. Do not add new tests to them.
