# CLAUDE.md — FhatOperatorsTriangularArrays

## Purpose

Triangular-array wrappers for the per-frequency sparse operators in
`PerFrequencyOperators/apply_sparse.jl`. Each function applies its
corresponding `Xᵐ` variant across all frequencies stored in a
`TriangularCoeffArray`, dispatching to the per-frequency in-place operator
for each `m`.

Naming convention: drop the supra-index (Ĝᵐ → Ĝ, ∂Ĝᵐ∂ζ → ∂Ĝ∂ζ, etc.).

---

## Signature convention

```julia
operator!(res::TriangularCoeffArray, f̂::TriangularCoeffArray)   # in-place
operator(f̂::TriangularCoeffArray)                               # out-of-place
```

- **No explicit `lmax` argument.** It is read from `f̂.lmax` (via
  `_lmax(f̂)` in `array_interface.jl`).
- The out-of-place version allocates `res = zero(f̂)` and delegates to
  the in-place version. It must never contain independent logic.

---

## Same-frequency operators

These operators output at the **same azimuthal frequency** as the input.
`res` must have the same Mspan and per-frequency column sizes as `f̂`.
`similar(f̂)` is always a valid allocation for `res`.

| Triangular wrapper | Per-frequency kernel |
|--------------------|----------------------|
| `Ĝ`                | `Ĝᵐ`                 |
| `ζ_∂Ĝ∂ζ`          | `ζ_∂Ĝᵐ∂ζ`            |
| `ζ̄_∂Ĝ∂ζ̄`          | `ζ̄_∂Ĝᵐ∂ζ̄`            |
| `∂²Ĝ∂ζ∂ζ̄`         | `∂²Ĝᵐ∂ζ∂ζ̄`           |
| `r_∂Ĝ∂r`           | `r_∂Ĝᵐ∂r`            |

---

## Frequency-shifting operators

These operators output at a **shifted azimuthal frequency**. Nevertheless,
`res` has the a**same Mspan** as `f̂`; the in-place version scatters output
into the column of `res` at the shifted frequency via
`mode_coefficients(res, m ± shift)`.

`similar(f̂)` is always a valid allocation for `res`.

| Triangular wrapper | Per-frequency kernel | Shift |
|--------------------|----------------------|-------|
| `∂Ĝ∂ζ`            | `∂Ĝᵐ∂ζ`              | −1    |
| `∂Ĝ∂ζ̄`            | `∂Ĝᵐ∂ζ̄`              | +1    |
| `∂²Ĝ∂ζ²`          | `∂²Ĝᵐ∂ζ²`            | −2    |
| `∂²Ĝ∂ζ̄²`          | `∂²Ĝᵐ∂ζ̄²`            | +2    |

**Reading shifted output:** after calling `op(f̂)` or `op!(res, f̂)`, the
result for input frequency `m` lives at `mode_coefficients(result, m ± shift)`,
not at `result.data[i]`.

**Boundary guard:** the in-place loop skips frequency `m` when
`size_current_m(lmax, m ± shift) == 0`, which occurs exactly at the boundary
of the Mspan where the shifted frequency would be empty. The corresponding
column of `res` is left at its initial value.
