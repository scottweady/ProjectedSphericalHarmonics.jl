# EllipticOperators

Coefficient-space elliptic operators for the projected spherical harmonics (PSH) basis on the disk. Implements the inverse Laplacian, Helmholtz solver, and all first- and second-order derivatives of `Δ⁻¹f` as linear maps on PSH coefficient arrays.

---

## API layers

```
DiskFunction                        ← high-level: Poisson solve + lazy derivative slots
    ↓ built on
Triangular operators (this folder)  ← mid-level: time-stepping, custom solves
    ↓ built on
Per-frequency kernels               ← low-level: inner loops only
```

**Use the highest layer that fits your task.**
Do not reach into the per-frequency kernels unless you are writing a custom frequency loop.

---

## Folder structure

| Path | Contents |
|------|----------|
| `CoefficientSpaceOperators.jl` | Top-level include; exports `helmholtz_matrix`, `traceĜ`, `neumann_traceĜ`, and re-exports everything below |
| `IndexingFunctions.jl` | `size_current_m`, `∂ζ_indexing_sparse`, `∂ζ̄_indexing_sparse`, … |
| `FhatOperatorsTriangularArrays/` | `TriangularCoeffArray`-level wrappers: `Ĝ`, `∂Ĝ∂ζ`, `∂Ĝ∂ζ̄`, `∂²Ĝ∂ζ²`, … |
| `PerFrequencyOperators/` | Per-frequency in-place kernels: `Ĝᵐ!`, `∂Ĝᵐ∂ζ!`, … |
| `BoundaryConditionOperators/` | Trace and Neumann-trace operators for boundary data |
| `FullFhatOperators/` | **Legacy.** Dense-matrix path — do not use in new code |

---

## Mid-level: Helmholtz workflow

Use this level when `DiskFunction` is not sufficient, e.g. time-stepping where the same `(α−Δ)` system is solved many times.

### Precompute the problem matrices (once per `lmax` and `α`)

```julia
lmax  = D.Mr
Mspan = vec(Array(D.Mspan))
Problem_matrices = [helmholtz_matrix(lmax, m, α) for m in Mspan]
```

### Per-step solve

```julia
f̂_tri = psh_triangular(rhs_nodal, D)
ĝ     = fft(g) / length(g)          # boundary data; zeros for homogeneous Dirichlet

Solution_vector = [Problem_matrices[i] \ [ĝ[i]; mode_coefficients(f̂_tri, Mspan[i])]
                   for i in eachindex(Mspan)]
# Solution_vector[i][1]      → harmonic coefficient b_m
# Solution_vector[i][2:end]  → Poisson density ρ_m
```

### Reconstruct nodal values

```julia
û_tri = TriangularCoeffArray{Float64}(lmax, Mspan)
for (i, m) in enumerate(Mspan)
    res = mode_coefficients(û_tri, m)
    Ĝᵐ!(res, Solution_vector[i][2:end], lmax, m)
    res[1] += Solution_vector[i][1]
end
ipsh!(u_nodal, û_tri, D)
```

### Exact derivatives from a stored density

After the solve, store `ρ_m = Solution_vector[i][2:end]` and `b_m = Solution_vector[i][1]`. Derivatives of `u` are then exact (no grid differentiation):

```julia
ρ̂_tri = TriangularCoeffArray{Float64}(lmax, Mspan)
for (i, m) in enumerate(Mspan)
    mode_coefficients(ρ̂_tri, m) .= Solution_vector[i][2:end]
end

∂ζu_tri = ∂Ĝ∂ζ(ρ̂_tri)     # ∂ζ(Δ⁻¹ρ); see FhatOperatorsTriangularArrays/

# Add harmonic correction at row 1 of each column (see src/DiskFunction/):
ĥ_vec = [Solution_vector[i][1] for i in eachindex(Mspan)]
∂ζĥ   = zeros(ComplexF64, length(Mspan))
∂ζ_HarmonicFunction!(∂ζĥ, ĥ_vec, Mspan)
for (i, m) in enumerate(Mspan)
    size_current_m(lmax, m) > 0 || continue
    mode_coefficients(∂ζu_tri, m)[1] += ∂ζĥ[i]
end
```

---

## Mid-level: triangular operators

All operators act on `TriangularCoeffArray`. See `FhatOperatorsTriangularArrays/` for the full reference.

**Same-frequency** (output column = input column):

| Operator | Meaning |
|----------|---------|
| `Ĝ(f̂)` | `Δ⁻¹f` |
| `ζ_∂Ĝ∂ζ(f̂)` | `ζ ∂ζ(Δ⁻¹f)` |
| `ζ̄_∂Ĝ∂ζ̄(f̂)` | `ζ̄ ∂ζ̄(Δ⁻¹f)` |
| `∂²Ĝ∂ζ∂ζ̄(f̂)` | `∂ζ∂ζ̄(Δ⁻¹f) = f/4` |
| `r_∂Ĝ∂r(f̂)` | `r ∂r(Δ⁻¹f)` |

**Frequency-shifting** (output scattered to column at shifted frequency):

| Operator | Shift | Meaning |
|----------|-------|---------|
| `∂Ĝ∂ζ(f̂)` | −1 | `∂ζ(Δ⁻¹f)` |
| `∂Ĝ∂ζ̄(f̂)` | +1 | `∂ζ̄(Δ⁻¹f)` |
| `∂²Ĝ∂ζ²(f̂)` | −2 | `∂ζ²(Δ⁻¹f)` |
| `∂²Ĝ∂ζ̄²(f̂)` | +2 | `∂ζ̄²(Δ⁻¹f)` |

All operators follow the convention: in-place `op!(res, f̂)`, out-of-place `op(f̂)`. No explicit `lmax` argument — it is read from `f̂.lmax`.

---

## Low-level: per-frequency kernels

Use only inside a custom per-frequency loop.

```julia
Ĝᵐ!(res, f̂ᵐ, lmax, m)          # Δ⁻¹f at frequency m
∂Ĝᵐ∂ζ!(res, f̂ᵐ, lmax, m)       # ∂ζ(Δ⁻¹f); output at frequency m−1
∂Ĝᵐ∂ζ̄!(res, f̂ᵐ, lmax, m)      # ∂ζ̄(Δ⁻¹f); output at frequency m+1
```

Indexing helpers:

```julia
size_current_m(lmax, m)          # number of even-parity modes at frequency m
∂ζ_indexing_sparse(lmax, m)      # output length for ∂ζ operator at frequency m
∂ζ̄_indexing_sparse(lmax, m)     # output length for ∂ζ̄ operator
```

---

## Do not use directly

- `FullFhatOperators/` — legacy dense-matrix path, superseded by triangular operators
- `PerFrequencyOperators/apply_dense.jl` — legacy dense path
- `inverse_laplacian_matrix_sparse`, `r_dot_∇Δ⁻¹_matrix_sparse`, `ζ∂ζΔ⁻¹_matrix_sparse` — internal matrix builders used only inside `helmholtz_matrix`
- `Inverse_laplacian_coef_m`, `Inverse_laplacian` — old API, superseded by `DiskFunction`
