# CLAUDE.md — EllipticOperators

## API layers

There are three levels of abstraction. **Prefer the highest level that fits the task.**

```
DiskFunction                  ← high-level: see src/DiskFunction/CLAUDE.md
    ↓ built on
Triangular operators  ← mid-level: use this level unless especifically asked for
    ↓ built on
Per-frequency kernels (Ĝᵐ!, ∂Ĝᵐ∂ζ!, …)   ← low-level: use only inside per-frequency loops
```

**Avoid:** `FullFhatOperators/` (dense-matrix path) and `PerFrequencyOperators/apply_dense.jl` — these are legacy and should not be used in new code.

---


## Mid-level API: Helmholtz workflow + triangular operators

Use this when `DiskFunction` is insufficient — e.g. time-stepping (where `(α−Δ)u=f` must be solved repeatedly) or operators that need direct access to the Poisson density.

### Helmholtz solve `(α−Δ)u = f`, Dirichlet BC `u|∂D = g`

```julia
lmax  = D.Mr
Mspan = vec(Array(D.Mspan))

# Precompute once per (lmax, α):
Problem_matrices = [helmholtz_matrix(lmax, m, α) for m in Mspan]

# Per solve — assemble RHS in coefficient space:
f̂_tri = psh_triangular(rhs_nodal, D)
ĝ     = fft(g) / length(g)    # zeros(Nθ) for homogeneous BC

Solution_vector = [Problem_matrices[i] \ [ĝ[i] ; mode_coefficients(f̂_tri, Mspan[i])]
                   for i in eachindex(Mspan)]
# Solution_vector[i][1]      = harmonic coefficient b_m
# Solution_vector[i][2:end]  = interior density ρ_m  (Poisson density for particular solution)

# Reconstruct PSH coefficients of u:
û_tri = TriangularCoeffArray{Float64}(lmax, Mspan)
for (i, m) in enumerate(Mspan)
    res = mode_coefficients(û_tri, m)
    Ĝᵐ!(res, Solution_vector[i][2:end], lmax, m)
    res[1] += Solution_vector[i][1]    # add harmonic coefficient
end
ipsh!(u_nodal, û_tri, D)
```

### Exact derivatives from a Helmholtz step

After solving, store `ρ_m = Solution_vector[i][2:end]` and `b_m = Solution_vector[i][1]`.
At the next step, compute `∂ζu` **exactly** from the stored density — no grid differentiation:

```julia
ρ̂_tri = TriangularCoeffArray{Float64}(lmax, Mspan)
for (i, m) in enumerate(Mspan)
    mode_coefficients(ρ̂_tri, m) .= Solution_vector[i][2:end]
end

∂ζu_tri = ∂Ĝ∂ζ(ρ̂_tri)   # exact ∂ζ(Δ⁻¹ρ); see FhatOperatorsTriangularArrays/CLAUDE.md

# Add harmonic derivative (see src/DiskFunction/CLAUDE.md for pattern):
ĥ_vec = [Solution_vector[i][1] for i in eachindex(Mspan)]
∂ζĥ   = zeros(ComplexF64, length(Mspan))
∂ζ_HarmonicFunction!(∂ζĥ, ĥ_vec, Mspan)
for (i, m) in enumerate(Mspan)
    size_current_m(lmax, m) > 0 || continue
    mode_coefficients(∂ζu_tri, m)[1] += ∂ζĥ[i]
end
```

### Triangular operators

See `FhatOperatorsTriangularArrays/CLAUDE.md` for the full table of `Ĝ!`, `∂Ĝ∂ζ!`, `∂Ĝ∂ζ̄!`, etc.

---

## Low-level API: per-frequency kernels

Use only when implementing a custom per-frequency loop. Prefer the triangular wrappers above otherwise.

```julia
Ĝᵐ!(res, f̂ᵐ, lmax, m)          # Δ⁻¹ at frequency m
∂Ĝᵐ∂ζ!(res, f̂ᵐ, lmax, m)       # ∂ζ(Δ⁻¹f) at frequency m; output at freq m-1
∂Ĝᵐ∂ζ̄!(res, f̂ᵐ, lmax, m)      # ∂ζ̄(Δ⁻¹f) at frequency m; output at freq m+1
```

Indexing utilities:
```julia
size_current_m(lmax, m)          # number of even-parity modes at frequency m
∂ζ_indexing_sparse(lmax, m)      # output size for ∂ζ operator at frequency m
∂ζ̄_indexing_sparse(lmax, m)     # output size for ∂ζ̄ operator
```

---

## Internal / do not use directly

- `FullFhatOperators/` — legacy dense-matrix wrappers; superseded by triangular operators
- `PerFrequencyOperators/apply_dense.jl` — legacy dense path
- `inverse_laplacian_matrix_sparse`, `r_dot_∇Δ⁻¹_matrix_sparse`, `ζ∂ζΔ⁻¹_matrix_sparse` — raw matrix builders used internally to construct `helmholtz_matrix`; do not call directly
- `Inverse_laplacian_coef_m`, `Inverse_laplacian` — old API, superseded by `DiskFunction`
