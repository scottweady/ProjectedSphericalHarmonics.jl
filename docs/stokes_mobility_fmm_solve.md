# `stokes_mobility_solve_suspension` — Frame Convention and Data Flow

## Frame Convention

All internal computation uses **body-frame** coordinates and velocities.
World-frame quantities are rotated in/out at the boundaries.

| Quantity | Frame |
|---|---|
| `D.z`, `D.dz` | Body (reference geometry, unrotated) |
| `f[i]` (force density) | Body |
| `u_buf[i]` (velocity buffer) | Body |
| GMRES unknown `U[i][1:2]` (translation) | Body |
| GMRES unknown `U[i][3]` (rotation `ω`) | Frame-independent |
| `uinf`, `uslip` passed in | World |
| `F` (applied force/torque) passed in | World |
| `U_sol` returned | World |

World-frame position of body `i` at point `z_body`: `xcm[i] + exp(iθcm[i]) * z_body`

---

## Top-level: `stokes_mobility_solve_suspension(xcm, θcm, uinf, uslip, F, D)`

```
INPUTS (world frame)
  xcm[i], θcm[i]    — position and orientation of body i
  uinf(z)           — background flow (function of world-frame position)
  uslip             — slip velocity (function or nothing)
  F[i] = [Fx, Fy, T] — applied force/torque in world frame

  ┌─────────────────────────────────────────┐
  │ 1. Evaluate uinf, uslip at world points │
  │    xcm[i] + exp(iθ) * D.z              │
  │    Rotate to body frame via exp(-iθ)    │
  └──────────────────┬──────────────────────┘
                     │  uinf_eval[i], uslip_eval[i]  (body frame)
  ┌──────────────────▼──────────────────────┐
  │ 2. Rotate F[i][1:2] to body frame       │
  │    F_body[i] = R(-θ) * [Fx; Fy]        │
  │    Torque F[i][3] unchanged             │
  └──────────────────┬──────────────────────┘
                     │  F_body[i]  (body frame)
  ┌──────────────────▼──────────────────────┐
  │ 3. Build RHS (body frame)               │
  │    rhs_field[i] = psh(uslip - uinf, D)  │
  │    rhs_force[i] = F_body[i]             │
  └──────────────────┬──────────────────────┘
                     │
  ┌──────────────────▼──────────────────────┐
  │ 4. GMRES: solve  A·x = rhs              │
  │    x = [f̃_1, f̃_2, ..., U_1, U_2, ...]  │
  │    (all body frame)                     │
  │    using matvec! and precond!           │
  └──────────────────┬──────────────────────┘
                     │  U_sol[i]  (body frame)
  ┌──────────────────▼──────────────────────┐
  │ 5. Rotate U_sol[i][1:2] to world frame  │
  │    U_world = R(θ) * U_body              │
  │    ω unchanged                          │
  └──────────────────┬──────────────────────┘
                     │
OUTPUTS
  f_sol[i]           — surface force density (body frame, at D.z points)
  U_sol[i]           — rigid body velocity [Ux, Uy, ω] (world frame)
```

---

## `matvec!(result, v)` — the GMRES operator

All quantities body-frame.

```
v = [f̃_1, ..., f̃_N, U_1, ..., U_N]   (body frame)

  ┌──────────────────────────────────────────────┐
  │ Unpack: f̃[i], U[i]                           │
  │ Recover: f[i] = ipsh(f̃[i], D) / D.w          │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ eval_suspension_velocity!(u_buf, ..., f, D) │
  │ → u_buf[i] = S·f evaluated at body i         │
  │   (body frame, see below)                    │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ Subtract rigid body velocity (body frame):   │
  │   u_buf[i][1] -= Ux - ω * yg                 │
  │   u_buf[i][2] -= Uy + ω * xg                 │
  │ where xg, yg = real/imag of D.z              │
  └──────────────┬───────────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────────┐
  │ Row 1: f̃_out[i] = psh(u_buf[i], D)           │
  │ Row 2: U_out[i] = [∫fx dA, ∫fy dA, ∫torque]  │
  └──────────────┬───────────────────────────────┘
                 │
result = pack(f̃_out, U_out)
```

---

## `eval_suspension_velocity!(u, sources, stoklets, xcm, θcm, f, D)`

Computes `u[i] = Σ_j S_{ij} f[j]` in body frame of each target `i`.

```
For each source body j:
  ┌─────────────────────────────────────────────────┐
  │ Self-interaction (j == i):                      │
  │   u[i] += 𝒮_st(f[i], D)        (body frame)    │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ Cross-body (j ≠ i) via FMM:                     │
  │                                                 │
  │ Source positions (world frame):                 │
  │   z_world = xcm[j] + exp(iθ[j]) * D.z          │
  │                                                 │
  │ Stoklet strengths (world frame):                │
  │   f_world = R(θ[j]) * f_body[j]                │
  │   stoklets = f_world * D.dz   (D.dz real scalar)│
  │                                                 │
  │ FMM call → world-frame velocities at all pts    │
  │                                                 │
  │ Subtract self-term from FMM output              │
  │   (𝒮_st already handles self-interaction)       │
  │                                                 │
  │ Rotate FMM output to body frame of target i:    │
  │   u_body = R(-θ[i]) * u_world                  │
  │   u[i] += u_body                               │
  └─────────────────────────────────────────────────┘
```

---

## `precond!(result, v)`

Block-diagonal approximation: treats each body independently using the
single-body solver `stokes_mobility_solve`. Operates entirely in body frame.

```
For each body i:
  rf[i]  = ipsh(f̃[i], D)           (residual velocity, body frame)
  rU[i]  = force/torque residual    (body frame)

  [δU, δω] = stokes_mobility_solve(-rf[i], rU[i][1:2], rU[i][3], D)
                                     (single-body, body frame)

  ũ[i] = rf[i] + A·δU              (body frame)
  δf[i] = 𝒮_st⁻¹(ũ[i], D)

  output: [psh(δf·w, D), δU, δω]
```