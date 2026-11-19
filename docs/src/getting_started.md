# Getting started

## Installation

ProjectedSphericalHarmonics is not registered (Julia 1.12 or later; dependencies AssociatedLegendrePolynomials,
FastGaussQuadrature, FFTW, FINUFFT, FMM3D, IterativeSolvers, LinearOperators, SpecialFunctions, and the standard
libraries LinearAlgebra and SparseArrays). Add it by URL, or develop a local copy by path:

```julia
using Pkg
Pkg.add(url = "https://github.com/scottweady/ProjectedSphericalHarmonics.jl")
# or: Pkg.develop(path = "path/to/ProjectedSphericalHarmonics")
```

The single-body operators run on one thread. The suspension solver's fast sums and corrections are threaded, so
start Julia with threads for it (`julia -t 8`). The tests run from the package root with
`julia --project -e 'using Pkg; Pkg.test()'` (about a minute).

## A discretization

`disk(Mℓ, Mₘ)` discretizes the unit disk for degrees ``\ell \le M_\ell`` and orders ``|m| \le M_m`` (`disk(M)` is
`disk(M, M)`). Its grid is a tensor product of ``M_\ell + 1`` Gauss–Legendre points in ``s = \sqrt{1 - r^2}``,
which cluster toward the rim, and ``2M_m + 1`` equispaced angles. A function on the disk is a matrix of its values
at the grid points:

```julia
using ProjectedSphericalHarmonics

D = disk(32, 16)
size(D.z)                          # (33, 33): radii × angles
x, y = real.(D.z), imag.(D.z)      # D.z holds the points as complex numbers x + iy
w = D.w                            # the weight w = √(1 - r²), zero on the rim

u = x .^ 2 .+ y                    # a function, as values on the grid
integral(u, D)                     # ∫ u dA = π/4; `integral` is sum(u .* D.dz)
trace(u, D)                        # its values on the rim, at the grid's angles (1 × 33)
```

Functions are complex arrays inside the package; pass real arrays or numbers where convenient. A number stands for
a constant function. A vector field is a tuple of two arrays, `(ux, uy)`.

## Transforms

`psh` computes PSH coefficients and `ipsh` goes back to grid values. The coefficients have the shape of the grid:
entry `[ℓ + 1, j]` multiplies ``y_\ell^m`` with ``m`` = `D.Mspan[j]` (columns ordered ``0, 1, …, M_m, -M_m, …,
-1``). The basis splits by the parity of ``\ell + m``. Even PSH are polynomials in ``x`` and ``y``. Odd PSH are
``w`` times polynomials, so they vanish on the rim like ``w``. Each transform works with one parity:

```julia
û = psh(u, D)                      # even coefficients (the default parity)
ipsh(û, D) ≈ u                     # true: u is a polynomial, in the even span
v̂ = psh(w .* u, D; parity=:odd)    # odd coefficients of w·u
ipsh(v̂, D; parity=:odd) ≈ w .* u   # true

ylm(5, 3, D.z)                     # one basis function at the grid points
```

See [The PSH basis](@ref method-basis) for the grid, the transforms and their cost.

## Laplace layer potentials

The single layer potential of the 3D Laplacian over a flat surface ``\Omega`` is

```math
\mathcal{S}[u](x) = \frac{1}{4\pi} \int_\Omega \frac{u(y)}{|x - y|}\, dy, \qquad x \in \Omega.
```

`𝒮(u, D)` evaluates it on the disk, and `𝒮⁻¹(f, D)` solves ``\mathcal{S}[u] = f``. The density of a first-kind
equation is singular at the edge like ``1/w``, so `𝒮` expects densities of the form ``g/w`` with ``g`` smooth, and
`𝒮⁻¹` returns them. For example, the charge density of a disk held at unit potential is ``4/(\pi w)``:

```julia
D = disk(32)
σ = 𝒮⁻¹(1, D)
maximum(abs.(σ .* D.w .- 4/π))     # 5e-13
integral(σ, D)                     # 8, the total charge
𝒮(σ, D) ≈ ones(size(D.z))          # true
```

Even PSH are eigenfunctions of the single layer: ``\mathcal{S}[y_\ell^m/w] = (\lambda_\ell^m/4)\, y_\ell^m``,
with ``\lambda_\ell^m`` given by `λlm(ℓ, m)`. The hypersingular operator `𝒩` (the normal derivative of the double
layer) acts on densities that vanish like ``w`` at the edge, and odd PSH are its eigenfunctions. `𝒩⁻¹` inverts it.

The Poisson problem ``\Delta u = f`` in the disk with ``u = g`` on the rim is `Δ⁻¹(f, g, D)`:

```julia
D = disk(32, 16)
u = Δ⁻¹(-1, 0, D)                  # u = (1 - r²)/4
maximum(abs.(u .- (1 .- abs2.(D.z)) ./ 4))     # 4e-15
```

## Non-circular domains

The same operators work on an ellipse and on a domain bounded by a smooth curve. `ellipse(a, b, M)` maps the disk
affinely onto the ellipse with semi-axes `a` and `b`. `domain(γ, M)` maps it conformally onto the inside of a closed
curve, given by samples `γ` at equispaced parameter values, counterclockwise, with the origin inside the curve:

```julia
Ω = ellipse(1.5, 1.0, 32)

θ = range(0, 2π, 129)[1:end-1]
Ω2 = domain(1.5 .* cos.(θ) .+ im .* (sin.(θ) .+ 0.3 .* sin.(2θ)), 32)    # a smooth, nonconvex curve

σ = 𝒮⁻¹(1, Ω)                      # functions on a Domain are sampled at Ω.z
integral(σ, Ω)                     # the ellipse's capacitance, 4πa/K(1 - b²/a²) (K: complete elliptic integral)
```

!!! warning "Experimental"
    `domain` is experimental: its accuracy and interface may change. See
    [Conformally mapped domains](@ref method-domains) for what is known not to work yet.

`Ω.z` holds the mapped grid points and `Ω.dz` the area weights. The weight `Ω.w` is still the disk's
``\sqrt{1 - |\zeta|^2}``, which vanishes on the boundary of ``\Omega`` like the square root of the distance to it.
The ellipse's operators are sparse in coefficient space. The conformal domain's are dense, which limits it to
moderate `M`; see [Non-circular domains](@ref method-domains).

## Stokes flow

The Stokes operators model flow in the plane ``z = 0``, driven by forces in that plane. The plane separates a fluid
of viscosity `μ_bottom` below from one of viscosity `μ_top` above. With ``\mu = \mu_\mathrm{bottom} +
\mu_\mathrm{top}``, the velocity is the single layer

```math
\mathcal{S}_\mathrm{st}[f](x) = \int_\Omega G(x - y)\, f(y)\, dy, \qquad
G(\mathbf{r}) = \frac{1}{4\pi\mu}\left(\frac{\mathbf{I}}{|\mathbf{r}|} + \frac{\mathbf{r}\mathbf{r}^T}{|\mathbf{r}|^3}\right).
```

`𝒮_st(f, Ω)` and `𝒮_st⁻¹(u, Ω)` take and return tuples `(fx, fy)` and `(ux, uy)`. Every Stokes function takes the
keywords `μ_bottom` (default 1) and `μ_top` (default 0):

| Setting | `μ_bottom` | `μ_top` |
|:--|:--|:--|
| Half-space under a stress-free surface (default) | `μ` | `0` |
| Free space, viscosity μ | `μ` | `μ` |
| Interface, μ₁ below and μ₂ above | `μ₁` | `μ₂` |

The force density on a translating plate, and its drag:

```julia
D = disk(32)
f = 𝒮_st⁻¹((1.0, 0.0), D)
integral(f[1], D)                  # 16/3 in the default half-space; 32/3 with μ_top = 1
```

`stokes_mobility_solve(uinf, uslip, F, Ω)` solves for the rigid motion of one body. Given a background flow `uinf`
(a tuple of arrays on the grid), an optional slip velocity `uslip` (or `nothing`), and the applied force and torque
`F = [Fx, Fy, T]`, it returns the velocity `U` and the rotation rate `ω`:

```julia
uinf = (imag.(D.z), zero(real.(D.z)))          # shear flow, ux = y
U, ω = stokes_mobility_solve(uinf, nothing, [0.0, 0.0, 0.0], D)    # a free disk in shear: U = (0, 0), ω = -1/2
```

It builds the body's 3 × 3 mobility matrix (`stokes_mobility_matrix(Ω)`) each time unless you pass it as `M`.

## Suspensions

`suspension_mobility_solve` solves the mobility problem of many bodies at once. Every body is a copy of one
discretization `Ω`, placed at a center `xcm[i]` (a complex number) and rotated by `θcm[i]`:

```julia
Ω   = ellipse(1.5, 0.75, 16)
xcm = [0.0 + 0.0im, 3.2 + 0.4im, 1.0 + 2.6im]
θcm = [0.0, π/3, -π/4]
F   = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.5, 0.0]]     # [Fx, Fy, T] on each body, world frame
uinf(z) = (zero(real.(z)), zero(real.(z)))                    # no background flow

f, U = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, Ω; eps=1e-8, gmrestol=1e-8)
# U[i] = [Ux, Uy, ω] of body i (world frame); f[i] = (fx, fy), its force density (body frame)
```

`uinf` and `uslip` are functions of the world-frame position (a complex array) returning a tuple of arrays, or
vectors of precomputed tuples, one per body. The interactions between bodies are summed by Ewald splitting with
NUFFTs (`method = :ewald`, the default) or by FMM3D (`method = :fmm`), to tolerance `eps`. At points closer to
another disk or ellipse than a few grid spacings, the sum over grid nodes is replaced by a closed form
(`close_eval = true`, the default). Bodies must not overlap.

In a time-stepping simulation the body shape doesn't change, so build what depends only on `Ω` once and pass it at
every step:

```julia
M      = stokes_mobility_matrix(Ω)            # single-body mobility matrix (same viscosities as the solve)
S_self = suspension_self_matrix(Ω)            # each body's own terms in the fast sum

for step in 1:nsteps
    _, U = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, Ω; M=M, S_self=S_self)
    # … move xcm, θcm with U …
end
```

`suspension_velocity_eval` maps force densities to velocities, and `suspension_velocity_solve` solves for the force
densities that produce given velocities. See [Suspensions](@ref method-suspensions) for the solver, its costs and
its options.
