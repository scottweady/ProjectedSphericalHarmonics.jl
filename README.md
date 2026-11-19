# ProjectedSphericalHarmonics.jl

A Julia library of spectral solvers for integral equations on flat open surfaces using projected spherical harmonics, including Laplace, Stokes and biharmonic layer potentials, conformally mapped domains, and FMM-accelerated many-particle Stokes suspensions.

## Documentation

https://scottweady.github.io/ProjectedSphericalHarmonics.jl/ (getting started, the method, examples, the API
reference and a code map), built from `docs/` and published to the gh-pages branch on every push to main. To build
it locally: `julia --project=docs -e 'using Pkg; Pkg.instantiate()'` once, then `julia --project=docs docs/make.jl`,
and open `docs/build/index.html`.

## Features

Currently the code supports:
- Laplace single layer and hypersingular operators, and their inverses
- Stokes single layer operator with a single-body mobility solver, for in-plane forces and, in free space, normal forces (out of the plane)
- Bilaplace single layer
- Differential operators (grad, div, lap, ∂n, ∂ζ, ∂ζ̄) and a Poisson solver
- Non-circular domains: ellipses (affine map of the disk) and, experimentally, general shapes via conformal mapping singularity swapping
- Many-particle Stokes suspensions: mobility and resistance solves accelerated by Ewald + NUFFT (via [FINUFFT](https://github.com/flatironinstitute/finufft)) or FMM (via [FMM3D](https://github.com/flatironinstitute/FMM3D)), with spectrally accurate close evaluation between nearby disks and ellipses

## Mathematical Background

The projected spherical harmonics are defined as

```math
y_\ell^m(r, \theta) = N_\ell^m P_\ell^{|m|}(\sqrt{1 - r^2}) e^{im\theta},
```

where $P_\ell^m$ are the associated Legendre polynomials and

```math
N^m_\ell = \sqrt{ \frac{2\ell+1}{2\pi} \frac{(\ell - |m|)!}{(\ell + |m|)!} }
```

is a normalization constant. Functions of the same parity (even or odd $\ell + m$) are orthogonal with respect to the weighted inner product

```math
\langle u, v \rangle = \int_D \frac{u(\mathbf{x}) v^*(\mathbf{x}) }{\sqrt{1 - |\mathbf{x}|^2}} {\rm d}\mathbf{x}.
```

## Installation

This package is not registered in the Julia General registry and must be installed directly from GitHub. To install, open Julia and enter the package manager by pressing ], then run:

```julia
add https://github.com/scottweady/ProjectedSphericalHarmonics.jl.git
```

Alternatively, from the Julia REPL without entering pkg mode:

```
using Pkg
Pkg.add(url="https://github.com/scottweady/ProjectedSphericalHarmonics.jl")
```

## Example usage

The projected spherical harmonics are particularly useful when working with layer potentials of the 3D laplacian. The single layer potential $\mathcal{S}$ is defined as

```math
\mathcal{S}[u](\mathbf{x}) = \frac{1}{4\pi}\int_D \frac{u(\mathbf{y})}{|\mathbf{x} - \mathbf{y}|} {\rm d}\mathbf{y}.
```

In particular, the even projected spherical harmonics are eigenfunctions of $\mathcal{S}$,

```math
\mathcal{S}[y_\ell^m/w] = \frac{\lambda_\ell^m}{4} y_\ell^m \quad \ell + m \text{ even},
```

where $w(\mathbf{x}) = \sqrt{1-|\mathbf{x}|^2}$ is the weight function. This can be tested with the following code:

```julia
using ProjectedSphericalHarmonics

# Degree of PSH expansion
Mr, Mθ = 64, 16

# Discretize
D = disk(Mr, Mθ)
ζ = D.ζ #grid points
w = D.w #weight function

# Define a function
l, m = 5, 3
u = ylm(l, m, ζ)

# Evaluate the single layer potential and its inverse
err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4.0)
println("Max error in 𝒮 for (l,m) = ($l,$m): ", maximum(abs.(err)))

err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
println("Max error in 𝒮⁻¹ for (l,m) = ($l,$m): ", maximum(abs.(err))) 
```

The projected spherical harmonics are also useful for solving Poisson problems:

```julia
using ProjectedSphericalHarmonics

# Degree of PSH expansion
Mr, Mθ = 64, 16

# Discretize
D = disk(Mr, Mθ)
ζ = D.ζ #grid points

f = -1 # right-hand side
g = 0 # boundary condition
u = Δ⁻¹(f, g, D)
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(ζ))))
println("Max error in Δ⁻¹ for f = -1, g = 0: $err")
```

### Non-circular domains

The same operators work on ellipses and on domains given by a boundary curve. The curve is mapped conformally onto the disk. Conformally mapped domains are experimental: the curve must enclose the origin (not checked), convergence slows for elongated shapes, and their operators are dense.

```julia
using ProjectedSphericalHarmonics

M = 32

# Ellipse with semi-axes a = 1.5, b = 1
Ω = ellipse(1.5, 1.0, M)

# General domain from samples of its boundary curve
θ = range(0, 2π, 128)[1:end-1]
Ω = domain(1.5 .* cos.(θ) .+ im .* sin.(θ), M)

# Force density on a plate translating with unit velocity in x, and the total drag
f = 𝒮_st⁻¹((1.0, 0.0), Ω)
println("Drag: ", integral(f[1], Ω))
```

### Viscosity convention

The Stokes operators model in-plane flow driven by forces in the plane z = 0, which separates a fluid of viscosity `μ_bottom` below from one of viscosity `μ_top` above. The Green's function is

$$G(\mathbf{r}) = \frac{1}{4\pi(\mu_\mathrm{bottom} + \mu_\mathrm{top})}\left(\frac{\mathbf{I}}{|\mathbf{r}|} + \frac{\mathbf{r}\mathbf{r}^T}{|\mathbf{r}|^3}\right).$$

Every Stokes function takes the keywords `μ_bottom` (default `1.0`) and `μ_top` (default `0.0`):

| Setting | `μ_bottom` | `μ_top` |
|---|---|---|
| Half-space under a stress-free surface (default) | `μ` | `0` |
| Free space, viscosity μ | `μ` | `μ` |
| Interface, μ₁ below and μ₂ above | `μ₁` | `μ₂` |

A precomputed mobility matrix passed as `M` must use the same viscosities.

In free space a body can also be pushed out of the plane: pass 3-tuples `(fx, fy, fz)` and `(ux, uy, uz)` to `𝒮_st` and `𝒮_st⁻¹`, or `F = [Fx, Fy, Fz, Tx, Ty, Tz]` to `stokes_mobility_solve`, with the fluid viscosity as `μ`. The normal part is the Laplace single layer, `uz = 𝒮[fz]/(2μ)`, and decouples from the in-plane part. Calls with 2-tuples and `F = [Fx, Fy, T]` are unchanged.

### Stokes mobility

`stokes_mobility_solve` finds the rigid-body velocity and angular velocity of a single body, given a background flow, an optional slip velocity, and the applied force and torque `[Fx, Fy, T]`.

```julia
D = disk(32)
uinf(z) = (imag.(z), zero(real.(z)))   # shear flow ux = y

U, ω = stokes_mobility_solve(uinf(D.z), nothing, [0.0, 0.0, 0.0], D)
```

`suspension_mobility_solve` solves the N-body problem with GMRES, using the single-body solve as a preconditioner. Particle interactions are summed by Ewald splitting with [FINUFFT](https://github.com/flatironinstitute/finufft) (`method=:ewald`, the default), planned once per solve, or with [FMM3D](https://github.com/flatironinstitute/FMM3D) (`method=:fmm`); `eps` is the tolerance of either. Every body shares one discretization `Ω`, and each body is placed at a center `xcm[i]` with orientation `θcm[i]`. Because every body shares `Ω`, the fast sum's within-body terms are removed with one precomputed matrix, `suspension_self_matrix(Ω)`. The solves build it, along with the single-body mobility matrix `M`, when they aren't given; a time-stepping simulation can build both once and pass `S_self` and `M` at every step.

```julia
Ω   = ellipse(2.0, 0.5, 16)
xcm = [0.0 + 0.0im, 1.05 + 0.3im]   # side by side, edge gap 0.05
θcm = [π/2, -π/2]
F   = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]

f, U = suspension_mobility_solve(xcm, θcm, uinf, nothing, F, Ω; eps=1e-10, gmrestol=1e-10)
# U[i] = [Ux, Uy, ω] for body i (world frame); f[i] = (fx, fy) force density (body frame)
```

The node quadrature used by the FMM loses accuracy when a target point is closer to another body than the node spacing. For disks and ellipses, the cross term at such targets is replaced by a closed-form continuation of the single layer off the source body, which stays spectrally accurate at any separation. This is on by default. The keyword `close_radius` sets the distance at which it applies (see `CLOSE_RADIUS`), and `close_eval=false` turns it off. General conformally mapped domains always use node quadrature. `suspension_velocity_eval` (forces to velocities) and `suspension_velocity_solve` (velocities to forces) are also exported.

More complete scripts are in [`examples/`](examples/), which has its own `Project.toml`.

## Testing

```julia
using Pkg
Pkg.test("ProjectedSphericalHarmonics")
```

Each check prints its error next to its tolerance and records a `@test`. The suite takes about a minute.