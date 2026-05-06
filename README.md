# ProjectedSphericalHarmonics.jl

A lightweight Julia library for solving integral equations on flat open surfaces using projected spherical harmonics.

Currently the code supports:
- Laplace single layer and hypersingular operators
- Stokes single layer operator with mobility solver
- Bilaplace single layer
- Differential operators (grad, div, lap)
- Non-circular domains via conformal mapping singularity swapping

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