# [Non-circular domains](@id method-domains)

A `Domain` is the image of the unit disk under a map ``z = f(\zeta)``. Functions on it are sampled at the mapped grid
points `Ω.z = f.(D.z)`, and the operators are pulled back to the disk:

```math
\mathcal{S}_\Omega[u](f(\zeta)) = \frac{1}{4\pi} \int_D \frac{u(f(\eta))}{|f(\zeta) - f(\eta)|}\, J(\eta)\, d\eta,
```

with ``J`` the Jacobian of the map. The singularity at ``\eta = \zeta`` comes from the factor ``1/|\zeta - \eta|``,
and the rest of the kernel is smooth. Keeping the disk's PSH for the singular factor makes the operators on
``\Omega`` as accurate as on the disk.

The weight is still the disk's, `Ω.w` ``= \sqrt{1 - |\zeta|^2}``. Because the maps are smooth up to the boundary,
it vanishes like the square root of the distance to ``\partial\Omega``, which is the edge behavior of the
densities on ``\Omega``. Densities on a `Domain` are therefore the same forms as on the disk (``g/w``, ``w g``).

## Ellipses

`ellipse(a, b, Mℓ, Mₘ)` uses the affine map ``f(\zeta) = a\,\mathrm{Re}\,\zeta + i\, b\,\mathrm{Im}\,\zeta``. Then
``|f(\zeta) - f(\eta)| = \rho(\phi)\, |\zeta - \eta|`` exactly, with ``\phi`` the angle of ``\zeta - \eta`` and
``\rho(\phi) = \sqrt{a^2\cos^2\phi + b^2\sin^2\phi}``. The single layer on the ellipse is the angular kernel
``ab/\rho(\phi)`` on the disk, and the Stokes single layer is three angular kernels, the entries of

```math
A_0(\phi) = \frac{ab}{\rho(\phi)}\left(\mathbf{I} + \hat\gamma\hat\gamma^T\right), \qquad \hat\gamma = \frac{(a\cos\phi,\, b\sin\phi)}{\rho(\phi)}.
```

All are sparse in coefficient space (see [Angular kernels](@ref method-stokes)). They have more diagonals than
the disk's, since ``1/\rho(\phi)`` has infinitely many Fourier modes. The number kept is set by their decay to
1e-14, and grows as ``b/a`` decreases. Building an ellipse is cheap: about 3 ms and 1.3 MiB at ``M = 16``, and
0.02 s and 5 MiB at ``M = 24``.

The ellipse also stores the Fourier coefficients of ``A_0`` (`Ω.ext`, an `EllipseGeometry`), which the
[close evaluation](@ref method-suspensions) between bodies uses.

## Conformally mapped domains

!!! warning "Experimental"
    Conformally mapped domains are experimental: their accuracy and interface may change. The limits below (the
    origin must be inside the curve, slower convergence for elongated shapes, dense operators, no close evaluation
    between bodies) are known, and not yet addressed. For ellipses, use `ellipse`.

`domain(γ, Mℓ, Mₘ)` maps the disk conformally onto the inside of a closed curve. `γ` is the curve sampled at
equispaced parameter values, counterclockwise. The map ``f`` is computed from the samples in three steps:

1. Solve an interior Dirichlet problem for a harmonic function with boundary values ``-\log|\gamma|``, by a
   double-layer integral equation on the boundary.
2. Take its harmonic conjugate, by a Cauchy integral.
3. Evaluate ``f`` and ``f'`` anywhere inside by barycentric Cauchy integrals.

This normalizes ``f(0) = 0``, so **the origin must lie inside the curve**. If it does not, `domain` returns without
an error but its operators are wrong: a unit circle centered at ``3`` gave a drag of ``-3 \times 10^{-4}``
instead of ``16/3``. The curve is upsampled by a factor of 5 by default, so it must be smooth and resolved by its
samples.

With ``M(\zeta, \eta) = (f(\zeta) - f(\eta))/(\zeta - \eta)``, which is smooth and nonzero with ``M(\zeta, \zeta) =
f'(\zeta)``, the single layer kernel on ``\Omega`` is ``1/(|M|\,|\zeta - \eta|)``. The package forms it as the
disk's operator, the matrix ``Q\,\hat K\,P`` of transform, eigenvalues and inverse transform on the grid, multiplied
entrywise by the smooth factor ``J(\eta)/|M(\zeta, \eta)|``. This is singularity swapping: the disk's operator
absorbs the singularity, and the smooth factor is applied pointwise. The hypersingular operator and the Stokes
blocks are built the same way. In the Stokes blocks the direction ``\hat r \hat r^T`` on ``\Omega`` has the angle
``\psi = \phi + \alpha``, with ``\phi`` the angle on the disk and ``\alpha`` that of ``M``. So it comes from the
disk operators of the kernels ``e^{\pm 2i\phi}/|\zeta - \eta|``, times ``e^{\pm 2i\alpha}``: for example
``\cos^2\psi = 1/2 + (e^{2i\alpha} e^{2i\phi} + e^{-2i\alpha} e^{-2i\phi})/4``.

The operators are then projected to coefficient space, as dense matrices. The transforms are not formed as matrices:
each is an FFT in ``\theta`` and, for each order ``m``, a radial sum, ``P = \operatorname{blockdiag}(R_m)\,(F
\otimes I)`` and ``Q = (F^* \otimes I)\,\operatorname{blockdiag}(Y_m)``. A disk operator ``\hat K`` that shifts the
order by ``\kappa`` (``\kappa = 0`` for the single layer and the hypersingular operator, ``\pm 2`` for the kernels
``e^{\pm 2i\phi}/|\zeta - \eta|``) gives ``Q\,\hat K\,P`` as a phase ``e^{i\kappa\theta}`` times a matrix circulant
in ``\theta``: a table of ``M_\ell^2 M_m`` numbers. The grid operators are not formed either. Each column of them
(one source point) is computed from these tables and the smooth factors, transformed by an FFT and projected
radially. The rows of the result are then transformed the same way. This takes ``O(N^2 M)`` time, and the memory
of the coefficient matrices:

| `M` | grid points ``N`` | `domain` | `ellipse` | `disk` |
|:--|:--|:--|:--|:--|
| 8 | 153 | 0.01 s, 0.3 MiB | 0.7 ms, 0.1 MiB | 0.2 ms |
| 16 | 561 | 0.02 s, 2.5 MiB | 3.4 ms, 1.3 MiB | 0.8 ms |
| 24 | 1225 | 0.05 s, 10 MiB | 9 ms, 5.3 MiB | 1.6 ms |
| 32 | 2145 | 0.12 s, 28 MiB | 0.02 s, 15 MiB | 2.9 ms |

(Laptop, 8 threads.) The grid operators `Ω.K_S`, `Ω.K_N` and `Ω.K_G` are formed when accessed, as dense
``N \times N`` matrices. A conformal domain keeps the LU factorization of each of its operators, made
by the first solve with it, so later solves are triangular solves. A domain whose shape changes must be built again,
with its factorizations; a moving rigid body need not, as its operators are those of its own frame. Use `ellipse` for
ellipses: it gives the same results far more cheaply. Close evaluation between bodies is available for disks and
ellipses only.
