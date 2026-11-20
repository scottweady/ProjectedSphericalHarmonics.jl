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

## Mapped domains

!!! warning "Experimental"
    Mapped domains (`domain`) are experimental: their accuracy and interface may change. Their operators are
    dense, and there is no close evaluation between bodies. For ellipses, use `ellipse`.

`domain(γ, Mℓ, Mₘ; map)` discretizes the inside of a closed curve, the image of the disk under a smooth map ``F``
with ``F(e^{it}) = \gamma(t)``. `γ` is the curve sampled at equispaced parameter values, counterclockwise. By
default (`map = :auto`) the map is the harmonic map below, which is the cheapest, or the Winslow map if the harmonic
map folds, with a message; `map = :harmonic`, `:winslow` or `:conformal` chooses one.

**The harmonic map** (`map = :harmonic`) is the harmonic extension of the samples: with ``\gamma(t) =
\sum_n \hat\gamma_n e^{int}``,

```math
F(\zeta) = \sum_{n \ge 0} \hat\gamma_n \zeta^n + \sum_{n \ge 1} \hat\gamma_{-n} \bar\zeta^n.
```

The parametrization of `γ` places the grid: the disk's boundary point ``e^{it}`` goes to ``\gamma(t)``, and the
grid follows. For ``\gamma = a\cos t + i b \sin t`` the map is affine, and `domain` gives the operators of
`ellipse(a, b, M)`. By the Radó–Kneser–Choquet theorem the map is one-to-one when the curve is convex. A
non-convex curve may make it fold; `domain` checks the Jacobian and throws an `ArgumentError` if it does.

The map is not conformal, so the distance on ``\Omega`` depends on the direction as well as the distance on the
disk. With the Wirtinger divided differences ``F(\zeta) - F(\eta) = P\,(\zeta - \eta) + Q\,\overline{(\zeta -
\eta)}``, both smooth, and ``\mu = Q/P``,

```math
\frac{1}{|F(\zeta) - F(\eta)|} = \frac{1}{|P|}\,\frac{1}{|1 + \mu e^{-2i\phi}|}\,\frac{1}{|\zeta - \eta|},
```

with ``\phi`` the angle of ``\zeta - \eta``. The angular factor is a Fourier series in ``e^{2ik\phi}``, with
coefficients ``\mu^{|k|}`` or ``\bar\mu^{|k|}`` times functions of ``|\mu|^2``, interpolated once per domain. So
the single layer is a sum over ``k`` of smooth coefficients times the disk operators of the kernels
``e^{2ik\phi}/|\zeta - \eta|``. The Stokes kernels expand the same way. The conformal map is the case ``\mu = 0``,
one term; the affine map has constant ``\mu``. The operators of a harmonic domain cost more to build than those of
a conformal one, as each pair of grid points sums ``2M_m + 1`` terms. There is no hypersingular operator `𝒩`, as
for `ellipse`.

**The Winslow map** (`map = :winslow`) is the map whose *inverse* ``\zeta(z)`` is harmonic on ``\Omega``, with the same
boundary values. The disk is convex, so by the same theorem the inverse, and so the map, is one-to-one for any
smooth simply connected domain: it does not fold on the non-convex shapes on which the harmonic map does, such as
stars with deep valleys. It is the elliptic grid generator of Winslow. The real and imaginary parts of
``\zeta(z)`` solve Dirichlet problems on ``\Omega``, by a second-kind double-layer equation on the boundary, and are
evaluated by the barycentric Cauchy formula, accurate up to the boundary. ``F`` at the nodes of a PSH grid of degree
``L`` follows by Newton's method, and is expanded in the even PSH (polynomials in ``x, y``, a well-conditioned basis),
with ``L`` raised until the boundary is reproduced to ``10^{-8}``. The divided differences are then the mean values
of the derivatives along the segment from ``\eta`` to ``\zeta``,

```math
P = \int_0^1 \partial F(\eta + t(\zeta - \eta))\,dt, \qquad Q = \int_0^1 \bar\partial F(\eta + t(\zeta - \eta))\,dt,
```

computed exactly by Gauss–Legendre quadrature. (For the harmonic map these are the divided differences above.) A
split of the polynomial term by term also satisfies the identity, but its ``|\mu|`` reaches far beyond 1 for
distant pairs on non-convex shapes; the mean values keep ``|\mu| < 1``: 0.54 for the star ``1 + 0.6\cos 3t``, 0.75
for the pinwheel below. The construction costs more than the harmonic map's, as the mean values are sums along
every segment: about 4 s at ``M = 16`` and 30 s at ``M = 32`` (workstation, 16 threads), once for a rigid body.

How deformed a shape can be is limited by the spectral discretization, not by the map: the operators need
``|\mu| < 1`` for every pair of grid points, and converge more slowly as ``|\mu|`` approaches 1 and the Jacobian of
the map becomes small. A chiral pinwheel with five arms, ``z = r(s)\,e^{i(s + 0.6(r(s) - 1))}``, ``r = 1 + 0.3\cos 5s``
(``|\mu| \le 0.75``), converges to six digits in its capacitance, drag and torque at ``M = 32``:

| `M` | capacitance | drag | torque |
|:--|:--|:--|:--|
| 16 | 8.761115 | 5.877040 | 7.902250 |
| 24 | 8.761112 | 5.876909 | 7.902228 |
| 32 | 8.761113 | 5.876909 | 7.902223 |

The parametrization still places the grid, and must stay smooth. Resampling a curve at equal arc length gives its
high-curvature parts (tips, lobes) only a small share of the parameter, so the curve needs hundreds of Fourier modes
(271 for a 4:1 ellipse, which ``a\cos t + i b\sin t`` gives with one) and the grid cannot resolve the map.

**The conformal map** (`map = :conformal`) has the same grid on every shape: it ignores the parametrization. It
crowds the grid on elongated shapes: the map of a 4:1 ellipse stretches by a factor of 300 more at the tips than
at the sides, and only 4 of the 561 nodes at ``M = 16`` lie within the outer quarters of the ellipse. Its errors
there are tens of percent at any affordable ``M``:

| ellipse ``a/b`` | conformal, capacitance error at ``M = 32`` | harmonic |
|:--|:--|:--|
| 1.5 | 2e-10 | 1e-14 |
| 2 | 1.5e-5 | 1e-14 |
| 4 | 0.37 | 1e-14 |

### The conformal map

`domain(γ, Mℓ, Mₘ; map = :conformal)` computes the conformal map ``f`` from the samples in three steps:

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
of the coefficient matrices. The harmonic map is built the same way, column by column, with its sum over ``k``
in place of the single shifts:

| `M` | grid points ``N`` | `domain` (harmonic) | `domain` (conformal) | `ellipse` | `disk` |
|:--|:--|:--|:--|:--|:--|
| 8 | 153 | 0.006 s | 0.01 s, 0.3 MiB | 0.7 ms, 0.1 MiB | 0.2 ms |
| 16 | 561 | 0.065 s | 0.02 s, 2.5 MiB | 3.4 ms, 1.3 MiB | 0.8 ms |
| 24 | 1225 | 0.25 s | 0.05 s, 10 MiB | 9 ms, 5.3 MiB | 1.6 ms |
| 32 | 2145 | 1.1 s | 0.12 s, 28 MiB | 0.02 s, 15 MiB | 2.9 ms |

(Laptop, 8 threads; the harmonic times were measured under load.) The harmonic map keeps the tables of the disk
operators of ``e^{2ik\phi}/|\zeta - \eta|`` for each grid size, so its first call at a given `M` costs more
(0.2 s more at ``M = 32``); the coefficient matrices take the same memory as the conformal map's. The grid operators
`Ω.K_S`, `Ω.K_N` and `Ω.K_G` are formed when accessed, as dense ``N \times N`` matrices. A mapped domain keeps the
LU factorization of each of its operators, made by the first solve with it, so later solves are triangular solves.
A domain whose shape changes must be built again, with its factorizations; a moving rigid body need not, as its
operators are those of its own frame. Use `ellipse` for
ellipses: it gives the same results far more cheaply. Close evaluation between bodies is available for disks and
ellipses only.
