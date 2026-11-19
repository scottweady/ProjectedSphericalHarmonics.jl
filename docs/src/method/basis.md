# [The PSH basis](@id method-basis)

## Projected spherical harmonics

A point ``(x, y) = (r\cos\theta, r\sin\theta)`` of the unit disk is the projection of the point
``(x, y, \sqrt{1 - r^2})`` of the upper unit hemisphere. The projected spherical harmonics are the spherical
harmonics at that point:

```math
y_\ell^m(r, \theta) = N_\ell^m\, P_\ell^{|m|}(w)\, e^{im\theta}, \qquad w = \sqrt{1 - r^2}, \qquad
N_\ell^m = \sqrt{\frac{2\ell + 1}{2\pi}\, \frac{(\ell - |m|)!}{(\ell + |m|)!}},
```

for ``\ell \ge 0`` and ``|m| \le \ell`` (`ylm(ℓ, m, z)`, with `Nlm(ℓ, m)` the normalization). The hemisphere's
area element is ``dA/w``, so harmonics of the same parity of ``\ell + m`` are orthonormal in the weighted inner
product

```math
\langle u, v \rangle = \int_D \frac{u(x)\, \overline{v(x)}}{\sqrt{1 - |x|^2}}\, dx.
```

(Each parity is the restriction to the hemisphere of the harmonics that are even, or odd, under reflection
through the equator, so each is complete on the hemisphere by itself.)

**Parity and the edge.** ``P_\ell^{|m|}(w)`` is ``(1 - w^2)^{|m|/2} = r^{|m|}`` times a polynomial in ``w`` of
degree ``\ell - |m|``, which has only even powers when ``\ell + m`` is even and only odd powers otherwise. Since
``w^2 = 1 - r^2``:

- **Even** ``y_\ell^m`` (``\ell + m`` even) are polynomials in ``x`` and ``y`` of degree ``\ell``.
- **Odd** ``y_\ell^m`` are ``w`` times polynomials of degree ``\ell - 1``, so they vanish on the rim like ``w``.

A smooth function is an even expansion. A function that vanishes like ``w`` (a double-layer density) is an odd
expansion. A function singular like ``1/w`` (a single-layer density) is ``1/w`` times an even expansion. Each of
these converges as fast as the smooth part allows.

## The grid

`disk(Mℓ, Mₘ)` keeps the modes ``\ell \le M_\ell``, ``|m| \le M_m``. Its grid is the tensor product of

- ``M_\ell + 1`` Gauss–Legendre points in ``s = \sqrt{1 - r^2} \in (0, 1)``, so radii ``r = \sqrt{1 - s^2}``
  cluster toward the rim, and
- ``2M_m + 1`` equispaced angles ``\theta``.

Functions are matrices of values on this grid, `(Mℓ + 1) × (2Mₘ + 1)`, and the coefficient arrays have the same
shape: row ``\ell + 1``, and one column per order ``m``, in the order ``0, 1, \dots, M_m, -M_m, \dots, -1``
(`D.Mspan`). The masks `D.even` and `D.odd` mark the entries of each parity with ``|m| \le \ell``.

Since ``r\,dr/w = -ds``, the weighted inner product is ``\int_0^1 \int_0^{2\pi} u \bar v\, d\theta\, ds``, which
the grid integrates exactly for products of modes in the retained range. The area weights `D.dz` (``r\,dr\,d\theta
= w\, ds\, d\theta``) integrate smooth functions to the same order, and integrate ``g/w`` with ``g`` smooth as
accurately as ``g``.

## Transforms

`psh(u, D; parity)` computes the coefficients of one parity, and `ipsh(û, D; parity)` evaluates an expansion of
one parity on the grid. Each is an FFT in ``\theta``, followed, for each order ``m``, by a sum over ``\ell`` in
which ``y_\ell^m`` at the grid radii comes from the three-term recurrence of the associated Legendre functions in
``\ell`` (coefficients precomputed in `D.a`, `D.am1`). The cost is ``O(M_\ell^2 M_m)`` for the sums plus
``O(M_\ell M_m \log M_m)`` for the FFTs. A transform pair is exact for expansions in the retained modes:
`psh(ipsh(û, D), D)` returns `û` to rounding. Other routines built on the transforms:

- `upsample(u, D, Nr, Nθ)` evaluates the expansion of `u` on a finer grid.
- `trace(u, D)` evaluates it on the rim.
- `psh_matrix` and `ipsh_matrix` build the dense matrices of the transforms.
- `integral(u, Ω)` is `sum(u .* Ω.dz)`.

## Differentiation

With ``\partial_\zeta = (\partial_x - i\,\partial_y)/2``, ``w\,\partial_\zeta y_\ell^m`` is a multiple of the mode
``y_\ell^{m-1}`` (`∂ylm∂ζ`), which has the other parity. So `∂ζ` of an even expansion is computed in two steps in
coefficient space: shift each coefficient to order ``m - 1`` with its closed-form factor, which gives ``w\,
\partial_\zeta u`` as an odd expansion, then remove the factor ``w`` by solving with the matrix of multiplication
by ``w``. That solve is
ill-conditioned, so `∂ζ`, `∂ζ̄`, `grad`, `∂x`, `∂y`, `div` and `lap` lose digits as the resolution grows. They are
for smooth functions at moderate resolution, and their docstrings carry a warning. The normal derivative on the
rim, `∂n`, is a weighted sum of the even coefficients with closed-form weights and does not have this problem.
