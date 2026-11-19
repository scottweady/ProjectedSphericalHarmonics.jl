# Examples

The scripts in `examples/` check the package against exact solutions. They have their own environment: from the
package root, run `julia --project=examples -e 'using Pkg; Pkg.instantiate()'` once, then
`julia --project=examples examples/<name>.jl`. The outputs below are from a laptop, each script taking 8–20 s, most
of it compilation.

## Eigenfunctions of the single layer (`eigenfunctions.jl`)

Applies `𝒮` to ``y_5^3/w`` and `𝒮⁻¹` to ``y_5^3`` on `disk(64, 16)`, and compares with the eigenvalue relation
``\mathcal{S}[y_\ell^m/w] = (\lambda_\ell^m/4)\, y_\ell^m``:

```
Max error in 𝒮 for (l,m) = (5,3): 4.4e-16
Max error in 𝒮⁻¹ for (l,m) = (5,3): 4.2e-9
```

The inverse's error is that of ``y/w`` near the rim, where ``w`` is small: dividing by ``w`` amplifies rounding.

## The Poisson problem (`poisson.jl`)

Solves ``\Delta u = -1`` in the disk with ``u = 0`` on the rim, by `Δ⁻¹(-1, 0, D)` on `disk(64, 16)`, against
``u = (1 - r^2)/4``:

```
Max error in Δ⁻¹ for f = -1, g = 0: 6.0e-15
```

## Drag and torque on a disk (`stokes.jl`)

The force on a disk translating at unit speed, and the torque on one rotating at unit rate, under a stress-free
surface (the default viscosities), on `disk(128, 64)`. Both are ``16/3`` exactly:

```
force on a translating disk: 5.333333333336814
torque on a rotating disk: 5.333333333338558
```

## An ellipse, two ways (`ellipse.jl`)

The charge density of an ellipse at unit potential, and the drag on an ellipse translating along its major axis, for
``a = 1.5``, ``b = 1``, at ``M = 32``. The ellipse is discretized twice: by `ellipse` (the affine map) and by
`domain` from 127 samples of its boundary (the conformal map). The references are the closed forms
``\sigma\sqrt{1 - x^2/a^2 - y^2/b^2} = 2/(b\,K(1 - b^2/a^2))`` and the drag ``4\pi/A`` with ``A`` from complete
elliptic integrals:

```
Using angle discretization
Error in single layer potential: 8.8e-11
Error in drag force on an ellipse: 9.4e-14
Using conformal discretization
Error in single layer potential: 6.6e-5
Error in drag force on an ellipse: 6.1e-9
```

The affine map is exact, so `ellipse` is limited only by the expansion. The conformal map is computed numerically
from the boundary samples, which limits `domain`. The pointwise density, measured against the ``1/w`` form, is the
most sensitive to the map's accuracy near the boundary.

## A second-kind equation by GMRES (`concentration.jl`)

Solves ``\sigma + 2\beta\, \mathcal{S}[\sigma] = 2\beta`` in coefficient space with the package's `gmres` wrapper,
for ``\beta = 0.1, 1, 10, 100`` on `disk(128, 64)`. As ``\beta \to \infty`` the solution tends to that of
``\mathcal{S}[\sigma] = 1``, ``\sigma_0 = 4/(\pi w)``. The script prints the difference at the center:

```
β = 0.1,  |σ(0) - σ₀(0)| = 1.09    (GMRES: 3 iterations)
β = 1,    |σ(0) - σ₀(0)| = 0.314   (5 iterations)
β = 10,   |σ(0) - σ₀(0)| = 0.0879  (8 iterations)
β = 100,  |σ(0) - σ₀(0)| = 0.0218  (17 iterations)
```

The difference falls roughly like ``1/\beta``, while the iteration count grows with ``\beta``, as the operator
approaches the first-kind ``\mathcal{S}``.

## Stability of a growing droplet (`microbial_droplet.jl`)

A linear stability calculation that combines most of the Laplace operators. A droplet grows from a uniform
concentration with buoyant forcing (``\mathrm{Ra} = \beta = 1``). Its base state is built from:
- the volume potentials `𝒱` and `ℬ`;
- `𝒩⁻¹` for the pressure;
- `𝒮` for the stream function;
- `∂n` for the normal velocity of the rim.

The perturbation for each azimuthal mode ``m`` comes from the shape derivatives `δ𝒩`, `δ𝒮`, `δ𝒱` and `δℬ`. On
`disk(32)` the script prints the growth rate ``\sigma_m`` for ``m = 0, \dots, 8``. For ``m \ge 2`` these agree to
2–5e-6 with the closed form ``\sigma_b = \frac{\beta\,\mathrm{Ra}}{96}\left(\frac{m\,\lambda_1^0\lambda_m^m}{4} -
\frac{9}{2(m - 1)(2m + 1)}\right)``, which the script computes but does not print (``-0.0015625`` at ``m = 2``,
``0.0064174`` at ``m = 3``). The rates for ``m = 0`` and ``1``, about 1e-6, are zero up to the same error.

```
(m, σₘ) = (2, -0.001560448350202179)
(m, σₘ) = (3, 0.006420022296074067)
(m, σₘ) = (4, 0.009660193515409412)
…
(m, σₘ) = (8, 0.015975572486573913)
```
