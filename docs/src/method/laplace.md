# [Laplace operators](@id method-laplace)

## Single layer and hypersingular operators

On a flat surface ``\Omega`` in the plane ``z = 0``, the single layer potential of the 3D Laplacian and its
hypersingular operator (the normal derivative of the double layer potential) are

```math
\mathcal{S}[u](x) = \frac{1}{4\pi} \int_\Omega \frac{u(y)}{|x - y|}\, dy, \qquad
\mathcal{N}[u](x) = \frac{1}{4\pi}\, \mathrm{f.p.}\!\int_\Omega \frac{u(y)}{|x - y|^3}\, dy,
```

where ``\mathrm{f.p.}`` denotes Hadamard's finite part. On the unit disk, the PSH of each parity are their
eigenfunctions:

```math
\mathcal{S}\left[\frac{y_\ell^m}{w}\right] = \frac{\lambda_\ell^m}{4}\, y_\ell^m \quad (\ell + m \text{ even}), \qquad
\mathcal{N}\left[y_\ell^m\right] = -\frac{y_\ell^m}{\lambda_\ell^m\, w} \quad (\ell + m \text{ odd}),
\qquad
\lambda_\ell^m = \frac{\Gamma\!\left(\frac{\ell + m + 1}{2}\right)\Gamma\!\left(\frac{\ell - m + 1}{2}\right)}
                      {\Gamma\!\left(\frac{\ell + m + 2}{2}\right)\Gamma\!\left(\frac{\ell - m + 2}{2}\right)}
```

(`λlm(ℓ, m)`). Each operator is applied in four steps:

| | Step 1 | Step 2 | Step 3 | Step 4 |
|:--|:--|:--|:--|:--|
| `𝒮(u, D)` | multiply by ``w`` | even `psh` | multiply by ``\lambda_\ell^m/4`` | even `ipsh` |
| `𝒮⁻¹(f, D)` | even `psh` | divide by ``\lambda_\ell^m/4`` | even `ipsh` | divide by ``w`` |
| `𝒩(u, D)` | odd `psh` | multiply by ``-1/\lambda_\ell^m`` | odd `ipsh` | divide by ``w`` |
| `𝒩⁻¹(f, D)` | multiply by ``w`` | odd `psh` | divide by ``-1/\lambda_\ell^m`` | odd `ipsh` |

The densities carry their edge behavior through the basis. `𝒮` takes a density ``g/w`` and `𝒮⁻¹` returns one;
`𝒩` takes a density ``w g`` and `𝒩⁻¹` returns one; ``g`` is smooth in every case. For example, `𝒮⁻¹(1, D)` is
``4/(\pi w)``, the charge density of a disk at unit potential, whose total charge is 8. The relations above hold to
rounding for every mode in the retained range. For general data, the error is the truncation error of ``g``'s PSH
series, which decreases exponentially when ``g`` is smooth.

On an ellipse or a conformally mapped domain, the same operators are matrices in coefficient space instead of
diagonals; see [Non-circular domains](@ref method-domains).

## The Poisson problem

On a flat surface the hypersingular operator is the surface Laplacian of the single layer, ``\mathcal{N} = \Delta
\mathcal{S}``. So ``u_p = \mathcal{S}\mathcal{N}^{-1} f`` satisfies ``\Delta u_p = f``. In even PSH coefficients
``\mathcal{S}\mathcal{N}^{-1}`` is a sparse matrix that couples each ``\ell`` only to ``\ell \pm 2`` at the same
order (`𝒮𝒩⁻¹`). `Δ⁻¹(f, g, D)` solves ``\Delta u = f`` in the disk with ``u = g`` on the rim. It adds to ``u_p`` the
harmonic function ``\sum_m c_m r^{|m|} e^{im\theta}``, whose Fourier coefficients ``c_m`` are those of ``g -
u_p|_{r=1}``. For ``f = -1``, ``g = 0`` it reproduces ``(1 - r^2)/4`` to 4e-15 at `disk(32, 16)`.

## Volume potentials

Two volume potentials on the disk are computed by dense direct sums over the grid, at a cost of ``O(N^2)`` in the
number ``N`` of grid points:

- `𝒱(u, D)`, the 2D Laplace potential ``\frac{1}{2\pi}\int_D \log|x - y|\, u(y)\, dy`` (so ``\Delta\mathcal{V}[u]
  = u``), with its log singularity subtracted. `𝒱(1, D)` is exactly ``(r^2 - 1)/4``.
- `ℬ(u, D; κ)`, the 2D bilaplacian potential ``\frac{1}{8\pi}\int_D |x - y|^2 (\log|x - y| - 1 + \kappa)\, u(y)\,
  dy`` (so ``\Delta^2 \mathcal{B}[u] = u``). Its kernel is continuous, so the plain grid quadrature is used.

Both converge algebraically, not spectrally. `𝒯(u, D)`, the single layer of the 3D bilaplacian (kernel
``|x - y|/8\pi``), is spectral. It uses ``|x - y| = |x - y|^2/|x - y|`` and expands ``|x - y|^2 = |x|^2 - x\bar y -
\bar x y + |y|^2`` in complex notation, which gives four applications of `𝒮`:

```math
\mathcal{T}[u] = \tfrac{1}{2}\left(|\zeta|^2 \mathcal{S}[u] - \zeta\, \mathcal{S}[\bar\zeta u] - \bar\zeta\,
\mathcal{S}[\zeta u] + \mathcal{S}[|\zeta|^2 u]\right).
```

## Shape derivatives

`δ𝒮(u, m, D)`, `δ𝒩(u, m, D)`, `δ𝒱(u, m, D)` and `δℬ(u, m, D)` differentiate the operators with respect to a
deformation of the disk. Each one is expressed by the disk's own operator applied to ``\zeta^k u``. The
deformation is the conformal map ``\zeta \mapsto \zeta + \varepsilon\, \zeta^{m+1}``, and the operator on the
deformed domain is pulled back to the disk's grid. For a real ``\varepsilon``,

```math
\frac{d}{d\varepsilon}\Big|_{\varepsilon = 0} \mathcal{S}_\varepsilon[u] =
\tfrac{1}{2}\left(\delta\mathcal{S}(u, m) + \overline{\delta\mathcal{S}(\bar u, m)}\right),
```

and likewise for the others. That is, `δ𝒮` is the holomorphic part of the derivative. For a real density it is
``\mathrm{Re}\,\delta\mathcal{S}(u, m)``. For `δ𝒮` and `δ𝒩` this was checked against central differences of the
operators on `domain` with ``\varepsilon = \pm 10^{-4}`` (agreement to 1e-7 for ``m = 1, 2, 3``).
