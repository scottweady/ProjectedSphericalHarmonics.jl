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

`𝒱(u, D)`, the 2D Laplace potential ``\frac{1}{2\pi}\int_D \log|x - y|\, u(y)\, dy`` (so ``\Delta\mathcal{V}[u]
= u``), and `𝒯_vol(u, D)`, the potential ``\frac{1}{8\pi}\int_D |x - y|\, u(y)\, dy``, take a smooth density
and are computed by product integration. The density is expanded in even PSH, ``u = \sum_j \hat u_j y_j``, and
the integrals of the kernel against each ``y_j`` are computed once per target radius: by rotational symmetry the
weights of a target ``r e^{i\alpha}`` are those of ``r`` times ``e^{i m_j \alpha}``. For the log kernel the Fourier
series ``\log|\rho e^{i\theta} - r| = \log\max(\rho, r) - \sum_{k \ge 1} \frac{q^k}{k} \cos k\theta``,
``q = \min(\rho, r)/\max(\rho, r)``, reduces each weight to radial integrals of polynomials (and of
``\rho\log\rho`` times a polynomial), done exactly by Gauss rules. For ``|x - y|`` the weights use polar
coordinates about the target, in which the integrand is a polynomial along each chord, with an angular rule
refined geometrically toward the rim. Both are spectrally accurate in ``M`` for smooth ``u`` and exact to
rounding for polynomial densities; `𝒱(1, D)` is ``(r^2 - 1)/4``. On the grid the operator is block diagonal in
``m`` (built once per disk and cached); `𝒱(x, u, D)` and `𝒯_vol(x, u, D)` evaluate at arbitrary points, inside,
on or outside the disk.

`ℬ(u, D; κ)`, the 2D bilaplacian potential ``\frac{1}{8\pi}\int_D |x - y|^2 (\log|x - y| - 1 + \kappa)\, u(y)\,
dy`` (so ``\Delta^2 \mathcal{B}[u] = u``), is a dense direct sum over the grid, ``O(N^2)`` in the number ``N`` of
grid points; its kernel is continuous, and it converges algebraically, not spectrally.

`𝒯(u, D)`, the single layer of the 3D bilaplacian (kernel
``|x - y|/8\pi``), is spectral. It uses ``|x - y| = |x - y|^2/|x - y|`` and expands ``|x - y|^2 = |x|^2 - x\bar y -
\bar x y + |y|^2`` in complex notation, which gives four applications of `𝒮`:

```math
\mathcal{T}[u] = \tfrac{1}{2}\left(|\zeta|^2 \mathcal{S}[u] - \zeta\, \mathcal{S}[\bar\zeta u] - \bar\zeta\,
\mathcal{S}[\zeta u] + \mathcal{S}[|\zeta|^2 u]\right).
```

### Volume potentials on mapped domains

On a `Domain` (`ellipse`, or `domain` with any map), `𝒱(u, Ω)` is spectrally accurate. Pulled back to the disk,
the density is ``g = (u \circ F)\,J``, smooth, and the kernel splits with the Wirtinger divided differences ``F(\zeta)
- F(\eta) = P\,\Delta\zeta + Q\,\overline{\Delta\zeta}`` of each pair of points (see [Mapped domains](domains.md)), ``\mu
= Q/P`` and ``\phi = \arg \Delta\zeta``:

```math
\log|F(\zeta) - F(\eta)| = \log|\Delta\zeta| + \log|P| + \sum_{n \ge 1} \frac{(-1)^{n+1}}{2n}
\left(\mu^n e^{-2in\phi} + \bar\mu^n e^{2in\phi}\right).
```

``P`` and ``\mu`` are smooth in both points, so each term is a smooth function times one of the kernels ``\log|\Delta
\zeta|`` or ``e^{2ik\phi}``, bounded but discontinuous in direction at the target. Their product-integration weights
against the even PSH, computed for one target per ring by polar quadrature about it, rotate to the other targets of
the ring, and composed with the transform give grid operators of the disk; the operator on ``\Omega`` sums them with
the coefficients of each pair, as for `𝒮`. The series is truncated where ``|\mu|^K`` reaches rounding. Taking ``\mu``
at the target alone, rather than for the pair, would leave a remainder that is continuous but not smooth at the
target, and its plain quadrature converges only algebraically. `volume_matrix(Ω; kernel)` returns these weights as an
``N \times N`` matrix on the grid, also for the kernel ``|x - y| = |P|\,|\Delta\zeta|\,|1 + \mu e^{-2i\phi}|``
(`kernel = :abs`), whose angular factor is expanded the same way. `𝒱(x, u, Ω)` evaluates at points ``x`` inside
``\Omega`` (the map is inverted by Newton's method), and anywhere for an ellipse, whose affine map extends to the plane.

On a ``1.5 \times 0.67`` ellipse, `𝒱(1, Ω)` matches the closed form (a quadratic inside) to ``10^{-15}`` at `M = 8`,
and a smooth density converges like the grid resolves it: errors of ``2 \times 10^{-8}``, ``10^{-12}`` and
``10^{-15}`` at `M = 8, 12, 16`, also at points within ``10^{-4}`` of the boundary.

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
