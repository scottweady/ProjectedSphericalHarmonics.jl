# [Stokes flow](@id method-stokes)

## In-plane Stokes flow

A thin body in the plane ``z = 0``, between a fluid of viscosity ``\mu_\mathrm{bottom}`` below and one of
viscosity ``\mu_\mathrm{top}`` above, moves in its plane. A force density ``f`` on it drives a flow whose velocity
in the plane is

```math
u(x) = \mathcal{S}_\mathrm{st}[f](x) = \int_\Omega G(x - y)\, f(y)\, dy, \qquad
G(\mathbf{r}) = \frac{1}{4\pi\mu}\left(\frac{\mathbf{I}}{|\mathbf{r}|} + \frac{\mathbf{r}\mathbf{r}^T}{|\mathbf{r}|^3}\right),
\qquad \mu = \mu_\mathrm{bottom} + \mu_\mathrm{top}.
```

This is the in-plane block of the 3D Stokeslet. It covers several settings:
- the default, ``\mu_\mathrm{bottom} = 1`` and ``\mu_\mathrm{top} = 0``: a half-space of unit viscosity under a
  stress-free surface;
- equal viscosities: free space, where ``G`` is the usual ``1/(8\pi\mu_\mathrm{bottom})`` Stokeslet;
- unequal viscosities: an interface between two fluids.

A translating disk of unit radius has drag ``16\mu/3`` in the half-space and ``32\mu/3`` in free space, and the
same values hold for the torque of a rotating disk.

## Angular kernels

Write ``x - y = |x - y|(\cos\phi, \sin\phi)``. Each entry of ``G`` is a function of the angle ``\phi`` alone,
divided by ``|x - y|``:

```math
4\pi\mu\, G = \frac{1}{|x - y|}\begin{pmatrix} 1 + \cos^2\phi & \sin\phi\cos\phi \\ \sin\phi\cos\phi & 1 + \sin^2\phi \end{pmatrix}.
```

A kernel ``g(\phi)/|x - y|`` with ``g(\phi) = \sum_k \hat g_k e^{ik\phi}`` acts on the PSH by shifting the order:
each Fourier mode ``k`` sends ``y_\ell^m/w`` to a multiple ``C_{\ell}^{m,k}\, y_\ell^{m+k}`` (`Clmn`), at the same
degree ``\ell``. The matrix in coefficient space is therefore sparse, with one diagonal per Fourier mode of
``g``. `laplace3d_angular_matrix` builds it from ``g`` or its Fourier coefficients and keeps modes until they fall
below a tolerance (1e-14 by default). ``\cos^2\phi``, ``\sin\phi\cos\phi`` and ``\sin^2\phi`` have the modes
``0, \pm 2``, so on the disk each block of ``G`` has three diagonals. The ellipse's blocks have more, since its
angular factor is a nonpolynomial function of ``\phi`` (see [Non-circular domains](@ref method-domains)).

`𝒮_st(f, Ω)` and `𝒮_st⁻¹(u, Ω)` act on tuples `(fx, fy)` with the ``2 \times 2`` block of these sparse matrices
(`Ω.K̂_G`), on the even coefficients of ``w f``. As with `𝒮`, the force density of a first-kind solve behaves like
``1/w`` at the edge, and `𝒮_st⁻¹` returns it in that form. The inverse is a sparse LU solve of a matrix of size
twice the number of even modes.

`𝒮_st(tgt, f, Ω)` evaluates the flow at points off the body by direct quadrature over the grid. It is accurate
away from the body, and loses accuracy within a few grid spacings of it.

## The mobility problem of one body

A rigid body moving with velocity ``U`` and rotation rate ``\omega`` in a background flow ``u_\infty``, with an
optional slip velocity ``u_s`` on its surface, carries the force density ``f`` that satisfies

```math
\mathcal{S}_\mathrm{st}[f] + u_\infty = U + \omega\, (-y, x) + u_s \ \text{ on } \Omega, \qquad
\int_\Omega f\, dA = (F_x, F_y), \quad \int_\Omega (x f_y - y f_x)\, dA = T,
```

with the applied force and torque given. Let ``e_1 = (1, 0)``, ``e_2 = (0, 1)`` and ``e_3 = (-y, x)`` be the rigid
motions. The force and torque of ``\mathcal{S}_\mathrm{st}^{-1} e_k`` form the ``3 \times 3`` resistance matrix,
which `stokes_mobility_matrix(Ω)` returns (``M`` below). By linearity,
`stokes_mobility_solve(uinf, uslip, F, Ω)`:

1. computes the force density ``\mathcal{S}_\mathrm{st}^{-1}[u_\infty - u_s]`` and its force and torque
   ``F_\infty``;
2. solves ``M\,(U, \omega) = F + F_\infty``.

Each call costs one inverse, or four if `M` is not passed.
