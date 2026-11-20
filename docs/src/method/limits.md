# Limits

What the method does not do, and where it needs care.

**Geometry**
- Surfaces are flat: a disk, an ellipse, or the inside of one smooth closed curve, in the plane ``z = 0``. There
  are no holes, no corners, and no curved surfaces.
- Mapped domains (`domain`) are experimental; the next items are their known limits.
- The harmonic map (the default) is one-to-one for convex curves. A non-convex curve may make it fold, which
  `domain` reports as an `ArgumentError`; reparametrizing the curve, or `map = :conformal`, may help. A harmonic
  domain has no hypersingular operator `𝒩`.
- The conformal map (`map = :conformal`) needs the origin inside the curve, and does not check. A curve that does
  not enclose it gives wrong operators without an error (see [Mapped domains](@ref method-domains)). It crowds the
  grid on elongated shapes: a 4:1 ellipse is off by 37% at ``M = 32``.
- The curve must be smooth and resolved by its samples.
- `domain` builds dense matrices in coefficient space, of size about ``N/4`` (``N`` grid points): 10 MiB at
  ``M = 24``, growing like ``M^4`` in memory and ``M^5`` (conformal) or ``M^6`` (harmonic) in time. Ellipses should use `ellipse`, which is sparse
  and cheap.

**Operators**
- The volume potentials `𝒱` and `ℬ` are dense direct sums, ``O(N^2)``, and converge algebraically.
- The differential operators (`∂ζ`, `∂ζ̄`, `grad`, `∂x`, `∂y`, `div`, `lap`) divide by ``w`` in coefficient space,
  which is ill-conditioned. They lose digits as the resolution grows. `∂n` does not have this problem.
- These are defined on the disk only: `𝒱`, `ℬ`, `𝒯`, `𝒮𝒩⁻¹`, `Δ⁻¹`, the differential operators, and the shape
  derivatives. The layer potentials `𝒮`, `𝒩`, their inverses, and the Stokes operators work on every domain.
- Densities must have the edge behavior the operator expects: ``g/w`` for `𝒮`, `𝒯` and `𝒮_st`, ``w g`` for `𝒩`.
  A density without it, such as a smooth density passed to `𝒮`, still converges, but only algebraically.
- `𝒮_st(tgt, f, Ω)` (the flow off a body) is direct quadrature over the grid, inaccurate within a few grid spacings
  of the body.

**Normal forces**
- The normal (out-of-plane) problem is defined in free space only (`μ`, one fluid above and below). It is
  available for single bodies; the suspension solver is in-plane only.

**Suspensions**
- Every body is a copy of the same `Ω`. Bodies of different shapes or sizes need separate discretizations, which
  the solver does not support.
- Bodies must not overlap. For disks and ellipses, the close-evaluation setup throws if they do; bodies from
  `domain` are not checked.
- Close evaluation applies to disks and ellipses only. Bodies from `domain` use the node sum at every target, which
  loses accuracy when they come within a few grid spacings of each other.
- The close-evaluation zone `close_radius` is fixed (``c = 8`` grid spacings by default) and does not follow `eps`.
  At loose `eps` it corrects more targets than needed. At tight `eps` and low ``M`` it corrects too few: for two
  disks at ``M = 16``, the cross velocity levels off at 2e-10, set by the targets just outside the zone, and
  `close_radius = 20` brings it to 3e-12 (see [Validation](@ref)).
- The Ewald sum is tested against FMM3D at `eps` 1e-6 and 1e-10 (relative differences 1.7e-9 and 4.8e-10 on 12
  disks), not at tighter tolerances.
- The fast sums run on all threads, with their own OpenMP threads alongside Julia's. A heavily loaded machine, or a
  multithreaded BLAS spinning between steps, slows them noticeably.
