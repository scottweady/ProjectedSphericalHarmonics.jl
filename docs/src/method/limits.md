# Limits

What the method does not do, and where it needs care.

**Geometry**
- Surfaces are flat: a disk, an ellipse, or the inside of one smooth closed curve, in the plane ``z = 0``. There
  are no holes, no corners, and no curved surfaces.
- Conformally mapped domains (`domain`) are experimental; the next three items are their known limits.
- `domain(γ, M)` needs the origin inside the curve, and does not check. A curve that does not enclose it gives wrong
  operators without an error (see [Conformally mapped domains](@ref method-domains)). The curve must be smooth and
  resolved by its samples.
- `domain` converges more slowly as the shape gets more elongated, at a rate set by the conformal map, and more
  boundary samples do not help. For an ellipse with ``a/b = 2`` built by `domain`, the conductor density is still
  off by 3e-2 at ``M = 32``, where ``a/b = 1.5`` reaches 6e-5 (see [Validation](@ref)).
- `domain` builds dense ``N \times N`` matrices (``N`` grid points): 123 MiB and 1.8 s at ``M = 24``, growing like
  ``M^4`` in memory. Ellipses should use `ellipse`, which is sparse and cheap.

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
