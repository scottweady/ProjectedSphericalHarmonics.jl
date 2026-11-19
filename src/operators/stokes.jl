
"""
    𝒮_st(f, Ω; μ_bottom=1.0, μ_top=0.0)

Single layer operator for in-plane Stokes flow driven by forces in the plane z = 0, which
separates fluids of viscosity `μ_bottom` (below) and `μ_top` (above). The Green's function is

    G(r) = (I + r̂ r̂ᵀ) / (4π (μ_bottom + μ_top) |r|).

The default (`μ_bottom = 1`, `μ_top = 0`) is a half-space of unit viscosity under a
stress-free surface. Free space is `μ_bottom = μ_top`, giving the usual 1/(8π μ_bottom).

# Arguments
- `f` : array of force values, given as a tuple (fx, fy)
- `Ω` : domain discretization
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- array of velocity values, given as a tuple (ux, uy)
"""
function 𝒮_st(f::Tuple, Ω; μ_bottom=1.0, μ_top=0.0)

  μ = μ_bottom + μ_top

  if isa(f[1], Number)
    f = (fill(f[1], size(Ω.z)), fill(f[2], size(Ω.z)))
  end

  f̂ = psh(f .* Ω.w, Ω, parity=:even)
  û = apply(Ω.K̂_G, f̂, Ω) ./ μ
  return ipsh(û, Ω, parity=:even)
  
end

"""
    𝒮_st(i, j, f, Ω; μ_bottom=1.0, μ_top=0.0)

Single layer operator for in-plane Stokes flow (see `𝒮_st(f, Ω)`), evaluated at a specific component

# Arguments
- `i` : row index of the Green's function matrix (1 for x-component, 2 for y-component)
- `j` : column index of the Green's function matrix (1 for x-component, 2 for y-component)
- `f` : array of force values, given as a tuple (fx, fy)
- `Ω` : domain discretization
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- array of function values for the specified component
"""
function 𝒮_st(i::Int, j::Int, f::AbstractMatrix{ComplexF64}, Ω; μ_bottom=1.0, μ_top=0.0)
  μ = μ_bottom + μ_top
  f̂ = psh(f .* Ω.w, Ω, parity=:even)
  û = apply(Ω.K̂_G[i, j], f̂, Ω) ./ μ
  return ipsh(û, Ω, parity=:even)
end

"""
    𝒮_st⁻¹(u, Ω; μ_bottom=1.0, μ_top=0.0)

Inverse of the single layer operator for in-plane Stokes flow (see `𝒮_st(f, Ω)`)

# Arguments
- `u` : array of velocity values, given as a tuple (ux, uy)
- `Ω` : domain discretization
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- array of force values, given as a tuple (fx, fy)
"""
function 𝒮_st⁻¹(u::Tuple, Ω; μ_bottom=1.0, μ_top=0.0)

  μ = μ_bottom + μ_top

  if isa(u[1], Number)
    u = (fill(u[1], size(Ω.z)), fill(u[2], size(Ω.z)))
  end

  û = psh(u, Ω, parity=:even)
  f̂ = solve(Ω.K̂_G, û, Ω, parity=:even) .* μ
  return ipsh(f̂, Ω, parity=:even) ./ Ω.w
  
end

# ---- Normal (out-of-plane) forces: free space only ------------------------------------------------
# On the plane z = 0 the 3D Stokeslet has G_xz = G_yz = 0, so a normal force density f_z drives only
# the normal velocity, with G_zz = 1/(8πμ|r|): u_z = 𝒮[f_z]/(2μ), the Laplace single layer. In a
# half-space or between two fluids the plane is an interface that a normal force would deform, so
# the normal problem is set up for one fluid of viscosity μ above and below.

_as_field(c, Ω) = c isa Number ? fill(ComplexF64(c), size(Ω.z)) : c

function _free_space_only(μ_bottom, μ_top)
  (μ_bottom === nothing && μ_top === nothing) && return nothing
  throw(ArgumentError("forces with a normal component are for free space only (one fluid above and below " *
                      "the plane): pass its viscosity as μ instead of μ_bottom and μ_top"))
end

"""
    𝒮_st(f::NTuple{3}, Ω; μ=1.0)

Stokes single layer of a force density with a normal component, `f = (fx, fy, fz)`, in free space:
one fluid of viscosity `μ` above and below the plane. Returns `(ux, uy, uz)`. The in-plane and
normal parts decouple on the plane: `(ux, uy)` is `𝒮_st((fx, fy), Ω; μ_bottom=μ, μ_top=μ)`, and

    uz = 𝒮[fz] / (2μ)      (G_zz = 1/(8πμ|r|), the Laplace single layer `𝒮`).

The normal problem is not defined in a half-space or between two fluids (a normal force would
deform the interface), so `μ_bottom` and `μ_top` are rejected here. Components may be numbers.
"""
function 𝒮_st(f::NTuple{3,Any}, Ω; μ=1.0, μ_bottom=nothing, μ_top=nothing)
  _free_space_only(μ_bottom, μ_top)
  fx, fy, fz = (_as_field(c, Ω) for c in f)
  ux, uy = 𝒮_st((fx, fy), Ω; μ_bottom=μ, μ_top=μ)
  return ux, uy, 𝒮(fz, Ω) ./ (2μ)
end

"""
    𝒮_st⁻¹(u::NTuple{3}, Ω; μ=1.0)

Inverse of `𝒮_st(f::NTuple{3}, Ω; μ)`: the force density `(fx, fy, fz)` that produces the
velocity `u = (ux, uy, uz)` on `Ω` in free space (viscosity `μ`). The normal part is
`fz = 2μ 𝒮⁻¹[uz]`; like the in-plane part, it has the ``1/w`` edge singularity. Components may be
numbers.
"""
function 𝒮_st⁻¹(u::NTuple{3,Any}, Ω; μ=1.0, μ_bottom=nothing, μ_top=nothing)
  _free_space_only(μ_bottom, μ_top)
  ux, uy, uz = (_as_field(c, Ω) for c in u)
  fx, fy = 𝒮_st⁻¹((ux, uy), Ω; μ_bottom=μ, μ_top=μ)
  return fx, fy, (2μ) .* 𝒮⁻¹(uz, Ω)
end

"""
    stokes_normal_mobility_matrix(Ω; μ=1.0)

The 3 × 3 matrix of the out-of-plane rigid motions of a body in free space (viscosity `μ`):
entry `[i, k]` is the generalized force `k` (``F_z``, ``T_x = \\int y f_z``, ``T_y = -\\int x f_z``)
of the motion `i` (translation ``U_z``, rotations ``Ω_x``, ``Ω_y``, with ``u_z = U_z + Ω_x y - Ω_y x``),
the counterpart of `stokes_mobility_matrix` for the normal direction. For the unit disk it is
`diag(16, 32/3, 32/3) μ`.
"""
function stokes_normal_mobility_matrix(Ω; μ=1.0)
  x, y = real.(Ω.z), imag.(Ω.z)
  M = zeros(ComplexF64, 3, 3)
  for (i, e) in enumerate((1.0, y, -x))
    fz = (2μ) .* 𝒮⁻¹(e, Ω)
    M[i, 1] = integral(fz, Ω)
    M[i, 2] = integral(y .* fz, Ω)
    M[i, 3] = integral(-x .* fz, Ω)
  end
  return M
end

"""
    stokes_mobility_matrix(Ω; μ_bottom=1.0, μ_top=0.0)

Compute the mobility matrix for a domain Ω in in-plane Stokes flow (see `𝒮_st(f, Ω)`).

# Arguments
- `Ω` : domain discretization (e.g., disk or ellipse)
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- 3x3 mobility matrix M relating forces and torques to velocities and angular velocity
"""
function stokes_mobility_matrix(Ω; μ_bottom=1.0, μ_top=0.0)

  x, y = real.(Ω.z), imag.(Ω.z)

  e1 = (1.0, 0.0)
  e2 = (0.0, 1.0)
  e3 = (-y, x)
  E = (e1, e2, e3)

  M = zeros(ComplexF64, 3, 3)

  for (i, e) in enumerate(E)
    f = 𝒮_st⁻¹(e, Ω; μ_bottom=μ_bottom, μ_top=μ_top)
    M[i,1] = integral(f[1], Ω)
    M[i,2] = integral(f[2], Ω)
    M[i,3] = integral(-y .* f[1] + x .* f[2], Ω)
  end

  return M

end

"""
    stokes_mobility_solve(uinf, uslip, F, Ω; M=[], μ_bottom=1.0, μ_top=0.0)

Solve for the velocity and angular velocity of a domain Ω in in-plane Stokes flow given background flow, forces, and torques.

# Arguments
- `uinf`  : background velocity field (tuple of x and y components)
- `uslip` : prescribed slip velocity (tuple of x and y components), or nothing for no-slip
- `F`     : applied force and torque vector [Fx, Fy, T]
- `Ω`     : domain discretization (e.g., disk or ellipse)
- `M`     : precomputed mobility matrix (optional; must be computed with the same viscosities)
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- Tuple (U, ω) where U is the velocity vector and ω is the angular velocity

# Normal forces (free space)
With `F = [Fx, Fy, Fz, Tx, Ty, Tz]` (six components), the body also moves out of the plane, in
free space with one fluid of viscosity `μ` (keyword, default 1.0; `μ_bottom` and `μ_top` are
rejected). `uinf` and `uslip` are then tuples `(ux, uy, uz)`, and the result is
`(Ux, Uy, Uz), (Ωx, Ωy, Ωz)`. The in-plane and normal problems decouple; `M` is the in-plane
matrix (computed with `μ_bottom = μ_top = μ`) and `Mn` the normal one
(`stokes_normal_mobility_matrix`), both computed here if not given.
"""
function stokes_mobility_solve(uinf, uslip, F, Ω; M=[], Mn=[], μ=nothing, μ_bottom=nothing, μ_top=nothing)

  if length(F) == 6
    return _stokes_mobility_solve6(uinf, uslip, F, Ω, M, Mn, something(μ, 1.0), μ_bottom, μ_top)
  end
  μ === nothing || throw(ArgumentError("μ is for forces with a normal component (F = [Fx, Fy, Fz, Tx, Ty, Tz]); " *
                                       "for in-plane forces pass μ_bottom and μ_top"))
  μ_bottom = something(μ_bottom, 1.0)
  μ_top    = something(μ_top, 0.0)

  isempty(M) ? M = stokes_mobility_matrix(Ω; μ_bottom=μ_bottom, μ_top=μ_top) : nothing

  x, y = real.(Ω.z), imag.(Ω.z)

  ueff = uslip === nothing ? uinf : uinf .- uslip
  finf = 𝒮_st⁻¹(ueff, Ω; μ_bottom=μ_bottom, μ_top=μ_top)

  Finf1 = integral(finf[1], Ω)
  Finf2 = integral(finf[2], Ω)
  Tinf  = integral(-y .* finf[1] + x .* finf[2], Ω)

  R = M \ [F[1] .+ Finf1; F[2] .+ Finf2; F[3] .+ Tinf]
  U, ω = (R[1], R[2]), R[3]

  return U, ω

end

# Six degrees of freedom in free space: the in-plane problem (Fx, Fy, Tz) by the three-component
# solve with μ_bottom = μ_top = μ, and the normal one (Fz, Tx, Ty) with the Laplace single layer
function _stokes_mobility_solve6(uinf, uslip, F, Ω, M, Mn, μ, μ_bottom, μ_top)

  _free_space_only(μ_bottom, μ_top)
  (uinf isa Tuple && length(uinf) == 3 && (uslip === nothing || (uslip isa Tuple && length(uslip) == 3))) ||
    throw(ArgumentError("with F = [Fx, Fy, Fz, Tx, Ty, Tz], uinf and uslip are tuples (ux, uy, uz) (uslip may be nothing)"))

  U2, ωz = stokes_mobility_solve((uinf[1], uinf[2]), uslip === nothing ? nothing : (uslip[1], uslip[2]),
                                 [F[1], F[2], F[6]], Ω; M=M, μ_bottom=μ, μ_top=μ)

  isempty(Mn) && (Mn = stokes_normal_mobility_matrix(Ω; μ=μ))
  x, y = real.(Ω.z), imag.(Ω.z)
  uz = uslip === nothing ? _as_field(uinf[3], Ω) : _as_field(uinf[3], Ω) .- _as_field(uslip[3], Ω)
  fz = (2μ) .* 𝒮⁻¹(uz, Ω)
  R = Mn \ [F[3] + integral(fz, Ω); F[4] + integral(y .* fz, Ω); F[5] + integral(-x .* fz, Ω)]

  return (U2[1], U2[2], R[1]), (R[2], R[3], ωz)

end

"""
    𝒮_st(tgt, f, Ω; μ_bottom=1.0, μ_top=0.0)

Evaluate the in-plane Stokes single layer potential at points outside the domain.

The Green's function is G(r) = (I + r̂ r̂ᵀ) / (4π (μ_bottom + μ_top) |r|), where r = x - y
is the 2D separation vector. Direct quadrature is used since tgt lies outside
the support and there is no singularity.

# Arguments
- `tgt` : evaluation points as a complex array (x + iy)
- `f`     : force density as a tuple (fx, fy)
- `Ω`     : domain discretization
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0; see `𝒮_st(f, Ω)`)

# Returns
- Tuple (ux, uy) with the same shape as `tgt`
"""
function 𝒮_st(tgt::ComplexF64, f::Tuple, Ω; μ_bottom=1.0, μ_top=0.0)

  μ = μ_bottom + μ_top

  if isa(f[1], Number)
    f = (fill(f[1], size(Ω.z)), fill(f[2], size(Ω.z)))
  end

  z, dz = vec(Ω.z), vec(Ω.dz)

  # Quadrature-weighted force components (dζ is real)
  fxw, fyw = vec(f[1]) .* dz, vec(f[2]) .* dz

  # Displacement matrix: r[i,j] = x[i] - z[j],  shape (n_eval × n_src)
  r = tgt .- z
  rx, ry, rabs = real.(r), imag.(r), abs.(r)

  # r · f  (dot product in 2D, shape n_eval × n_src)
  rdotf = rx .* fxw .+ ry .* fyw

  ux = (1 / (4π * μ)) .* sum(fxw ./ rabs .+ rdotf .* rx ./ rabs.^3)
  uy = (1 / (4π * μ)) .* sum(fyw ./ rabs .+ rdotf .* ry ./ rabs.^3)

  return (ux, uy)

end

𝒮_st(tgt::AbstractArray{ComplexF64}, f::Tuple, Ω; μ_bottom=1.0, μ_top=0.0) = 𝒮_st.(tgt, Ref(f), Ref(Ω); μ_bottom=μ_bottom, μ_top=μ_top)

export stokes_mobility_solve, stokes_mobility_matrix, stokes_normal_mobility_matrix
