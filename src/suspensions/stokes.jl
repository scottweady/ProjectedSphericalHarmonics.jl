
using FMM3D

rotate(ux, uy, θ) = (cos(θ) .* ux .- sin(θ) .* uy,
                     sin(θ) .* ux .+ cos(θ) .* uy)

function suspension_velocity_eval!(u_body, sources, stoklets, xcm, θcm, f_body, Ω; eps=1e-6)

  N_body   = length(f_body)
  Nz  = length(Ω.z)
  shp = size(Ω.z)

  # Zero output
  for i = 1 : N_body
    fill!(u_body[i][1], 0)
    fill!(u_body[i][2], 0)
  end

  # Self-interaction via PSH operator (body frame, no rotation needed)
  for i = 1 : N_body
    ui_body = 𝒮_st(f_body[i], Ω)
    u_body[i][1] .+= ui_body[1]
    u_body[i][2] .+= ui_body[2]
  end

  # Off-diagonal interactions via FMM3D
  # Sources and stoklets are in world frame: rotate body-frame quantities by θcm[i]
  x, y, dz = real.(vec(Ω.z)), imag.(vec(Ω.z)), real.(vec(Ω.dz))

  for j = 1 : N_body
    idx  = (j-1)*Nz .+ (1:Nz)
    rx, ry = rotate(x, y, θcm[j])
    sources[1, idx]  = real(xcm[j]) .+ rx
    sources[2, idx]  = imag(xcm[j]) .+ ry
    # Rotate body-frame force to world frame; area element Ω.dz is a real scalar
    fx_world, fy_world = rotate(real.(vec(f_body[j][1])), real.(vec(f_body[j][2])), θcm[j])
    stoklets[1, idx] = fx_world .* dz
    stoklets[2, idx] = fy_world .* dz
  end

  # One FMM call: evaluate at all source locations (ppreg=1 excludes exact self-term)
  out = stfmm3d(eps, sources; stoklet=stoklets, ppreg=1)

  # Subtract within-disk contributions: FMM includes them but 𝒮_st already handles them
  for j = 1 : N_body
    idx = (j-1)*Nz .+ (1:Nz)
    self = st3ddir(sources[:, idx], sources[:, idx]; stoklet=stoklets[:, idx], ppregt=1)
    out.pot[:, idx] .-= self.pottarg
  end

  # FMM uses G/(8π), stokeslet_direct uses G/(4π), so multiply by 2
  # Rotate world-frame FMM output back to body frame of each target body
  for i = 1 : N_body
    idx  = (i-1)*Nz .+ (1:Nz)
    ux_world = 2 .* reshape(out.pot[1, idx], shp)
    uy_world = 2 .* reshape(out.pot[2, idx], shp)
    du1, du2 = rotate(ux_world, uy_world, -θcm[i])
    u_body[i][1] .+= du1
    u_body[i][2] .+= du2
  end

  return u_body

end

function suspension_velocity_eval(xcm, θcm, f_body::Vector, Ω; eps=1e-6)
  N_body   = length(f_body)
  Nz  = length(Ω.z)
  shp = size(Ω.z)
  u_body = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1 : N_body]
  sources  = zeros(Float64, 3, N_body * Nz)
  stoklets = zeros(Float64, 3, N_body * Nz)
  return suspension_velocity_eval!(u_body, sources, stoklets, xcm, θcm, f_body, Ω; eps=eps)
end

"""
    suspension_velocity_solve(xcm, θcm, u_body, Ω; eps=1e-6)

Solve the Stokes suspension problem: given collocation-space velocities u_body, find
collocation-space forces f_body such that stokes_suspension(xcm, f_body, Ω) ≈ u_body.

Uses GMRES with the forward operator in PSH coefficient space via the substitution
f = f̃/Ω.w to improve conditioning.

# Arguments
- `xcm` : vector of N particle center positions
- `θcm` : vector of N particle orientations (radians)
- `u_body`   : vector of N tuples of velocity matrices (ux, uy) at collocation points
- `Ω`   : disk discretization

# Returns
- Vector of N tuples of force matrices (fx, fy) at collocation points
"""
function suspension_velocity_solve(xcm, θcm, u_body::Vector, Ω; eps=1e-6)

  N_body   = length(u_body)
  Nz  = length(Ω.z)
  shp = size(Ω.z)

  # Pre-allocate buffers reused across GMRES iterations
  ũ_buf    = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1 : N_body]
  sources  = zeros(Float64, 3, N_body * Nz)
  stoklets = zeros(Float64, 3, N_body * Nz)

  pack(V̂) = vcat([vcat(vec(V̂[i][1]), vec(V̂[i][2])) for i in 1 : N_body]...)

  function unpack(v)
    [(reshape(v[(2(i-1))*Nz .+ (1:Nz)], shp),
      reshape(v[(2i-1)*Nz   .+ (1:Nz)], shp)) for i in 1 : N_body]
  end

  function A!(result, v)
    f̃ = unpack(v)
    f  = [fi ./ Ω.w for fi in ipsh.(f̃, Ref(Ω))]
    suspension_velocity_eval!(ũ_buf, sources, stoklets, xcm, θcm, f, Ω; eps=eps)
    result .= pack(psh.(ũ_buf, Ref(Ω)))
  end

  û  = psh.(u_body, Ref(Ω))
  F̃  = unpack(gmres(A!, pack(û)))
  return [fi ./ Ω.w for fi in ipsh.(F̃, Ref(Ω))]

end

# IterativeSolvers.gmres! requires ldiv!(Pl, x); this wraps a function f!(out, in)
struct _FuncPrecond f!::Function end
LinearAlgebra.ldiv!(P::_FuncPrecond, x::AbstractVector) = (tmp = copy(x); P.f!(x, tmp); x)

"""
    suspension_mobility_solve(xcm, θcm, uinf, uslip, F, Ω; eps=1e-6, M=[])

Solve the N-body Stokes mobility problem via GMRES.

The boundary condition is `uinf + S·f = A·U + uslip`, giving the linear system
`[S, -A; B, 0] [f; U] = [uslip - uinf; F]`. For rigid particles with no prescribed
slip, pass `uslip = nothing` (or a zero function). For squirmers, `uslip` is the
prescribed surface slip velocity.

Both `uinf` and `uslip` can be functions `z -> (ux, uy)` evaluated at the actual body
positions `xcm[i] + exp(iθcm[i]) * Ω.z`, or pre-evaluated vectors of N tuples.

Preconditioned by independent single-body mobility solves (block-diagonal).

# Arguments
- `xcm`   : vector of N complex center positions
- `θcm`   : vector of N orientations (radians)
- `uinf`  : background flow as a function z -> (ux, uy), or vector of N tuples
- `uslip` : prescribed slip velocity (same form as uinf), or nothing for no-slip
- `F`     : vector of N vectors [Fx, Fy, T] (forces and torques)
- `Ω`     : domain discretization (shared by all bodies)
- `M`     : precomputed single-body mobility matrix (optional)

# Returns
- `f` : vector of N force density tuples (fx, fy)
- `U` : vector of N rigid body velocity vectors [Ux, Uy, ω]
"""
function suspension_mobility_solve(xcm, θcm, uinf, uslip, F::Vector, Ω; eps=1e-6, M=[])

  N_body   = length(xcm)
  Nz  = length(Ω.z)
  shp = size(Ω.z)
  x, y = real.(Ω.z), imag.(Ω.z)

  eval_field(u) = u isa Function  ? [u(xcm[i] .+ exp(im*θcm[i]) .* Ω.z) for i in 1 : N_body] :
                  u === nothing   ? [(zeros(shp), zeros(shp)) for _ in 1 : N_body] : u
  uinf_eval  = [rotate(eval_field(uinf)[i]...,  -θcm[i]) for i in 1 : N_body]
  uslip_eval = [rotate(eval_field(uslip)[i]..., -θcm[i]) for i in 1 : N_body]

  # Rotate world-frame forces/torques to body frame
  F_body = [vcat(collect(rotate(F[i][1], F[i][2], -θcm[i])), F[i][3]) for i in 1 : N_body]

  # Precompute single-body mobility matrix (same disk for all bodies)
  M_mob = isempty(M) ? stokes_mobility_matrix(Ω) : M

  # Packing layout: [f̃_1_x; f̃_1_y; ...; f̃_N_x; f̃_N_y; U_1; ...; U_N]
  # where f̃_i = psh(f_i .* Ω.w, Ω) so that f_i = ipsh(f̃_i, Ω) ./ Ω.w
  Nfield = 2N_body * Nz
  Ntotal = Nfield + 3N_body

  function pack(f̃, U)
    v = zeros(ComplexF64, Ntotal)
    for i in 1 : N_body
      v[(2(i-1))*Nz .+ (1:Nz)] = vec(f̃[i][1])
      v[(2i-1)*Nz   .+ (1:Nz)] = vec(f̃[i][2])
    end
    for i in 1 : N_body
      v[Nfield + 3(i-1) + 1] = U[i][1]
      v[Nfield + 3(i-1) + 2] = U[i][2]
      v[Nfield + 3(i-1) + 3] = U[i][3]
    end
    return v
  end

  function unpack(v)
    f̃ = [(reshape(v[(2(i-1))*Nz .+ (1:Nz)], shp),
           reshape(v[(2i-1)*Nz   .+ (1:Nz)], shp)) for i in 1 : N_body]
    U  = [[v[Nfield + 3(i-1) + 1],
            v[Nfield + 3(i-1) + 2],
            v[Nfield + 3(i-1) + 3]] for i in 1 : N_body]
    return f̃, U
  end

  # Pre-allocated FMM buffers
  u_buf    = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1 : N_body]
  sources  = zeros(Float64, 3, N_body * Nz)
  stoklets = zeros(Float64, 3, N_body * Nz)

  function matvec!(result, v)
    f̃, U = unpack(v)
    f = [ipsh(f̃[i], Ω) ./ Ω.w for i in 1 : N_body]

    # S·f: self via spectral (inside FMM call), cross via FMM
    suspension_velocity_eval!(u_buf, sources, stoklets, xcm, θcm, f, Ω; eps=eps)

    # Subtract A·U: both u_buf and U are body-frame
    for i in 1 : N_body
      Ux, Uy, ω = real(U[i][1]), real(U[i][2]), real(U[i][3])
      u_buf[i][1] .-= Ux .- ω .* y
      u_buf[i][2] .-= Uy .+ ω .* x
    end

    # Field residual in PSH coefficient space
    f̃_out = [psh(u_buf[i], Ω) for i in 1 : N_body]

    # B·f: force/torque integrals
    U_out = [[integral(f[i][1], Ω),
               integral(f[i][2], Ω),
               integral(-y .* f[i][1] .+ x .* f[i][2], Ω)] for i in 1 : N_body]

    result .= pack(f̃_out, U_out)
  end

  function precond!(result, v)
    f̃, rU = unpack(v)
    # Interpret PSH coefficients as velocity fields for the preconditioner
    rf = [ipsh(f̃[i], Ω) for i in 1 : N_body]

    new_f̃ = Vector{Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(undef, N_body)
    new_U  = Vector{Vector{ComplexF64}}(undef, N_body)

    for i in 1 : N_body
      # stokes_mobility_solve uses convention S·f = A·U - uinf, so pass -rf
      δU, δω = stokes_mobility_solve(.-rf[i], nothing,
                                      (real(rU[i][1]), real(rU[i][2]), real(rU[i][3])), Ω; M=M_mob)

      # Velocity that δf must produce: S·δf = A·δU - (-rf) = A·δU + rf
      ũ = (rf[i][1] .+ δU[1] .- δω .* y,
           rf[i][2] .+ δU[2] .+ δω .* x)

      # Recover force density via single-body inverse
      δf = 𝒮_st⁻¹(ũ, Ω)

      # Store PSH coefficients of δf·w (the packed representation)
      new_f̃[i] = psh(δf .* Ω.w, Ω)
      new_U[i]  = [δU[1], δU[2], δω]
    end

    result .= pack(new_f̃, new_U)
  end

  # RHS: system is S·f - A·U = -uinf (boundary condition S·f + uinf = A·U)
  # uinf_eval and F_body are already in body frame
  rhs = pack([psh(uslip_eval[i] .- uinf_eval[i], Ω) for i in 1 : N_body], F_body)

  # GMRES with block-diagonal preconditioner
  sol = zeros(ComplexF64, Ntotal)
  op  = LinearOperator(ComplexF64, Ntotal, Ntotal, false, false, matvec!)
  sol, history = gmres!(sol, op, rhs; Pl=_FuncPrecond(precond!), log=true, reltol=1e-6)

  if history.isconverged
    println("GMRES converged in $(history.iters) iterations.")
  else
    println("GMRES did not converge in $(history.iters) iterations.")
  end

  # Unpack and convert back to physical space
  # U is in body frame — rotate translational part back to world frame
  f̃_sol, U_sol = unpack(sol)
  f_sol = [ipsh(f̃_sol[i], Ω) ./ Ω.w for i in 1 : N_body]
  U_sol_rb = [vcat(collect(rotate(real(U_sol[i][1]), real(U_sol[i][2]), θcm[i])),
                   real(U_sol[i][3])) for i in 1 : N_body]

  return f_sol, U_sol_rb

end

export suspension_mobility_solve, suspension_velocity_eval, suspension_velocity_solve
  
𝒮_st(xcm, θcm, f_body::Vector, Ω; eps=1e-6) = suspension_velocity_eval(xcm, θcm, f_body, Ω; eps=eps)
𝒮_st⁻¹(xcm, θcm, u_body::Vector, Ω; eps=1e-6) = suspension_velocity_solve(xcm, θcm, u_body, Ω; eps=eps)
