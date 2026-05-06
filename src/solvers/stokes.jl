
using FMM3D

function stokes3d_single_layer_fmm!(u, sources, stoklets, xcm, f, D; eps=1e-6)

  N   = length(f)
  Nζ  = length(D.ζ)
  shp = size(D.ζ)

  # Zero output
  for i = 1:N
    fill!(u[i][1], 0)
    fill!(u[i][2], 0)
  end

  # Self-interaction via PSH operator
  for i = 1:N
    ui = 𝒮_st(f[i], D)
    u[i][1] .+= ui[1]
    u[i][2] .+= ui[2]
  end

  # Off-diagonal interactions via FMM3D
  # Embed 2D disk in 3D (z = 0); stoklet strength = force × area element dζ

  for j = 1:N
    idx = (j-1)*Nζ .+ (1:Nζ)
    ζ_j = vec(D.ζ) .+ xcm[j]
    sources[1, idx]  = real.(ζ_j)
    sources[2, idx]  = imag.(ζ_j)
    stoklets[1, idx] = real.(vec(f[j][1]) .* vec(D.dζ))
    stoklets[2, idx] = real.(vec(f[j][2]) .* vec(D.dζ))
  end

  # One FMM call: evaluate at all source locations (ppreg=1 excludes exact self-term)
  out = stfmm3d(eps, sources; stoklet=stoklets, ppreg=1)

  # Subtract within-disk contributions: FMM includes them but 𝒮_st already handles them
  for j = 1:N
    idx = (j-1)*Nζ .+ (1:Nζ)
    self = st3ddir(sources[:, idx], sources[:, idx];
                   stoklet=stoklets[:, idx], ppregt=1)
    out.pot[:, idx] .-= self.pottarg
  end

  # FMM uses G/(8π), stokeslet_direct uses G/(4π), so multiply by 2
  for i = 1:N
    idx = (i-1)*Nζ .+ (1:Nζ)
    u[i][1] .+= 2 .* reshape(out.pot[1, idx], shp)
    u[i][2] .+= 2 .* reshape(out.pot[2, idx], shp)
  end

  return u

end

function stokes3d_single_layer_fmm(xcm, f::Vector, D::Disk; eps=1e-6)
  N   = length(f)
  Nζ  = length(D.ζ)
  shp = size(D.ζ)
  u        = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1:N]
  sources  = zeros(Float64, 3, N * Nζ)
  stoklets = zeros(Float64, 3, N * Nζ)
  return stokes3d_single_layer_fmm!(u, sources, stoklets, xcm, f, D; eps=eps)
end

"""
    stokes3d_single_layer_fmm_solve(xcm, u, D; eps=1e-6)

Solve the Stokes suspension problem: given collocation-space velocities u, find
collocation-space forces f such that stokes_suspension(xcm, f, D) ≈ u.

Uses GMRES with the forward operator in PSH coefficient space via the substitution
f = f̃/D.w to improve conditioning.

# Arguments
- `xcm` : vector of N particle center positions
- `u`   : vector of N tuples of velocity matrices (ux, uy) at collocation points
- `D`   : disk discretization

# Returns
- Vector of N tuples of force matrices (fx, fy) at collocation points
"""
function stokes3d_single_layer_fmm_solve(xcm, u::Vector, D::Disk; eps=1e-6)

  N   = length(u)
  Nζ  = length(D.ζ)
  shp = size(D.ζ)

  # Pre-allocate buffers reused across GMRES iterations
  ũ_buf    = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1:N]
  sources  = zeros(Float64, 3, N * Nζ)
  stoklets = zeros(Float64, 3, N * Nζ)

  pack(V̂) = vcat([vcat(vec(V̂[i][1]), vec(V̂[i][2])) for i in 1:N]...)

  function unpack(v)
    [(reshape(v[(2(i-1))*Nζ .+ (1:Nζ)], shp),
      reshape(v[(2i-1)*Nζ   .+ (1:Nζ)], shp)) for i in 1:N]
  end

  function A!(result, v)
    f̃ = unpack(v)
    f  = [fi ./ D.w for fi in ipsh.(f̃, Ref(D))]
    stokes3d_single_layer_fmm!(ũ_buf, sources, stoklets, xcm, f, D; eps=eps)
    result .= pack(psh.(ũ_buf, Ref(D)))
  end

  û  = psh.(u, Ref(D))
  F̃  = unpack(gmres(A!, pack(û)))
  return [fi ./ D.w for fi in ipsh.(F̃, Ref(D))]

end

𝒮_st(xcm, f::Vector, D::Disk; eps=1e-6) = stokes3d_single_layer_fmm(xcm, f, D; eps=eps)
𝒮_st⁻¹(xcm, u::Vector, D::Disk; eps=1e-6) = stokes3d_single_layer_fmm_solve(xcm, u, D; eps=eps)

# IterativeSolvers.gmres! requires ldiv!(Pl, x); this wraps a function f!(out, in)
struct _FuncPrecond f!::Function end
LinearAlgebra.ldiv!(P::_FuncPrecond, x::AbstractVector) = (tmp = copy(x); P.f!(x, tmp); x)

export stokes_mobility_fmm_solve, stokes3d_single_layer_fmm

"""
    stokes_mobility_fmm_solve(xcm, uinf, uslip, F, D; eps=1e-6)

Solve the N-body Stokes mobility problem via GMRES.

The boundary condition is `uinf + S·f = A·U + uslip`, giving the linear system
`[S, -A; B, 0] [f; U] = [uslip - uinf; F]`. For rigid particles with no prescribed
slip, pass `uslip = nothing` (or a zero function). For squirmers, `uslip` is the
prescribed surface slip velocity.

Both `uinf` and `uslip` can be functions `z -> (ux, uy)` evaluated at the actual body
positions `D.z .+ xcm[i]`, or pre-evaluated vectors of N tuples.

Preconditioned by independent single-body mobility solves (block-diagonal).

# Arguments
- `xcm`   : vector of N complex center positions
- `uinf`  : background flow as a function z -> (ux, uy), or vector of N tuples
- `uslip` : prescribed slip velocity (same form as uinf), or nothing for no-slip
- `F`     : vector of N vectors [Fx, Fy, T] (forces and torques)
- `D`     : disk discretization (shared by all bodies)

# Returns
- `f` : vector of N force density tuples (fx, fy)
- `U` : vector of N rigid body velocity vectors [Ux, Uy, ω]
"""
function stokes_mobility_fmm_solve(xcm, uinf, uslip, F::Vector, D::Disk; eps=1e-6)

  N   = length(xcm)
  Nζ  = length(D.ζ)
  shp = size(D.ζ)
  xg  = real.(D.z)
  yg  = imag.(D.z)

  eval_field(u) = u isa Function  ? [u(D.z .+ xcm[i]) for i in 1:N] :
                  u === nothing   ? [(zeros(shp), zeros(shp)) for _ in 1:N] : u
  uinf_eval  = eval_field(uinf)
  uslip_eval = eval_field(uslip)

  # Precompute single-body mobility matrix (same disk for all bodies)
  M_mob = stokes_mobility_matrix(D)

  # Packing layout: [f̃_1_x; f̃_1_y; ...; f̃_N_x; f̃_N_y; U_1; ...; U_N]
  # where f̃_i = psh(f_i .* D.w, D) so that f_i = ipsh(f̃_i, D) ./ D.w
  Nfield = 2N * Nζ
  Ntotal = Nfield + 3N

  function pack(f̃, U)
    v = zeros(ComplexF64, Ntotal)
    for i in 1:N
      v[(2(i-1))*Nζ .+ (1:Nζ)] = vec(f̃[i][1])
      v[(2i-1)*Nζ   .+ (1:Nζ)] = vec(f̃[i][2])
    end
    for i in 1:N
      v[Nfield + 3(i-1) + 1] = U[i][1]
      v[Nfield + 3(i-1) + 2] = U[i][2]
      v[Nfield + 3(i-1) + 3] = U[i][3]
    end
    return v
  end

  function unpack(v)
    f̃ = [(reshape(v[(2(i-1))*Nζ .+ (1:Nζ)], shp),
           reshape(v[(2i-1)*Nζ   .+ (1:Nζ)], shp)) for i in 1:N]
    U  = [[v[Nfield + 3(i-1) + 1],
            v[Nfield + 3(i-1) + 2],
            v[Nfield + 3(i-1) + 3]] for i in 1:N]
    return f̃, U
  end

  # Pre-allocated FMM buffers
  u_buf    = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1:N]
  sources  = zeros(Float64, 3, N * Nζ)
  stoklets = zeros(Float64, 3, N * Nζ)

  function matvec!(result, v)
    f̃, U = unpack(v)
    f = [ipsh(f̃[i], D) ./ D.w for i in 1:N]

    # S·f: self via spectral (inside FMM call), cross via FMM
    stokes3d_single_layer_fmm!(u_buf, sources, stoklets, xcm, f, D; eps=eps)

    # Subtract A·U: rigid body velocity field on each body
    for i in 1:N
      Ux, Uy, ω = real(U[i][1]), real(U[i][2]), real(U[i][3])
      u_buf[i][1] .-= Ux .- ω .* yg
      u_buf[i][2] .-= Uy .+ ω .* xg
    end

    # Field residual in PSH coefficient space
    f̃_out = [psh(u_buf[i], D) for i in 1:N]

    # B·f: force/torque integrals
    U_out = [[integral(f[i][1], D),
               integral(f[i][2], D),
               integral(-yg .* f[i][1] .+ xg .* f[i][2], D)] for i in 1:N]

    result .= pack(f̃_out, U_out)
  end

  function precond!(result, v)
    f̃, rU = unpack(v)
    # Interpret PSH coefficients as velocity fields for the preconditioner
    rf = [ipsh(f̃[i], D) for i in 1:N]

    new_f̃ = Vector{Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(undef, N)
    new_U  = Vector{Vector{ComplexF64}}(undef, N)

    for i in 1:N
      # stokes_mobility_solve uses convention S·f = A·U - uinf, so pass -rf
      δU, δω = stokes_mobility_solve(.-rf[i],
                                      (real(rU[i][1]), real(rU[i][2])),
                                      real(rU[i][3]), D; M=M_mob)

      # Velocity that δf must produce: S·δf = A·δU - (-rf) = A·δU + rf
      ũ = (rf[i][1] .+ δU[1] .- δω .* yg,
           rf[i][2] .+ δU[2] .+ δω .* xg)

      # Recover force density via single-body inverse
      δf = 𝒮_st⁻¹(ũ, D)

      # Store PSH coefficients of δf·w (the packed representation)
      new_f̃[i] = psh(δf .* D.w, D)
      new_U[i]  = [δU[1], δU[2], δω]
    end

    result .= pack(new_f̃, new_U)
  end

  # RHS: system is S·f - A·U = -uinf (boundary condition S·f + uinf = A·U)
  rhs = pack([psh(uslip_eval[i] .- uinf_eval[i], D) for i in 1:N], F)

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
  f̃_sol, U_sol = unpack(sol)
  f_sol = [ipsh(f̃_sol[i], D) ./ D.w for i in 1:N]
  U_sol_rb = [[real(U_sol[i][1]), real(U_sol[i][2]), real(U_sol[i][3])] for i in 1:N]

  return f_sol, U_sol_rb

end