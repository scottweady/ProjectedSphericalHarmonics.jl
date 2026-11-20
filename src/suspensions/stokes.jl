
using FMM3D

rotate(ux, uy, θ) = (cos(θ) .* ux .- sin(θ) .* uy,
                     sin(θ) .* ux .+ cos(θ) .* uy)

include("close_eval.jl")
include("ewald.jl")

"World-frame collocation points of every body, stacked as columns of `sources` (3 × N_body·Nz)."
function suspension_sources!(sources, xcm, θcm, Ω)
  Nz = length(Ω.z)
  x, y = real.(vec(Ω.z)), imag.(vec(Ω.z))
  for j in eachindex(xcm)
    idx = (j-1)*Nz .+ (1:Nz)
    rx, ry = rotate(x, y, θcm[j])
    sources[1, idx] = real(xcm[j]) .+ rx
    sources[2, idx] = imag(xcm[j]) .+ ry
  end
  return sources
end

"""
    suspension_self_matrix(Ω)

Within-body block of the fast sum's node quadrature, in the body frame: the 2Nz × 2Nz matrix
taking a body's node stoklets (fx·dz; fy·dz) to the velocities at its own nodes, i = j
excluded, in the FMM3D normalization G/(8π). Every body shares `Ω` and the Stokeslet is
rotation covariant (G(Rr) = R G(r) Rᵀ), so one matrix serves all bodies at any position and
orientation. It depends on `Ω` only: build it once and pass it to the suspension solves as
`S_self` to reuse it across time steps. Memory is 32 Nz² bytes (36 MiB at Nz = 1089).
"""
function suspension_self_matrix(Ω)
  Nz = length(Ω.z)
  x, y = real.(vec(Ω.z)), imag.(vec(Ω.z))
  S = zeros(2Nz, 2Nz)
  Threads.@threads for j in 1 : Nz
    for i in 1 : Nz
      i == j && continue
      rx, ry = x[i] - x[j], y[i] - y[j]
      ρ = hypot(rx, ry)
      a, b = 1 / (8π * ρ), 1 / (8π * ρ^3)
      S[i, j]         = a + b * rx^2
      S[i, Nz + j]    = b * rx * ry
      S[Nz + i, j]    = b * rx * ry
      S[Nz + i, Nz + j] = a + b * ry^2
    end
  end
  return S
end

"""
`S * F` threaded over row blocks with Julia threads. A multithreaded OpenBLAS call here leaves
its threads spinning and slows the OpenMP threads of FINUFFT, FMM3D and the PSH transforms
that follow (a mobility solve ran ~7× slower). Each block of S is read once and its block of
the output stays in cache.
"""
function _threaded_matmul(S::AbstractMatrix{Float64}, F::AbstractMatrix{Float64})
  m, n = size(S, 1), size(F, 2)
  size(S, 2) == size(F, 1) || throw(DimensionMismatch("S is $(size(S)), F is $(size(F))"))
  U = zeros(m, n)
  nb = 64
  Threads.@threads for r0 in 1 : nb : m
    rows = r0 : min(r0 + nb - 1, m)
    @inbounds for k in axes(S, 2), j in 1 : n
      f = F[k, j]
      @simd for r in rows
        U[r, j] = muladd(S[r, k], f, U[r, j])
      end
    end
  end
  return U
end

"""
Subtract each body's own node-quadrature contribution from the fast-sum output `pot`
(world frame): with `S_self` by one matrix product over all bodies in the body frame,
otherwise by a direct sum per body.
"""
function _subtract_self!(pot, sources, stoklets, θcm, Nz, S_self)
  N_body = length(θcm)
  if isnothing(S_self)
    for j = 1 : N_body
      idx = (j-1)*Nz .+ (1:Nz)
      pot[:, idx] .-= st3ddir(sources[:, idx], sources[:, idx]; stoklet=stoklets[:, idx], ppregt=1).pottarg
    end
    return pot
  end
  size(S_self) == (2Nz, 2Nz) || throw(DimensionMismatch("S_self is $(size(S_self)), expected $((2Nz, 2Nz)); build it with suspension_self_matrix(Ω)"))
  Fb = Matrix{Float64}(undef, 2Nz, N_body)
  @inbounds for j = 1 : N_body
    c, s = cos(θcm[j]), sin(θcm[j])
    for k = 1 : Nz
      fx, fy = stoklets[1, (j-1)*Nz + k], stoklets[2, (j-1)*Nz + k]
      Fb[k, j], Fb[Nz + k, j] = c * fx + s * fy, -s * fx + c * fy
    end
  end
  Ub = _threaded_matmul(S_self, Fb)
  @inbounds for j = 1 : N_body
    c, s = cos(θcm[j]), sin(θcm[j])
    for k = 1 : Nz
      ux, uy = Ub[k, j], Ub[Nz + k, j]
      pot[1, (j-1)*Nz + k] -= c * ux - s * uy
      pot[2, (j-1)*Nz + k] -= s * ux + c * uy
    end
  end
  return pot
end

"Plan for the cross-body sum: an `StokesEwald` for `method = :ewald`, `nothing` for `:fmm`."
function suspension_fast_plan(method, sources, eps)
  method === :ewald && return StokesEwald(sources; tol=eps)
  method === :fmm   && return nothing
  throw(ArgumentError("method must be :ewald or :fmm, got $(repr(method))"))
end

"""
    suspension_velocity_eval!(u_body, sources, stoklets, xcm, θcm, f_body, Ω; eps=1e-6, close=nothing, f̃_body=nothing, method=:ewald, ewald=nothing, S_self=nothing, μ_bottom=1.0, μ_top=0.0)

Velocity (body frame) at the collocation points of every body due to the force densities
`f_body` (body frame). Self terms are spectral; cross terms use one fast sum over the
collocation nodes (`method = :ewald`, Ewald + NUFFT, or `:fmm`, FMM3D), and near targets in
`close` (from `close_eval_plan`) are corrected with the closed form. For `:ewald`, pass a
plan `ewald` (`StokesEwald` of the current `sources`) to reuse it across calls. `S_self`
(`suspension_self_matrix(Ω)`) removes each body's own terms from the fast sum by one matrix
product; without it they are removed by a direct sum per body. `f̃_body`
are the even PSH coefficients of f·ω; they are computed from `f_body` if not given.
`μ_bottom` and `μ_top` are the viscosities below and above the plane (see `𝒮_st(f, Ω)`).
"""
function suspension_velocity_eval!(u_body, sources, stoklets, xcm, θcm, f_body, Ω; eps=1e-6, close=nothing, f̃_body=nothing, method=:ewald, ewald=nothing, S_self=nothing, μ_bottom=1.0, μ_top=0.0)

  μ = μ_bottom + μ_top
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
    ui_body = 𝒮_st(f_body[i], Ω; μ_bottom=μ_bottom, μ_top=μ_top)
    u_body[i][1] .+= ui_body[1]
    u_body[i][2] .+= ui_body[2]
  end
  N_body == 1 && return u_body       # no cross terms

  # Off-diagonal interactions via one fast sum over all nodes
  # Sources and stoklets are in world frame: rotate body-frame quantities by θcm[i]
  suspension_sources!(sources, xcm, θcm, Ω)
  dz = real.(vec(Ω.dz))

  for j = 1 : N_body
    idx  = (j-1)*Nz .+ (1:Nz)
    # Rotate body-frame force to world frame; area element Ω.dz is a real scalar
    fx_world, fy_world = rotate(real.(vec(f_body[j][1])), real.(vec(f_body[j][2])), θcm[j])
    stoklets[1, idx] = fx_world .* dz
    stoklets[2, idx] = fy_world .* dz
  end

  # Evaluate at all source locations, i = j excluded, in the FMM3D normalization G/(8π)
  if method === :fmm
    pot = stfmm3d(eps, sources; stoklet=stoklets, ppreg=1).pot
  else
    E = isnothing(ewald) ? suspension_fast_plan(method, sources, eps) : ewald
    pot = stokes_ewald!(similar(sources), E, stoklets)
  end

  # Subtract within-body contributions: the fast sum includes them but 𝒮_st already handles them
  _subtract_self!(pot, sources, stoklets, θcm, Nz, S_self)

  # Close evaluation: replace node quadrature at targets near another body
  if !isnothing(close) && !isempty(close)
    f̃ = isnothing(f̃_body) ? [psh(f_body[j] .* Ω.w, Ω) for j in 1 : N_body] : f̃_body
    close_eval_correct!(pot, sources, stoklets, close, f̃, θcm, Ω)
  end

  # Fast sum uses G/(8π) with unit viscosity; 𝒮_st uses G/(4π μ) with μ = μ_bottom + μ_top, so multiply by 2/μ
  # Rotate world-frame output back to body frame of each target body
  for i = 1 : N_body
    idx  = (i-1)*Nz .+ (1:Nz)
    ux_world = (2 / μ) .* reshape(pot[1, idx], shp)
    uy_world = (2 / μ) .* reshape(pot[2, idx], shp)
    du1, du2 = rotate(ux_world, uy_world, -θcm[i])
    u_body[i][1] .+= du1
    u_body[i][2] .+= du2
  end

  return u_body

end

"""
    suspension_velocity_eval(xcm, θcm, f_body, Ω; eps=1e-6, close_eval=true, close_radius=CLOSE_RADIUS, method=:ewald, S_self=nothing, μ_bottom=1.0, μ_top=0.0)

Velocity at the collocation points of every body due to the force densities `f_body` of all of
them: the forward map of `suspension_velocity_solve`. Body `i` is `Ω` rotated by `θcm[i]` and
centered at `xcm[i]`; `f_body[i]` is a tuple `(fx, fy)` on `Ω`'s grid in the body frame, and the
result is a vector of such tuples of velocities, also in each body's frame. Self terms are
spectral (`𝒮_st`); cross terms are one fast sum (`method = :ewald` or `:fmm`, tolerance `eps`),
corrected by the closed form at targets near another disk or ellipse unless `close_eval = false`.
See `suspension_mobility_solve` for the other keywords.
"""
function suspension_velocity_eval(xcm, θcm, f_body::Vector, Ω; eps=1e-6, close_eval=true, close_radius=CLOSE_RADIUS, method=:ewald, S_self=nothing, μ_bottom=1.0, μ_top=0.0)
  close = close_eval ? close_eval_plan(xcm, θcm, Ω; close_radius=close_radius) : nothing
  N_body   = length(f_body)
  Nz  = length(Ω.z)
  shp = size(Ω.z)
  u_body = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1 : N_body]
  sources  = zeros(Float64, 3, N_body * Nz)
  stoklets = zeros(Float64, 3, N_body * Nz)
  return suspension_velocity_eval!(u_body, sources, stoklets, xcm, θcm, f_body, Ω; eps=eps, close=close, method=method, S_self=S_self, μ_bottom=μ_bottom, μ_top=μ_top)
end

"""
    suspension_velocity_solve(xcm, θcm, u_body, Ω; eps=1e-6, method=:ewald, S_self=nothing, μ_bottom=1.0, μ_top=0.0)

Solve the Stokes suspension problem: given collocation-space velocities u_body, find
collocation-space forces f_body such that stokes_suspension(xcm, f_body, Ω) ≈ u_body.

Uses GMRES with the forward operator in PSH coefficient space via the substitution
f = f̃/Ω.w to improve conditioning.

# Arguments
- `xcm` : vector of N particle center positions
- `θcm` : vector of N particle orientations (radians)
- `u_body`   : vector of N tuples of velocity matrices (ux, uy) at collocation points
- `Ω`   : disk discretization
- `close_eval`   : correct cross terms between nearby bodies with the closed form (ellipses and disks)
- `close_radius` : near-target radius for close evaluation (see `CLOSE_RADIUS`)
- `method` : fast sum for the cross terms, `:ewald` (Ewald + NUFFT, default) or `:fmm` (FMM3D);
        `eps` is its tolerance
- `S_self` : `suspension_self_matrix(Ω)`, built here if not given; it depends on `Ω` only, so
        a simulation can build it once and pass it at every step
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- Vector of N tuples of force matrices (fx, fy) at collocation points
"""
function suspension_velocity_solve(xcm, θcm, u_body::Vector, Ω; eps=1e-6, gmrestol=1e-6, close_eval=true, close_radius=CLOSE_RADIUS, method=:ewald, S_self=nothing, μ_bottom=1.0, μ_top=0.0)

  N_body   = length(u_body)
  Nz  = length(Ω.z)
  shp = size(Ω.z)

  # Pre-allocate buffers reused across GMRES iterations
  ũ_buf    = [(zeros(ComplexF64, shp), zeros(ComplexF64, shp)) for _ in 1 : N_body]
  sources  = zeros(Float64, 3, N_body * Nz)
  stoklets = zeros(Float64, 3, N_body * Nz)

  # Closed-form maps for nearby bodies, and the fast-sum plan (geometry only)
  close = close_eval ? close_eval_plan(xcm, θcm, Ω; close_radius=close_radius) : nothing
  ewald = suspension_fast_plan(method, suspension_sources!(sources, xcm, θcm, Ω), eps)
  S_self = (isnothing(S_self) && N_body > 1) ? suspension_self_matrix(Ω) : S_self

  pack(V̂) = vcat([vcat(vec(V̂[i][1]), vec(V̂[i][2])) for i in 1 : N_body]...)

  function unpack(v)
    [(reshape(v[(2(i-1))*Nz .+ (1:Nz)], shp),
      reshape(v[(2i-1)*Nz   .+ (1:Nz)], shp)) for i in 1 : N_body]
  end

  function A!(result, v)
    f̃ = unpack(v)
    f  = [fi ./ Ω.w for fi in ipsh.(f̃, Ref(Ω))]
    suspension_velocity_eval!(ũ_buf, sources, stoklets, xcm, θcm, f, Ω; eps=eps, close=close, f̃_body=f̃, method=method, ewald=ewald, S_self=S_self, μ_bottom=μ_bottom, μ_top=μ_top)
    result .= pack(psh.(ũ_buf, Ref(Ω)))
  end

  û  = psh.(u_body, Ref(Ω))
  F̃  = unpack(gmres(A!, pack(û); reltol=gmrestol))
  return [fi ./ Ω.w for fi in ipsh.(F̃, Ref(Ω))]

end

# IterativeSolvers.gmres! requires ldiv!(Pl, x); this wraps a function f!(out, in)
struct _FuncPrecond f!::Function end
LinearAlgebra.ldiv!(P::_FuncPrecond, x::AbstractVector) = (tmp = copy(x); P.f!(x, tmp); x)

"""
    suspension_mobility_solve(xcm, θcm, uinf, uslip, F, Ω; eps=1e-6, M=[], S_self=nothing, method=:ewald, μ_bottom=1.0, μ_top=0.0)

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
- `M`     : precomputed single-body mobility matrix (optional; must be computed with the same viscosities)
- `close_eval`   : correct cross terms between nearby bodies with the closed form (ellipses and disks)
- `close_radius` : near-target radius for close evaluation (see `CLOSE_RADIUS`)
- `method` : fast sum for the cross terms, `:ewald` (Ewald + NUFFT, default) or `:fmm` (FMM3D);
        `eps` is its tolerance
- `S_self` : `suspension_self_matrix(Ω)`, built here if not given; it depends on `Ω` only, so
        a simulation can build it once and pass it at every step
- `μ_bottom`, `μ_top` : viscosities below and above the plane (default: 1.0, 0.0, a half-space
        under a stress-free surface; set both equal for free space)

# Returns
- `f` : vector of N force density tuples (fx, fy)
- `U` : vector of N rigid body velocity vectors [Ux, Uy, ω]
"""
function suspension_mobility_solve(xcm, θcm, uinf, uslip, F::Vector, Ω; eps=1e-6, gmrestol=1e-6, M=[], S_self=nothing, verbose=false, close_eval=true, close_radius=CLOSE_RADIUS, method=:ewald, μ_bottom=1.0, μ_top=0.0)

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
  M_mob = isempty(M) ? stokes_mobility_matrix(Ω; μ_bottom=μ_bottom, μ_top=μ_top) : M

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

  # Closed-form maps for nearby bodies, and the fast-sum plan (geometry only)
  close = close_eval ? close_eval_plan(xcm, θcm, Ω; close_radius=close_radius) : nothing
  ewald = suspension_fast_plan(method, suspension_sources!(sources, xcm, θcm, Ω), eps)
  S_self = (isnothing(S_self) && N_body > 1) ? suspension_self_matrix(Ω) : S_self

  function matvec!(result, v)
    f̃, U = unpack(v)
    f = [ipsh(f̃[i], Ω) ./ Ω.w for i in 1 : N_body]

    # S·f: self via spectral (inside FMM call), cross via FMM
    suspension_velocity_eval!(u_buf, sources, stoklets, xcm, θcm, f, Ω; eps=eps, close=close, f̃_body=f̃, method=method, ewald=ewald, S_self=S_self, μ_bottom=μ_bottom, μ_top=μ_top)

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

  # Block-diagonal preconditioner: the single-body mobility problem per body, in even PSH
  # coefficients (psh ∘ ipsh is the identity on them). With g = K⁻¹ r̃ for the residual r̃
  # read as a velocity, the single-body solve gives
  #   R = M⁻¹ (rU − μ B g),   δf̃ = μ g + Σₖ Rₖ h̃ₖ,   h̃ₖ = μ K⁻¹ psh(eₖ),
  # with B g the force and torque of the density ipsh(g)/w and e = (1,0), (0,1), (−y,x) the
  # rigid motions. K is factored once per solve (once per domain for a `Domain`).
  μ    = μ_bottom + μ_top
  ieven = findall(Ω.even)
  Ne   = length(ieven)
  K    = Ω.K̂_G
  Klu  = _factorization(Ω, K)
  evenvec(û) = vcat(û[1][ieven], û[2][ieven])
  H    = μ .* (Klu \ hcat(evenvec(psh((ones(shp), zeros(shp)), Ω)),
                         evenvec(psh((zeros(shp), ones(shp)), Ω)),
                         evenvec(psh((-y, x), Ω))))

  function precond!(result, v)
    f̃, rU = unpack(v)

    new_f̃ = Vector{Tuple{Matrix{ComplexF64}, Matrix{ComplexF64}}}(undef, N_body)
    new_U  = Vector{Vector{ComplexF64}}(undef, N_body)

    for i in 1 : N_body
      g  = Klu \ evenvec(f̃[i])
      ĝ1 = zeros(ComplexF64, shp); ĝ1[ieven] .= @view g[1:Ne]
      ĝ2 = zeros(ComplexF64, shp); ĝ2[ieven] .= @view g[Ne+1:end]
      g1, g2 = ipsh(ĝ1, Ω) ./ Ω.w, ipsh(ĝ2, Ω) ./ Ω.w
      Bg = [integral(g1, Ω), integral(g2, Ω), integral(-y .* g1 .+ x .* g2, Ω)]

      R = M_mob \ (real.(rU[i]) .- μ .* Bg)
      c = μ .* g .+ H * R

      δ1 = zeros(ComplexF64, shp); δ1[ieven] .= @view c[1:Ne]
      δ2 = zeros(ComplexF64, shp); δ2[ieven] .= @view c[Ne+1:end]
      new_f̃[i] = (δ1, δ2)
      new_U[i]  = R
    end

    result .= pack(new_f̃, new_U)
  end

  # RHS: system is S·f - A·U = -uinf (boundary condition S·f + uinf = A·U)
  # uinf_eval and F_body are already in body frame
  rhs = pack([psh(uslip_eval[i] .- uinf_eval[i], Ω) for i in 1 : N_body], F_body)

  # GMRES with block-diagonal preconditioner
  sol = zeros(ComplexF64, Ntotal)
  op  = LinearOperator(ComplexF64, Ntotal, Ntotal, false, false, matvec!)
  sol, history = gmres!(sol, op, rhs; Pl=_FuncPrecond(precond!), log=true, reltol=gmrestol)

  if verbose
    if history.isconverged
      println("GMRES converged in $(history.iters) iterations.")
    else
      println("GMRES did not converge in $(history.iters) iterations.")
    end
  end

  # Unpack and convert back to physical space
  # U is in body frame — rotate translational part back to world frame
  f̃_sol, U_sol = unpack(sol)
  f_sol = [ipsh(f̃_sol[i], Ω) ./ Ω.w for i in 1 : N_body]
  U_sol_rb = [vcat(collect(rotate(real(U_sol[i][1]), real(U_sol[i][2]), θcm[i])),
                   real(U_sol[i][3])) for i in 1 : N_body]

  return f_sol, U_sol_rb

end

export suspension_mobility_solve, suspension_velocity_eval, suspension_velocity_solve, suspension_self_matrix
  
𝒮_st(xcm, θcm, f_body::Vector, Ω; eps=1e-6, method=:ewald, μ_bottom=1.0, μ_top=0.0) = suspension_velocity_eval(xcm, θcm, f_body, Ω; eps=eps, method=method, μ_bottom=μ_bottom, μ_top=μ_top)
𝒮_st⁻¹(xcm, θcm, u_body::Vector, Ω; eps=1e-6, gmrestol=1e-6, method=:ewald, μ_bottom=1.0, μ_top=0.0) = suspension_velocity_solve(xcm, θcm, u_body, Ω; eps=eps, gmrestol=gmrestol, method=method, μ_bottom=μ_bottom, μ_top=μ_top)
