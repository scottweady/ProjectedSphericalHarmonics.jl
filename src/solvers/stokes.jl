
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