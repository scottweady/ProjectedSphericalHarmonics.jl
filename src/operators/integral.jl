
"""
    wrap_operator(name, operator!)

Macro to wrap integral operators. 
  
For a given operator `operator!`, it defines two functions: one that takes a 
matrix input and another that takes a scalar input, which is then broadcasted 
to match the size of the disk discretization.

"""
macro wrap_operator(name, operator!)
  name! = Symbol(name, :!)
  quote
    function $(esc(name!))(u::AbstractArray{ComplexF64}, D::Disk)
      return $(esc(operator!))(u, D)
    end
    function $(esc(name))(u, D::Disk)
      if isa(u, Number)
        u = fill(u, size(D.ζ))
      end
      return $(esc(operator!))(ComplexF64.(u), D)
    end
  end
end

"""
    laplace3d_single_layer!(u, D)

Single layer of 3D Laplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk  
"""
function laplace3d_single_layer!(u::AbstractMatrix{ComplexF64}, D::Disk)

  u .*= D.w
  psh!(u, D, parity=:even)
  u .*= D.Ŝ
  ipsh!(u, D, parity=:even)
  return u

end

"""
    laplace3d_single_layer_inverse!(f, D)

Inverse of 𝒮

# Arguments
- `f` : single layer potential on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk  
"""
function laplace3d_single_layer_inverse!(f::AbstractMatrix{ComplexF64}, D::Disk)

  psh!(f, D, parity=:even)
  f ./= D.Ŝ
  ipsh!(f, D, parity=:even)
  f ./= D.w
  return f

end

"""
    laplace3d_hypersingular(u, D)

Hypersingular operator

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- hypersingular operator evaluated on the disk
"""
function laplace3d_hypersingular!(u::AbstractMatrix{ComplexF64}, D::Disk)

  psh!(u, D, parity=:odd)
  u .*= D.N̂
  ipsh!(u, D, parity=:odd)
  u ./= D.w
  return u

end

"""
    laplace3d_hypersingular_inverse!(f, D::Disk)

Inverse of 𝒩

# Arguments
- `f` : hypersingular operator on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk
"""
function laplace3d_hypersingular_inverse!(f::AbstractMatrix{ComplexF64}, D::Disk)

  f .*= D.w
  psh!(f, D, parity=:odd)
  f ./= D.N̂
  ipsh!(f, D, parity=:odd)
  return f
  
end

"""
    laplace2d_volume(u, D)
    
Volume potential of 2D Laplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- volume potential evaluated on the disk

"""
function laplace2d_volume(u, D::Disk)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

  ζ, dζ = D.ζ, D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)

  δζ = ζ .- transpose(ζ)
  V = (1 / 2π) * log.(abs.(δζ) .+ (δζ .== 0)) .* dζ';
  Vu = V * u .+ ((abs2.(ζ) .- 1) / 4 .- sum(V, dims=2)) .* u

  return reshape(Vu, shp)

end

𝒱(u, D::Disk) = laplace2d_volume(u, D)

"""
    bilaplace2d_volume(u, D; κ=0)

Volume potential of 2D Bilaplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- volume potential evaluated on the disk
"""
function bilaplace2d_volume(u, D::Disk; κ=0)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ, dζ = D.ζ, D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)

  δζ = ζ .- transpose(ζ)
	B = (1 / 8π) * abs2.(δζ) .* (log.(abs.(δζ) .+ (δζ .== 0)) .- 1 .+ κ) .* dζ';
  Bu = B * u

  return reshape(Bu, shp)

end

ℬ(u, D::Disk; κ=0) = bilaplace2d_volume(u, D; κ=κ)

"""
    bilaplace3d_single_layer(u, D)

    Single layer of 3D Bilaplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk
"""
function bilaplace3d_single_layer(u, D::Disk)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ, dζ = D.ζ, D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)

  δζ = ζ .- transpose(ζ)
	T = (1 / 8π) * abs.(δζ) .* dζ';
  Tu = T * u

  return reshape(Tu, shp)

end

𝒯(u, D::Disk) = bilaplace3d_single_layer(u, D)

"""
    𝒮𝒩⁻¹(u, D)

Composition of 𝒮 and 𝒩⁻¹

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- 𝒮𝒩⁻¹ evaluated on the disk
"""
function 𝒮𝒩⁻¹(u, D::Disk)
    û = psh(u, D)
    f̂ = apply(D.ŜN̂⁻¹, û, D)
    return ipsh(f̂, D, parity=:even)
end

"""
    𝒮_st(f, D; η=1.0)

Single layer operator for the half-space Stokes equations

# Arguments
- `f` : density function on the disk
- `D` : discretization of the disk
- `η` : viscosity (default: 1.0)

# Returns
- single layer potential evaluated on the disk
"""
function 𝒮_st(f::Tuple, D::Disk; η=1.0)

  if isa(f[1], Number)
    f = (fill(f[1], size(D.ζ)), fill(f[2], size(D.ζ)))
  end

  f̂ = psh(f .* D.w, D, parity=:even)
  û = apply(D.Ĝ, f̂, D) ./ η
  return ipsh(û, D, parity=:even)
  
end

function 𝒮_st(i::Int, j::Int, f::AbstractMatrix{ComplexF64}, D; η=1.0)
  f̂ = psh(f .* D.w, D, parity=:even)
  û = apply(D.Ĝ[i, j], f̂, D) ./ η
  return ipsh(û, D, parity=:even)
end

"""
    𝒮_st⁻¹(u, D; η=1.0)

Inverse of the single layer operator for the half-space Stokes equations

# Arguments
- `u` : velocity field on the disk
- `D` : discretization of the disk
- `η` : viscosity (default: 1.0)

# Returns
- density function (force) on the disk
"""
function 𝒮_st⁻¹(u::Tuple, D::Disk; η=1.0)

  if isa(u[1], Number)
    u = (fill(u[1], size(D.ζ)), fill(u[2], size(D.ζ)))
  end

  û = psh(u, D, parity=:even)
  f̂ = solve(D.Ĝ, û, D) .* η
  return ipsh(f̂, D, parity=:even) ./ D.w
  
end

function stokes_mobility_matrix(D::Disk)
  return [16/3 0 0; 0 16/3 0; 0 0 16/3]
end

function stokes_mobility_matrix(Ω::Domain)  

  x, y = real.(Ω.z), imag.(Ω.z)

  e1 = (1.0, 0.0)
  e2 = (0.0, 1.0)
  e3 = (-y, x)
  E = (e1, e2, e3)

  M = zeros(ComplexF64, 3, 3)

  for (i, e) in enumerate(E)
    f = 𝒮_st⁻¹(e, Ω)
    M[i,1] = integral(f[1], Ω)
    M[i,2] = integral(f[2], Ω)
    M[i,3] = integral(-y .* f[1] + x .* f[2], Ω)
  end

  return M

end

function stokes_mobility_solve(uinf, F, T, D::Disk; M=[])
  
  if isempty(M)
    M = stokes_mobility_matrix(D)
  end

  # Form righthand side
  û = psh(uinf, D)
  f̂ = solve(D.Ĝ, û, D)
  
  # Compute contributions from the background flow
  Finf1 = f̂[1][1,1] / Nlm(0,0)
  Finf2 = f̂[2][1,1] / Nlm(0,0)
  Tinf = (-imag.(f̂[1][2,end]) + real.(f̂[2][2,end])) / (-Nlm(1, 1))

  # Add contributions from the background flow to the force and torque
  F1, F2 = F[1] .+ Finf1, F[2] .+ Finf2
  T = T .+ Tinf

  # Solve for the velocity and angular velocity
  R = M \ [F1; F2; T]
  U, ω = (R[1], R[2]), R[3]

  return U, ω

end

function stokes_mobility_solve(uinf, F, T, Ω::Domain; M=[])
  
  if isempty(M)
    M = stokes_mobility_matrix(Ω)
  end

  # Get grid points
  x, y = real.(Ω.z), imag.(Ω.z)

  # Form righthand side
  finf = 𝒮_st⁻¹(uinf, Ω)

  # Compute contributions from the background flow
  Finf1 = integral(finf[1], Ω)
  Finf2 = integral(finf[2], Ω)
  Tinf = integral(-y .* finf[1] + x .* finf[2], Ω)
  
  # Add contributions from the background flow to the force and torque
  F1, F2 = F[1] .+ Finf1, F[2] .+ Finf2
  T = T .+ Tinf

  # Solve for the velocity and angular velocity
  R = M \ [F1; F2; T]
  U, ω = (R[1], R[2]), R[3]

  return U, ω

end

export stokes_mobility_matrix, stokes_mobility_solve

@wrap_operator 𝒮 laplace3d_single_layer!
@wrap_operator 𝒮⁻¹ laplace3d_single_layer_inverse!
@wrap_operator 𝒩 laplace3d_hypersingular!
@wrap_operator 𝒩⁻¹ laplace3d_hypersingular_inverse!
