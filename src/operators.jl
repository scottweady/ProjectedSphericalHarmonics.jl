
using FFTW

import Base: div

"""
    trace(u, D)

Evaluate function on boundary of disk

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- function evaluated on the boundary of the disk
"""
function trace(u, D)
    û = psh(u, D)
    return ipsh(û, D, [1.0], parity=:even)
end

function trace(u::Tuple, D)
    return (trace(u[1], D), trace(u[2], D))
end

"""
    ∂n(u, D)

Normal derivative operator

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- normal derivative of function on the disk

"""
function ∂n(u, D)

  # Even expansion of u
  û = psh(u, D; parity=:even)

  # Apply normal derivative operator in coefficient space
  ∂û∂n = sum(D.even .* (D.∂n̂ .* û), dims=1)

  # Inverse transform to physical space
  ∂u∂n = ifft(∂û∂n) * length(∂û∂n)

  return ∂u∂n

end

"""
    ∂r(u, D)

Radial derivative

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- radial derivative of function on the disk

Warning: Ill-conditioned, use with caution.
"""
function ∂r(u, D; tol=1e-8)
  shp = size(u)
  û = psh(u, D, parity=:even)
  û[abs.(û) .< tol] .= 0.0
  return reshape(D.∂Y∂r * vec(û), shp)
end

"""
    ∂θ(u, D)

Angular derivative

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- angular derivative of function on the disk
"""
function ∂θ(u, D)
  û = psh(u, D, parity=:even)
  ∂û∂θ = D.∂θ̂ .* û
  return ipsh(∂û∂θ, D, parity=:even)
end

"""
    grad(u, D)

Gradient operator

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- tuple `(ux, uy)` of x- and y- derivatives of function on the disk

Warning: Ill-conditioned, use with caution.
"""
function grad(u, D; parity=:even)

  r, θ = abs.(D.ζ), angle.(D.ζ)
  ∂u∂r, ∂u∂θ = ∂r(u, D), ∂θ(u, D)

  ∂u∂x = cos.(θ) .* ∂u∂r .- (sin.(θ) ./ r) .* ∂u∂θ
  ∂u∂y = sin.(θ) .* ∂u∂r .+ (cos.(θ) ./ r) .* ∂u∂θ

  return (∂u∂x, ∂u∂y)

end

"""
    ∂x(u, D)

x-derivative

See `grad(u, D)`.
"""
function ∂x(u, D)
    return grad(u, D)[1]
end

"""
    ∂y(u, D)

y-derivative

See `grad(u, D)`.
"""
function ∂y(u, D)
    return grad(u, D)[2]
end

"""
    div(u, D)

Divergence operator

# Arguments
- `u` : tuple `(ux, uy)` of x- and y- components of a vector field on the disk
- `D` : discretization of the disk

# Returns
- divergence of vector field on the disk

Warning: Ill-conditioned, use with caution.
"""
function div(u::Tuple, D)
    return ∂x(u[1], D) .+ ∂y(u[2], D)
end

"""
    lap(u, D)

Laplacian operator

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- Laplacian of function on the disk

Warning: Ill-conditioned, use with caution.
"""
function lap(u, D)
  return div(grad(u, D), D)
end

"""
    laplace3d_single_layer(u, D)

Single layer of 3D Laplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk  
"""
function laplace3d_single_layer(u, D)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  # Even expansion of u * w
  ûw = psh(u .* D.w, D, parity=:even)

  # Compute weighted coefficients
  f̂ = D.Ŝ .* ûw

  # Evaluate on grid
  return ipsh(f̂, D, parity=:even)

end

function 𝒮(u, D)
  return laplace3d_single_layer(u, D)
end

"""
    laplace3d_single_layer_inverse(f, D)

Inverse of 𝒮

# Arguments
- `f` : single layer potential on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk  
"""
function laplace3d_single_layer_inverse(f, D)

  if isa(f, Number)
    f = fill(f, size(D.ζ))
  end

  # Even expansion of f
  f̂ = psh(f, D, parity=:even)

  # Compute weighted coefficients
  ûw = f̂ ./ D.Ŝ

  # Evaluate on grid
  return ipsh(ûw, D, parity=:even) ./ D.w

end

function 𝒮⁻¹(f, D)
  return laplace3d_single_layer_inverse(f, D)
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
function laplace3d_hypersingular(u, D)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  # Odd expansion of u
  û = psh(u, D, parity=:odd)

  # Compute weighted coefficients
  f̂w = D.N̂ .* û

  # Evaluate on grid
  return ipsh(f̂w, D, parity=:odd) ./ D.w

end

function 𝒩(u, D)
  return laplace3d_hypersingular(u, D)
end

"""
    laplace3d_hypersingular_inverse(f, D)

Inverse of 𝒩

# Arguments
- `f` : hypersingular operator on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk
"""
function laplace3d_hypersingular_inverse(f, D)

  if isa(f, Number)
    f = fill(f, size(D.ζ))
  end

  # Weighted odd expansion of f 
  f̂w = psh(f .* D.w, D, parity=:odd)

  # Compute coefficients
  û = f̂w ./ D.N̂

  # Evaluate on grid
  return ipsh(û, D, parity=:odd)

end

function 𝒩⁻¹(f, D)
  return laplace3d_hypersingular_inverse(f, D)
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
function laplace2d_volume(u, D)

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

function 𝒱(u, D)
  return laplace2d_volume(u, D)
end

"""
    bilaplace2d_volume(u, D; κ=0)

Volume potential of 2D Bilaplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- volume potential evaluated on the disk
"""
function bilaplace2d_volume(u, D; κ=0)

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

function ℬ(u, D; κ=0)
  return bilaplace2d_volume(u, D; κ=κ)
end

"""
    bilaplace3d_single_layer(u, D)

    Single layer of 3D Bilaplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk
"""
function bilaplace3d_single_layer(u, D)

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

function 𝒯(u, D)
  return bilaplace3d_single_layer(u, D)
end

"""
    δ𝒮(u, m, D)
    
Shape derivative of 𝒮

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δ𝒮(u, m, D)

  shp = size(u)

	ζ = D.ζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒮(arg, D)
	
	for _ = 0 : m
		val .+= -𝒮(arg, D) .* fac
		fac .*= ζ
		arg ./= ζ
	end

  return reshape(val, shp)

end

"""
    δ𝒩(u, m, D)

Shape derivative of 𝒩

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δ𝒩(u, m, D)

	ζ = D.ζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒩(arg, D)
	
	for _ = 0 : m
		val .+= -3𝒩(arg, D) .* fac
		fac .*= ζ
		arg ./= ζ
	end

	return val

end

"""
    δ𝒱(u, m, D)

Shape derivative of 𝒱

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
""" 
function δ𝒱(u, m, D)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ, dζ = D.ζ, D.dζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒱(arg, D)

	for _ = 0 : m
		val += (1 / 2π) * fac .* sum(arg .* dζ)
		fac .*= ζ
		arg ./= ζ
	end

  val = reshape(val, shp)

	return val

end

"""
    δℬ(u, m, D)

Shape derivative of ℬ

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δℬ(u, m, D)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ = D.ζ
  dζ = D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * ℬ(arg, D)

	for _ = 0 : m
		val += 2ℬ(arg, D) .* fac
		val += (1 / 8π) * fac .* sum(abs2.(ζ .- transpose(ζ)) .* transpose(arg .* dζ), dims=2)
		fac .*= ζ
		arg ./= ζ
	end

  return reshape(val, shp)

end
