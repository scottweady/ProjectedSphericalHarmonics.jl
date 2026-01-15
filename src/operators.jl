
using FFTW

export ∂n, ∂r, ∂θ, ∂x, ∂y, grad, div, lap
export 𝒮, 𝒮⁻¹, 𝒩, 𝒩⁻¹, 𝒱, ℬ, δ𝒮, δ𝒩, δ𝒱, δℬ
export psh_transform

"""
    psh_transform(u, D; kind=:even)

PSH transform of function `u` on disk `D`.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk
- `kind` : either `:even` or `:odd` expansion

# Returns
- PSH coefficients of `u`
"""
function psh_transform(u, D; kind=:even)

  # Degree of radial expansion
  M = D.M

  # Quadrature weights
  ζ = @view D.ζ[1 : (M + 1)]
  dζ = @view D.dζ[1 : (M + 1)] 

  # Basis functions
  Y = @view getfield(D.Y, kind)[1 : (M + 1), :]

  # Compute transform
  u = reshape(u, M + 1, 2M + 1)
  uₖ = fft(u, 2)
  uₖ = Y' * (uₖ .* (dζ ./ sqrt.(1 .- abs2.(ζ))))

  # Get relevant coefficients
  modes = D.modes[getfield(D, kind)]
  azimuthal_modes = [mod(m, 2M + 1) + 1 for (_, m) in modes]
  uₖ = [uₖ[i, j] for (i, j) in enumerate(azimuthal_modes)]

  return uₖ
  
end

"""
    ∂n(u, D)

Normal derivative operator

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- normal derivative of function evaluated at the boundary

Warning: Ill-conditioned, use with caution.
"""
function ∂n(u, D; tol=1e-8)
  uₖ = psh_transform(u, D, kind=:even)
  uₖ[abs.(uₖ) .< tol] .= 0.0
  return D.∂Y∂n.even * uₖ
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
function ∂r(u, D)
    uₖ = psh_transform(u, D, kind=:even)
    return D.∂Y∂r.even * uₖ
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
    uₖ = psh_transform(u, D, kind=:even)
    return D.Y.even * (D.∂̂.θ * uₖ)
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
function grad(u, D)

    r, θ = abs.(D.ζ), angle.(D.ζ)
    ∂u∂r, ∂u∂θ = ∂r(u, D), ∂θ(u, D)

    ux = cos.(θ) .* ∂u∂r .- (sin.(θ) ./ r) .* ∂u∂θ
    uy = sin.(θ) .* ∂u∂r .+ (cos.(θ) ./ r) .* ∂u∂θ

    return (ux, uy)

end

"""
    ∂x(u, D)

x-derivative

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- x-derivative of function on the disk

Warning: Ill-conditioned, use with caution.
"""
function ∂x(u, D)
    return grad(u, D)[1]
end

"""
    ∂y(u, D)

y-derivative

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- y-derivative of function on the disk

Warning: Ill-conditioned, use with caution.
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
function div(u, D)
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
    𝒮(u, D)

Single layer of 3D Laplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk  
"""
function 𝒮(u, D)

  # Even expansion of u * w
  uwₖ = psh_transform(u .* D.w, D, kind=:even)

  # Compute weighted coefficients
  fₖ = D.Ŝ * uwₖ

  # Evaluate on grid
  return D.Y.even * fₖ

end

"""
    𝒮⁻¹(f, D)

Inverse of 𝒮

# Arguments
- `f` : single layer potential on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk  
"""
function 𝒮⁻¹(f, D)

  # Even expansion of f
  fₖ = psh_transform(f, D, kind=:even)

  # Compute weighted coefficients
  uwₖ = D.Ŝ \ fₖ

  # Evaluate on grid
  return (D.Y.even * uwₖ) ./ D.w

end

"""
    𝒩(u, D)

Hypersingular operator

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- hypersingular operator evaluated on the disk
"""
function 𝒩(u, D)

  # Odd expansion of u
  uₖ = psh_transform(u, D, kind=:odd)

  # Compute weighted coefficients
  fwₖ = D.N̂ * uₖ

  # Evaluate on grid
  return (D.Y.odd * fwₖ) ./ D.w

end


"""
    𝒩⁻¹(f, D)

Inverse of 𝒩

# Arguments
- `f` : hypersingular operator on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk
"""
function 𝒩⁻¹(f, D)

  # Weighted odd expansion of f 
  fwₖ = psh_transform(f .* D.w, D, kind=:odd)

  # Compute coefficients
  uₖ = D.N̂ \ fwₖ

  # Evaluate on grid
  return D.Y.odd * uₖ

end

"""
    𝒱(u, D)
    
Single layer of 2D Laplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk

"""
function 𝒱(u, D)

    ζ, dζ = D.ζ, D.dζ
    δζ = ζ .- transpose(ζ)
    V = (1 / 2π) * log.(abs.(δζ) .+ (δζ .== 0)) .* dζ';
    return V * u .+ ((abs2.(ζ) .- 1) / 4 .- sum(V, dims=2)) .* u

end

"""
    ℬ(u, D)

Single layer of 2D Bilaplacian

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk
"""
function ℬ(u, D; κ=0)

	ζ, dζ = D.ζ, D.dζ
    δζ = ζ .- transpose(ζ)
	B = (1 / 8π) * abs2.(δζ) .* (log.(abs.(δζ) .+ (δζ .== 0)) .- 1 .+ κ) .* dζ';
    return B * u

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

	ζ = D.ζ
	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒮(arg, D)
	
	for _ = 0 : m
		val .+= -𝒮(arg, D) .* fac
		fac .*= ζ
		arg ./= ζ
	end

	return val

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

	ζ = D.ζ
    dζ = D.dζ
	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒱(arg, D)

	for _ = 0 : m
		val += (1 / 2π) * fac .* sum(ones(size(ζ)) .* transpose(arg .* dζ), dims=2)
		fac .*= ζ
		arg ./= ζ
	end

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

	ζ = D.ζ
    dζ = D.dζ
	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * ℬ(arg, D)

	for _ = 0 : m
		val += 2ℬ(arg, D) .* fac
		val += (1 / 8π) * fac .* sum(abs2.(ζ .- transpose(ζ)) .* transpose(arg .* dζ), dims=2)
		fac .*= ζ
		arg ./= ζ
	end

	return val

end
