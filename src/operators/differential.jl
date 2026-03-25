
using FFTW
import Base: div

"""
    ∂ζ(u, D)

Complex differentiation

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- complex derivative of function on the disk

Warning: Ill-conditioned, use with caution.
"""
function ∂ζ(u, D::Disk; tol=1e-15)
  û = psh(u, D, parity=:even)
  û[abs.(û) .< tol] .= 0
  ∂ûw∂ζ = circshift(D.∂ζ̂ .* û, (0, -1))
  ∂ûw∂ζ[:, D.Mₘ + 1] .= 0.0 #zero out m = M mode
  ∂û∂ζ = Ŵ⁻¹(∂ûw∂ζ, D)
  return ipsh(∂û∂ζ, D, parity=:even)
end

"""
    ∂ζ̄(u, D)

Complex conjugate differentiation

See `∂ζ(u, D)`.
"""
function ∂ζ̄(u, D::Disk; tol=1e-15)
  ū = conj.(u)
  ∂ū∂ζ = ∂ζ(ū, D, tol=tol)
  return conj.(∂ū∂ζ)
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
function ∂n(u, D::Disk)

  # Even expansion of u
  û = psh(u, D; parity=:even)

  # Apply normal derivative operator in coefficient space
  ∂û∂n = sum(D.even .* (D.∂n̂ .* û), dims=1)

  # Inverse transform to physical space
  ∂u∂n = ifft(∂û∂n) * length(∂û∂n)

  return ∂u∂n

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
function grad(u, D::Disk)
  ∂u∂ζ, ∂u∂ζ̄ = ∂ζ(u, D), ∂ζ̄(u, D)
  ∂u∂x =   real.(∂u∂ζ + ∂u∂ζ̄)
  ∂u∂y =  -imag.(∂u∂ζ - ∂u∂ζ̄)
  return (ComplexF64.(∂u∂x), ComplexF64.(∂u∂y))
end

"""
    ∂x(u, D)

x-derivative

See `grad(u, D)`.
"""
function ∂x(u, D::Disk)
    return grad(u, D)[1]
end

"""
    ∂y(u, D)

y-derivative

See `grad(u, D)`.
"""
function ∂y(u, D::Disk)
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
function div(u::Tuple, D::Disk)
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
function lap(u, D::Disk)
  return 4 * ∂ζ̄(∂ζ(u, D), D)
end