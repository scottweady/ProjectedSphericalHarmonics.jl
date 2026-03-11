
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
    function $(esc(name!))(u::AbstractMatrix{ComplexF64}, D::Disk)
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
    stokes3d_single_layer(σ::Tuple{Tuple, Tuple}, D::Disk)

"""

# 3D Laplace single layer on non-circular domains
function stokeslet_decomposition(f, D::Disk)

  ζ = D.ζ

  # Conformal kernel
  f1 = vec(f[1])
  f1t = transpose(f1)
  f2 = vec(f[2])
  f2t = transpose(f2)
  ζ = vec(ζ)
  ζt = transpose(ζ)
  
  x, y = real.(ζ .- ζt), imag.(ζ .- ζt)
  r = abs.(ζ .- ζt)
  x̂ = x ./ r
  ŷ = y ./ r
  K11 = (1.0 .+ x̂ .* x̂) .* (f1 .- f1t)
  K12 = x̂ .* ŷ .* (f2 .- f2t)
  K21 = x̂ .* ŷ .* (f1 .- f1t)
  K22 = (1.0 .+ ŷ .* ŷ) .* (f2 .- f2t)

  K11[diagind(K11)] .= 0.0
  K12[diagind(K12)] .= 0.0
  K21[diagind(K21)] .= 0.0
  K22[diagind(K22)] .= 0.0

  # Low-rank approximation of kernel
  U11, V11 = aca(K11)
  U12, V12 = aca(K12)
  U22, V22 = aca(K22)

  U11, Σ11, V11 = svd(K11)
  tol = 1e-6
  id = Σ11/Σ11[1] .> tol
  U11, V11 = U11[:, id] * Diagonal(Σ11[id]), V11[:, id]

  return ( (U11, U12), (U12, U22) ), ( (V11, V12), (V12, V22) )

end

export stokeslet_decomposition

function stokes3d_single_layer(σ::Tuple{Tuple, Tuple}, D::Disk)

  x = real.(D.ζ), imag.(D.ζ)
  Z = zeros(ComplexF64, size(D.ζ))
  F = ((copy(Z), copy(Z)), (copy(Z), copy(Z)))
  
  for j = 1 : 2
    xⱼ = x[j]
    for k = 1 : 2
      σⱼₖ = σ[j][k]
      for i = 1 : 2
        xᵢ = x[i]
        δᵢⱼ = (i == j) ? 1.0 : 0.0
        F[i][k] .+= δᵢⱼ .* 𝒮(σⱼₖ, D) + xᵢ .* xⱼ .* 𝒩(σⱼₖ, D) - xᵢ .* 𝒩(xⱼ .* σⱼₖ, D) - xⱼ .* 𝒩(xᵢ .* σⱼₖ, D) + 𝒩(xᵢ .* xⱼ .* σⱼₖ, D)
      end
    end
  end

  Gσ = div(F[1], D), div(F[2], D)
  return Gσ

end
  
export stokes3d_single_layer

@wrap_operator 𝒮 laplace3d_single_layer!
@wrap_operator 𝒮⁻¹ laplace3d_single_layer_inverse!
@wrap_operator 𝒩 laplace3d_hypersingular!
@wrap_operator 𝒩⁻¹ laplace3d_hypersingular_inverse!

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

function 𝒱(u, D::Disk)
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

function ℬ(u, D::Disk; κ=0)
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

function 𝒯(u, D::Disk)
  return bilaplace3d_single_layer(u, D)
end