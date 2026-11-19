
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

  K(x) = (1/4π) * 1/|x|

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- single layer potential evaluated on the disk  
"""
function laplace3d_single_layer!(u::AbstractMatrix{ComplexF64}, D::Disk)

  u .*= D.w
  psh!(u, D, parity=:even)
  u .*= D.K̂_S
  ipsh!(u, D, parity=:even)
  return u

end

"""
    laplace3d_single_layer_inverse!(f, D)

Inverse of 𝒮

  K(x) = (1/4π) * 1/|x|

# Arguments
- `f` : single layer potential on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk  
"""
function laplace3d_single_layer_inverse!(f::AbstractMatrix{ComplexF64}, D::Disk)

  psh!(f, D, parity=:even)
  f ./= D.K̂_S
  ipsh!(f, D, parity=:even)
  f ./= D.w
  return f

end

"""
    laplace3d_hypersingular(u, D)

Hypersingular operator

  K(x) = (1/4π) * 1/|x|^3

# Arguments
- `u` : density function on the disk
- `D` : discretization of the disk

# Returns
- hypersingular operator evaluated on the disk
"""
function laplace3d_hypersingular!(u::AbstractMatrix{ComplexF64}, D::Disk)

  psh!(u, D, parity=:odd)
  u .*= D.K̂_N
  ipsh!(u, D, parity=:odd)
  u ./= D.w
  return u

end

"""
    laplace3d_hypersingular_inverse!(f, D::Disk)

Inverse of 𝒩

  K(x) = (1/4π) * 1/|x|^3

# Arguments
- `f` : hypersingular operator on the disk
- `D` : discretization of the disk

# Returns
- density function on the disk
"""
function laplace3d_hypersingular_inverse!(f::AbstractMatrix{ComplexF64}, D::Disk)

  f .*= D.w
  psh!(f, D, parity=:odd)
  f ./= D.K̂_N
  ipsh!(f, D, parity=:odd)
  return f
  
end

"""
    laplace2d_volume(u, D)
    
Volume potential of 2D Laplacian

  K(x) = (1/2π) * log|x|

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

"""
    𝒱(u, D::Disk)

Volume potential of the 2D Laplacian on the unit disk,

```math
\\mathcal{V}[u](x) = \\frac{1}{2\\pi} \\int_D \\log|x - y|\\, u(y)\\, dy,
```

so that ``Δ\\mathcal{V}[u] = u`` and `𝒱(1, D)` ``= (|x|^2 - 1)/4``. A dense direct sum over the grid
with the singularity subtracted (``O(N^2)`` in the number of grid points `N`); accurate to about
the quadrature error of a log singularity on the grid, not spectrally.
"""
𝒱(u, D::Disk) = laplace2d_volume(u, D)

"""
    bilaplace2d_volume(u, D; κ=0)

Volume potential of 2D Bilaplacian

  K(x) = (1/8π) * |x|^2 * (log|x| - 1 + κ)

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

"""
    ℬ(u, D::Disk; κ=0)

Volume potential of the 2D bilaplacian on the unit disk,

```math
\\mathcal{B}[u](x) = \\frac{1}{8\\pi} \\int_D |x - y|^2 \\left(\\log|x - y| - 1 + κ\\right) u(y)\\, dy,
```

so that ``Δ^2 \\mathcal{B}[u] = u``; `κ` adds a multiple of the biharmonic ``\\int |x - y|^2 u``. A
dense direct sum over the grid (``O(N^2)``), algebraically accurate (the kernel is continuous).
"""
ℬ(u, D::Disk; κ=0) = bilaplace2d_volume(u, D; κ=κ)

"""
  bilaplace3d_single_layer(u, D)

  Single layer of 3D Bilaplacian

    K(x) = (1/8π) * |x|

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
	ζ = D.ζ
  return 0.5 * (abs2.(ζ) .* 𝒮(u, D) - ζ .* 𝒮(conj.(ζ) .* u, D) - conj.(ζ) .* 𝒮(ζ .* u, D) + 𝒮(abs2.(ζ) .* u, D))
end

"""
    𝒯(u, D::Disk)

Single layer of the 3D bilaplacian on the unit disk,

```math
\\mathcal{T}[u](x) = \\frac{1}{8\\pi} \\int_D |x - y|\\, u(y)\\, dy,
```

computed spectrally from four applications of `𝒮` (``|x - y| = |x - y|^2/|x - y|``). Like `𝒮`, it
expects a density with the ``1/w`` edge behavior.
"""
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

@wrap_operator 𝒮 laplace3d_single_layer!
@wrap_operator 𝒮⁻¹ laplace3d_single_layer_inverse!
@wrap_operator 𝒩 laplace3d_hypersingular!
@wrap_operator 𝒩⁻¹ laplace3d_hypersingular_inverse!

"""
    𝒮(u, Ω)

Single layer potential of the 3D Laplacian on the disk or a `Domain` `Ω`, evaluated on `Ω`,

```math
\\mathcal{S}[u](x) = \\frac{1}{4\\pi} \\int_Ω \\frac{u(y)}{|x - y|}\\, dy.
```

`u` is a density of the form ``g/w`` with ``g`` smooth, the edge behavior of solutions of
first-kind equations on an open surface; `g = u .* w` is expanded in even PSH. On the unit disk
these are eigenfunctions, ``\\mathcal{S}[y_ℓ^m/w] = (λ_ℓ^m/4)\\, y_ℓ^m`` (`λlm`), so `𝒮` is diagonal
in coefficient space; on a `Domain` it is a sparse (`ellipse`) or dense (`domain`) matrix there.
`u` may be a number (a constant density).
"""
𝒮

"""
    𝒮⁻¹(f, Ω)

Inverse of `𝒮`: the density ``u`` with ``\\mathcal{S}[u] = f`` on `Ω`, for smooth `f`. The result
has the ``1/w`` edge singularity: on the unit disk `𝒮⁻¹(1, D)` is ``4/(π w)``. `f` may be a number.
"""
𝒮⁻¹

"""
    𝒩(u, Ω)

Hypersingular operator of the 3D Laplacian on the disk or a `Domain` `Ω` (the normal derivative
of the double layer potential, kernel ``1/(4π|x - y|^3)`` in the finite-part sense). `u` is a
density vanishing like ``w`` at the edge (``w`` times a smooth function), expanded in odd PSH;
on the unit disk these are eigenfunctions, ``\\mathcal{N}[y_ℓ^m] = -y_ℓ^m/(λ_ℓ^m w)`` for ``ℓ + m``
odd. Returns a smooth function. `u` may be a number.
"""
𝒩

"""
    𝒩⁻¹(f, Ω)

Inverse of `𝒩`: the density ``u`` (vanishing like ``w`` at the edge) with ``\\mathcal{N}[u] = f`` on
`Ω`, for smooth `f`. `f` may be a number.
"""
𝒩⁻¹

"""

Overload operators for Domain type using conformal mapping approach.

"""

function 𝒮(u, Ω::Domain)
  if isa(u, Number)
    u = fill(u, size(Ω.z))
  end
  û = psh(u .* Ω.w, Ω, parity=:even)
  Ŝu = apply(Ω.K̂_S, û, Ω, parity=:even)
  return ipsh(Ŝu, Ω, parity=:even)
end

function 𝒮⁻¹(f, Ω::Domain)
  if isa(f, Number)
    f = fill(f, size(Ω.z))
  end
  f̂ = psh(f, Ω, parity=:even)
  û = solve(Ω.K̂_S, f̂, Ω, parity=:even)
  return ipsh(û, Ω, parity=:even) ./ Ω.w
end

function 𝒩(u, Ω::Domain)
  if isa(u, Number)
    u = fill(u, size(Ω.z))
  end
  û = psh(u, Ω, parity=:odd)
  N̂u = apply(Ω.K̂_N, û, Ω, parity=:odd)
  return ipsh(N̂u, Ω, parity=:odd) ./ Ω.w
end

function 𝒩⁻¹(f, Ω::Domain)
  if isa(f, Number)
    f = fill(f, size(Ω.z))
  end
  f̂ = psh(f .* Ω.w, Ω, parity=:odd)
  û = solve(Ω.K̂_N, f̂, Ω, parity=:odd)
  return ipsh(û, Ω, parity=:odd)
end
