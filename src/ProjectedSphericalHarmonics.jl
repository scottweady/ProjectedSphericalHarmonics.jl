
module ProjectedSphericalHarmonics

"""
Useful operations
"""

Base.broadcasted(::typeof(+), u::Tuple, w::AbstractArray) = 
  map(x -> x .+ w, u)

Base.broadcasted(::typeof(-), u::Tuple, w::AbstractArray) = 
  map(x -> x .- w, u)

Base.:-(u::Tuple) = map(x -> -x, u)

Base.:+(u::Tuple, v::Tuple) = map(+, u, v)
Base.:-(u::Tuple, v::Tuple) = map(-, u, v)

Base.broadcasted(::typeof(*), u::Tuple, w::AbstractArray) = 
  map(x -> x .* w, u) 

Base.broadcasted(::typeof(/), u::Tuple, w::AbstractArray) = 
  map(x -> x ./ w, u) 



"""
Domain data types and initializers
"""

# Disk
include("domains/disk.jl")
export disk, Disk

# Non-circular domains
include("conformal/conformal.jl")
include("algorithms/aca.jl")
include("domains/domain.jl")
include("domains/ellipse.jl")

export domain, ellipse, Domain

"""
Spectral discretization
"""

# Grids
include("spectral/grids.jl")

# Eigenfunctions
include("spectral/eigenfunctions.jl")
export ylm, ∂ylm∂ζ, Nlm, λlm, Clmn

# Transforms
include("spectral/transforms.jl")
export psh, ipsh, psh!, ipsh!, upsample

"""
Integral and differential operators
"""

# Integral operators
include("operators/matrices.jl")
include("operators/integral.jl")
include("operators/stokes.jl")
export 𝒮, 𝒩, 𝒮⁻¹, 𝒩⁻¹, 𝒮𝒩⁻¹
export 𝒱, ℬ, 𝒯
export 𝒮_st, 𝒮_st⁻¹

# Differential operators
include("operators/differential.jl")
export ∂n, ∂ζ, ∂ζ̄, ∂x, ∂y, grad, div, lap

# Other operators
include("operators/operators.jl")
export trace, integral, apply, solve

# Extend operators to non-circular domains (overloads 𝒮, 𝒩, etc.)
include("conformal/shape_derivatives.jl")
export δ𝒮, δ𝒩, δ𝒱, δℬ

"""
Solvers
"""

include("solvers/solvers.jl")
export Δ⁻¹, gmres

include("solvers/stokes.jl")

end