
module ProjectedSphericalHarmonics

"""
Domain data types and initializers
"""

# Disk
include("domains/disk.jl")
export disk

# Domain
include("domains/domain.jl")
export domain

"""
Spectral discretization
"""

# Grids
include("spectral/grids.jl")

# Eigenfunctions
include("spectral/eigenfunctions.jl")
export ylm, ∂ylm∂ζ, Nlm, λlm

# Transforms
include("spectral/transforms.jl")
export psh, ipsh, psh!, ipsh!

# Algorithms
include("algorithms/aca.jl")

"""
Integral and differential operators
"""

# Integral operators
include("operators/integral.jl")
export 𝒮, 𝒩, 𝒱, ℬ, 𝒯, 𝒮⁻¹, 𝒩⁻¹
export 𝒮!, 𝒩!, 𝒮⁻¹!, 𝒩⁻¹!

# Differential operators
include("operators/differential.jl")
export ∂n, ∂ζ, ∂ζ̄, ∂x, ∂y, grad, div, lap

# Other operators
include("operators/operators.jl")
export trace, integral

# Extension to non-circular domains
include("domains/operators.jl")
export δ𝒮, δ𝒩, δ𝒱, δℬ

"""
Solvers
"""

include("solvers/solvers.jl")
export Δ⁻¹, solve

include("solvers/constructor.jl")

end