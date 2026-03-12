
module ProjectedSphericalHarmonics

include("grids.jl")

# Eigenfunctions
include("eigenfunctions.jl")
export ylm, ∂ylm∂ζ, Nlm, λlm

# Initializer
include("initializer.jl")
export disk

# Transforms
include("transforms.jl")
export psh, ipsh

# Integral and differential operators
include("operators.jl")
export 𝒮, 𝒩, 𝒱, ℬ, 𝒯, 𝒮⁻¹, 𝒩⁻¹, δ𝒮, δ𝒩, δ𝒱, δℬ
export ∂n, ∂θ, ∂ζ, ∂ζ̄, ∂x, ∂y, grad, div, lap
export trace

include("solvers.jl")
export Δ⁻¹, solve

include("CoefficientSpaceOperators.jl")
export Inverse_laplacian_coef_m, Inverse_laplacian_coef_m_sparse, Inverse_laplacian, DirichletTraceInverseLaplacian_m, DirichletTraceColumnLaplacian_m, ModifiedPoissonSystemMatrix
export GenerateSparseMatrix_InverseLaplacian
export ∂ζΔ⁻¹_m_sparse, ∂ζ̄Δ⁻¹_m_sparse

end