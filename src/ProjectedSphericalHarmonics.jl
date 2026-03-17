
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

include("EllipticSolverCoefficientOperator/CoefficientSpaceOperators.jl")
export Inverse_laplacian_coef_m, Inverse_laplacian_coef_m_sparse, Inverse_laplacian, DirichletTraceInverseLaplacian_m, DirichletTraceColumnLaplacian_m, ModifiedPoissonSystemMatrix, NeumannTraceΔ⁻¹_sparse
export GenerateSparseMatrix_InverseLaplacian, GenerateSparseMatrix_r_dot_∇Δ⁻¹, GenerateSparseMatrix_ζ∂ζΔ⁻¹
export ∂ζΔ⁻¹_m_sparse, ∂ζ̄Δ⁻¹_m_sparse
export ζ∂ζΔ⁻¹_m_sparse, ζ̄∂ζ̄Δ⁻¹_m_sparse
export ∂ζΔ⁻¹, ∂ζ̄Δ⁻¹, r_dot_∇Δ⁻¹
export ∂ζ∂ζΔ⁻¹_m_sparse, ∂ζ∂ζ̄Δ⁻¹_m_sparse, ∂ζ̄∂ζ̄Δ⁻¹_m_sparse
export Inverse_laplacian_coef_m_sparse!, ∂ζΔ⁻¹_m_sparse!, ∂ζ̄Δ⁻¹_m_sparse!, ζ∂ζΔ⁻¹_m_sparse!, ζ̄∂ζ̄Δ⁻¹_m_sparse!
export ∂ζ∂ζΔ⁻¹_m_sparse!, ∂ζ∂ζ̄Δ⁻¹_m_sparse!, ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!, r_dot_∇Δ⁻¹!
export size_current_m, ∂ζ_indexing_sparse, ∂ζ̄_indexing_sparse, ∂ζ∂ζ_indexing_sparse, ∂ζ̄∂ζ̄_indexing_sparse
export DirichletTrace


include("AbstractDiskFunction.jl")
export AbstractDiskFunction

include("TriangularCoeffArray.jl")
export TriangularCoeffArray, column, ncolumns, NodalToTriangularArray, TriangularArrayToPSH

include("HarmonicSolver/CoefficientSpaceHarmonic.jl")
export HarmonicFunction, SolveHarmonicFunction_coefficient, EvaluateHarmonicFunction
export ∂ζ_HarmonicFunction!, ∂ζ̄_HarmonicFunction!

include("DiskFunction.jl")
export DiskFunction, DiskFunction!, add!, sub!, evaluate

end
