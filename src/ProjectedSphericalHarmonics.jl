
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
export Inverse_laplacian_coef_m, Inverse_laplacian
export traceĜ, traceĜ_column, helmholtz_matrix, neumann_traceĜ
export inverse_laplacian_matrix_sparse, r_dot_∇Δ⁻¹_matrix_sparse, ζ∂ζΔ⁻¹_matrix_sparse
export ∂ζΔ⁻¹, ∂ζ̄Δ⁻¹, r_dot_∇Δ⁻¹
export Ĝᵐ, ∂Ĝᵐ∂ζ, ∂Ĝᵐ∂ζ̄, ζ_∂Ĝᵐ∂ζ, ζ̄_∂Ĝᵐ∂ζ̄
export ∂²Ĝᵐ∂ζ², ∂²Ĝᵐ∂ζ∂ζ̄, ∂²Ĝᵐ∂ζ̄², r_∂Ĝᵐ∂r
export Ĝᵐ!, ∂Ĝᵐ∂ζ!, ∂Ĝᵐ∂ζ̄!, ζ_∂Ĝᵐ∂ζ!, ζ̄_∂Ĝᵐ∂ζ̄!
export ∂²Ĝᵐ∂ζ²!, ∂²Ĝᵐ∂ζ∂ζ̄!, ∂²Ĝᵐ∂ζ̄²!, r_∂Ĝᵐ∂r!
export size_current_m, ∂ζ_indexing_sparse, ∂ζ̄_indexing_sparse, ∂ζ∂ζ_indexing_sparse, ∂ζ̄∂ζ̄_indexing_sparse


include("AbstractDiskFunction.jl")
export AbstractDiskFunction

include("TriangularCoeffArray/TriangularCoeffArray.jl")
export TriangularCoeffArray, mode_coefficients, ncolumns, NodalToTriangularArray, TriangularArrayToPSH
export parity, ordering, circshift_fft_to_natural, circshift_natural_to_fft

include("TransformsForTriangles/transforms.jl")
export psh!, ipsh!, psh_triangular

include("HarmonicSolver/CoefficientSpaceHarmonic.jl")
export HarmonicFunction, SolveHarmonicFunction_coefficient, EvaluateHarmonicFunction
export ∂ζ_HarmonicFunction!, ∂ζ̄_HarmonicFunction!

include("DiskFunction.jl")
export DiskFunction, DiskFunction!, add!, sub!, evaluate

end
