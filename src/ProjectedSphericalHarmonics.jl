
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

include("EllipticOperators/CoefficientSpaceOperators.jl")
export Inverse_laplacian_coef_m, Inverse_laplacian
export traceĜ, traceĜ_column, helmholtz_matrix, neumann_traceĜ
export inverse_laplacian_matrix_sparse, r_dot_∇Δ⁻¹_matrix_sparse, ζ∂ζΔ⁻¹_matrix_sparse
export ∂ζΔ⁻¹, ∂ζ̄Δ⁻¹, r_dot_∇Δ⁻¹
export Ĝᵐ, ∂Ĝᵐ∂ζ, ∂Ĝᵐ∂ζ̄, ζ_∂Ĝᵐ∂ζ, ζ̄_∂Ĝᵐ∂ζ̄
export ∂²Ĝᵐ∂ζ², ∂²Ĝᵐ∂ζ∂ζ̄, ∂²Ĝᵐ∂ζ̄², r_∂Ĝᵐ∂r
export Ĝᵐ!, ∂Ĝᵐ∂ζ!, ∂Ĝᵐ∂ζ̄!, ζ_∂Ĝᵐ∂ζ!, ζ̄_∂Ĝᵐ∂ζ̄!
export ∂²Ĝᵐ∂ζ²!, ∂²Ĝᵐ∂ζ∂ζ̄!, ∂²Ĝᵐ∂ζ̄²!, r_∂Ĝᵐ∂r!
export size_current_m, ∂ζ_indexing_sparse, ∂ζ̄_indexing_sparse, ∂ζ∂ζ_indexing_sparse, ∂ζ̄∂ζ̄_indexing_sparse
export Ĝ, ∂Ĝ∂ζ, ∂Ĝ∂ζ̄, ζ_∂Ĝ∂ζ, ζ̄_∂Ĝ∂ζ̄
export ∂²Ĝ∂ζ², ∂²Ĝ∂ζ∂ζ̄, ∂²Ĝ∂ζ̄², r_∂Ĝ∂r
export Ĝ!, ∂Ĝ∂ζ!, ∂Ĝ∂ζ̄!, ζ_∂Ĝ∂ζ!, ζ̄_∂Ĝ∂ζ̄!
export ∂²Ĝ∂ζ²!, ∂²Ĝ∂ζ∂ζ̄!, ∂²Ĝ∂ζ̄²!, r_∂Ĝ∂r!


include("DiskFunction/AbstractDiskFunction.jl")
export AbstractDiskFunction

include("TriangularCoeffArray/TriangularCoeffArray.jl")
export TriangularCoeffArray, mode_coefficients, ncolumns, NodalToTriangularArray, TriangularArrayToPSH
export parity, ordering, lmax, circshift_fft_to_natural, circshift_natural_to_fft

include("EllipticOperators/FhatOperatorsTriangularArrays/apply_triangular.jl")

include("TransformsForTriangles/transforms.jl")
export psh!, ipsh!, psh_triangular

include("HarmonicOperators/harmonic_operators.jl")
export harmonic_coeff_from_dirichlet!, harmonic_coeff_from_dirichlet
export dirichlet_trace_harmonic!, dirichlet_trace_harmonic
export neumann_trace_harmonic!, neumann_trace_harmonic
export ∂ζ_harmonic!, ∂ζ_harmonic
export ∂ζ̄_harmonic!, ∂ζ̄_harmonic
export ζ_∂ζ_harmonic!, ζ_∂ζ_harmonic
export ζ̄_∂ζ̄_harmonic!, ζ̄_∂ζ̄_harmonic

include("DiskFunction/HarmonicFunctions.jl")
export HarmonicFunction, SolveHarmonicFunction_coefficient, EvaluateHarmonicFunction
export ∂ζ_HarmonicFunction!, ∂ζ̄_HarmonicFunction!

include("DiskFunction/DiskFunction.jl")
export DiskFunction, DiskFunction!, add!, sub!, evaluate

include("EllipticSolver/EllipticSolution.jl")
export EllipticSolution

end
