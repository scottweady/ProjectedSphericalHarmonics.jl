
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
include("conformal/harmonic.jl")
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

include("suspensions/stokes.jl")

# FINUFFT's threaded FFTs cannot run under FFTW.jl's Julia-threads callback (installed when
# Julia has more than one thread), so restore FFTW's own threading at load time
function __init__()
  if FFTW.fftw_provider == "fftw" && Threads.nthreads() > 1
    FFTW.dlopen(FFTW.libfftw3); FFTW.dlopen(FFTW.libfftw3f)
    ccall((:fftw_threads_set_callback, FFTW.libfftw3), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
    ccall((:fftwf_threads_set_callback, FFTW.libfftw3f), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), C_NULL, C_NULL)
  end
end

end