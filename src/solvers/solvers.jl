
using IterativeSolvers, LinearAlgebra, LinearOperators

"""
    Δ⁻¹(f, g, D)

Inverse Laplacian on the disk with Dirichlet boundary conditions.

# Arguments
- `f` : right-hand side function on the disk
- `g` : boundary data on the unit circle
- `D` : discretization of the disk

# Returns
- Solution `u` to the Poisson equation Δu = f with u|∂D = g
"""
function Δ⁻¹(f, g, D::Disk)

  # Check if scalar input
  if isa(f, Number)
    f = fill(f, size(D.ζ))
  end
  if isa(g, Number)
    g = fill(g, size(D.θ))
  end

  # Compute the particular solution
  uₚ = 𝒮𝒩⁻¹(f, D)

  # Compute the boundary value correction
  ûₕ = fft(g - trace(uₚ, D))
  ûₕ = D.r.^abs.(D.Mspan) .* ûₕ
  uₕ = ifft(ûₕ, 2)

  # Return
  return uₕ + uₚ

end

"""
    gmres_wrapper(L!, f)

Solve the linear system L σ = f using GMRES.

# Arguments
- `L!` : function that applies the linear operator L
- `f` : right-hand side vector

# Returns
- Solution vector σ
"""
function gmres(L!, f)

  # Store original shape
  shp = size(f)

  # Vectorize right-hand side
  f = vec(f)

  # Size 
  N = length(f)

  # Initial guess
  σ = zeros(eltype(f), N)

  # Solve using GMRES
  reltol = 1e-6
  op = LinearOperator(eltype(f), N, N, false, false, L!)
  σ, history = gmres!(σ, op, f; log=true, reltol=reltol)

  # Compute residual norm
  f̃ = similar(f)
  L!(f̃, σ)
  err = norm(f̃ - f) / norm(f)

  # Display convergence information
  if history.isconverged
      println("GMRES converged in $(history.iters) iterations (residual norm: $err).")
  else
      println("GMRES did not converge in $(history.iters) iterations (residual norm: $err).")
  end

  # Reshape and return
  return reshape(σ, shp)

end