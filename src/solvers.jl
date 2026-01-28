
using IterativeSolvers, LinearAlgebra, LinearOperators

function Δ⁻¹(f, g, D)

  # Check if scalar input
  if isa(f, Number)
      f = fill(f, length(D.ζ))
  end
  if isa(g, Number)
      g = fill(g, length(D.θ))
  end

  shp = size(f)
  f, g = vec(f), vec(g)
  
  # Compute the particular solution
  uₚ = 𝒮(𝒩⁻¹(f, D), D)

  # Compute the boundary value correction
  ûₕ = fft(g - trace(uₚ, D))
  ûₕ = D.r.^abs.(D.Mspan) .* transpose(ûₕ)
  uₕ = vec(ifft(ûₕ, 2))

  # Return
  return reshape(uₕ + uₚ, shp)

end

function solve(L!, f)

  # Size 
  N = length(f)

  # Initial guess
  σ = zeros(eltype(f), N)

  # Solve using GMRES
  op = LinearOperator(eltype(f), N, N, false, false, L!)
  σ, history = gmres!(σ, op, f; log=true, reltol=1e-10)

  # Compute residual norm
  f̃ = similar(f)
  L!(f̃, σ)
  err = norm(f̃ - f) / norm(f)

  # Display convergence information
  println("GMRES converged in $(history.iters) iterations (residual norm: $err).")

  return σ

end