
"""
    trace(u, Ω)

Evaluate function on boundary of doman

# Arguments
- `u` : array of function values
- `Ω` : domain discretization

# Returns
- function evaluated on the boundary of the domain
"""
trace(u::AbstractArray, Ω) = ipsh(psh(u, Ω), Ω, [1.0], parity=:even)
trace(u::NTuple{N}, Ω) where N = map(x -> trace(x, Ω), u)

"""
    integral(u, Ω)

Integrate function on the domain

# Arguments
- `u` : array of function values
- `Ω` : domain discretization

# Returns
- integral of u over Ω
"""
integral(u, Ω::Disk)   = sum(u .* Ω.dz)
integral(u, Ω::Domain) = sum(u .* Ω.dz)

"""
    apply(M, û, Ω; parity=:even)

Apply operator M to function û on the domain

# Arguments
- `M` : linear operator
- `û` : array of spectral coefficients
- `Ω` : domain discretization
- `parity` : parity of the function (default: even)

# Returns
- array of spectral coefficients resulting from applying M to û
"""
function apply(M, û::AbstractArray, Ω; parity=:even)
  idx = findall(getproperty(Ω, parity))
  f̂ = zeros(ComplexF64, size(Ω.z))
  f̂[idx] = M * û[idx]
  return f̂
end

function apply(M, f̂::Tuple, Ω; parity=:even)
  return (apply(M[1,1], f̂[1], Ω; parity=parity) .+ apply(M[1,2], f̂[2], Ω; parity=parity),
          apply(M[2,1], f̂[1], Ω; parity=parity) .+ apply(M[2,2], f̂[2], Ω; parity=parity))
end


"""
    solve(M, f̂, Ω; parity=:even)

Solve M * û = f̂ for û with given parity

# Arguments
- `M` : linear operator
- `f̂` : array of spectral coefficients
- `Ω` : domain discretization
- `parity` : parity of the function (default: even)

# Returns
- array of spectral coefficients of û
"""
function solve(M, f̂::AbstractArray, Ω; parity=:even)
  idx = findall(getproperty(Ω, parity))
  û = zeros(ComplexF64, size(Ω.z))
  û[idx]  = M \ f̂[idx]
  return û
end

function solve(M, f̂::Tuple, Ω; parity=:even)
  idx = findall(getproperty(Ω, parity))
  Ne = length(idx)
  M = [M[1,1] M[1,2]; M[2,1] M[2,2]]
  v  = M \ [f̂[1][idx]; f̂[2][idx]]
  û1 = zeros(ComplexF64, size(Ω.z)); û1[idx] .= v[1:Ne]
  û2 = zeros(ComplexF64, size(Ω.z)); û2[idx] .= v[Ne+1:end]
  return û1, û2
end

"""
    Ŵ(û, Ω)

Multiply by weight in the spectral domain

# Arguments
- `û` : array of spectral coefficients
- `Ω` : domain discretization

# Returns
- array of spectral coefficients resulting from applying W to û in the spectral domain
"""
function Ŵ(û, Ω)
  return reshape(Ω.W * vec(û), size(û))
end

"""
    Ŵ⁻¹(f̂, Ω)

Multiply by inverse weight in the spectral domain

# Arguments
- `f̂` : array of spectral coefficients
- `Ω` : domain discretization

# Returns
- array of spectral coefficients resulting from applying W⁻¹ to f̂ in the spectral domain
"""
function Ŵ⁻¹(f̂, Ω)
  shp = size(f̂)
  f̂ = vec(f̂)
  û = zeros(ComplexF64, length(f̂))
  û[Ω.Wqr.jb] = Ω.Wqr.W \ f̂[Ω.Wqr.ib]
  return reshape(û, shp)
end