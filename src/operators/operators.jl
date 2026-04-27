
"""
    trace(u, D)

Evaluate function on boundary of disk

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- function evaluated on the boundary of the disk
"""
trace(u::AbstractArray, D) = ipsh(psh(u, D), D, [1.0], parity=:even)
trace(u::NTuple{N}, D) where N = map(x -> trace(x, D), u)

"""
    integral(u, D)

Integrate function on the disk

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns- integral of the function on the disk
"""
integral(u, D) = sum(u .* D.dζ)

"""
    apply(M, û, D; parity=:even)

Apply operator M to function û on the disk

# Arguments
- `M` : operator to apply
- `û` : function on the disk in the spectral domain
- `D` : discretization of the disk
- `parity` : parity of the function (default: even)

# Returns
- function resulting from applying M to û
"""
function apply(M, û::AbstractArray, D; parity=:even)
  idx = findall(getproperty(D, parity))
  f̂ = zeros(ComplexF64, size(D.ζ))
  f̂[idx] = M * û[idx]
  return f̂
end

function apply(M, f̂::Tuple, D; parity=:even)
  return (apply(M[1,1], f̂[1], D; parity=parity) .+ apply(M[1,2], f̂[2], D; parity=parity),
          apply(M[2,1], f̂[1], D; parity=parity) .+ apply(M[2,2], f̂[2], D; parity=parity))
end


"""
    solve(M, f̂, D; parity=:even)

Solve the equation M * û = f̂ for û on the disk

# Arguments
- `M` : operator to solve
- `f̂` : function on the disk in the spectral domain
- `D` : discretization of the disk
- `parity` : parity of the function (default: even)

# Returns
- function û resulting from solving M * û = f̂
"""
function solve(M, f̂::AbstractArray, D; parity=:even)
  idx = findall(getproperty(D, parity))
  û = zeros(ComplexF64, size(D.ζ))
  û[idx]  = M \ f̂[idx]
  return û
end

function solve(M, f̂::Tuple, D; parity=:even)
  idx = findall(getproperty(D, parity))
  Ne = length(idx)
  M = [M[1,1] M[1,2]; M[2,1] M[2,2]]
  v  = M \ [f̂[1][idx]; f̂[2][idx]]
  û1 = zeros(ComplexF64, size(D.ζ)); û1[idx] .= v[1:Ne]
  û2 = zeros(ComplexF64, size(D.ζ)); û2[idx] .= v[Ne+1:end]
  return û1, û2
end

"""
    Ŵ(û, D)

Multiply by weight in the spectral domain

# Arguments
- `û` : function on the disk in the spectral domain
- `D` : discretization of the disk

# Returns
- function resulting from applying W to û in the spectral domain
"""
function Ŵ(û, D)
  return reshape(D.W * vec(û), size(û))
end

"""
    Ŵ⁻¹(f̂, D)

Multiply by inverse weight in the spectral domain

# Arguments
- `f̂` : function on the disk in the spectral domain
- `D` : discretization of the disk

# Returns
- function resulting from applying W⁻¹ to f̂ in the spectral domain
"""
function Ŵ⁻¹(f̂, D)
  shp = size(f̂)
  f̂ = vec(f̂)
  û = zeros(ComplexF64, length(f̂))
  û[D.Wqr.jb] = D.Wqr.W \ f̂[D.Wqr.ib]
  return reshape(û, shp)
end