
"""
    trace(u, D)

Evaluate function on boundary of disk

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk

# Returns
- function evaluated on the boundary of the disk
"""
function trace(u, D)
    û = psh(u, D)
    return ipsh(û, D, [1.0], parity=:even)
end

function trace(u::Tuple, D)
    return (trace(u[1], D), trace(u[2], D))
end

function integral(u, D)
  return sum(u .* D.dζ)
end

function apply(M, û, D::Disk; parity=:even)
  idx = findall(getfield(D, parity))
  f̂ = zeros(ComplexF64, size(D.ζ))
  f̂[idx] = M * û[idx]
  return f̂
end

function solve(M, f̂, D::Disk; parity=:even)
  idx = findall(getfield(D, parity))
  û = zeros(ComplexF64, size(D.ζ))
  û[idx]  = M \ f̂[idx]
  return û
end

function apply(M, f̂::Tuple, D::Disk; parity=:even)
  return (apply(M[1,1], f̂[1], D; parity=parity) .+ apply(M[1,2], f̂[2], D; parity=parity),
          apply(M[2,1], f̂[1], D; parity=parity) .+ apply(M[2,2], f̂[2], D; parity=parity))
end

function solve(M, f̂::Tuple, D::Disk; parity=:even)
  idx = findall(getfield(D, parity))
  Ne = length(idx)
  M = [M[1,1] M[1,2]; M[2,1] M[2,2]]
  v  = M \ [f̂[1][idx]; f̂[2][idx]]
  û1 = zeros(ComplexF64, size(D.ζ)); û1[idx] .= v[1:Ne]
  û2 = zeros(ComplexF64, size(D.ζ)); û2[idx] .= v[Ne+1:end]
  return û1, û2
end

function Ŵ(û, D)
  return reshape(D.W * vec(û), size(û))
end

function Ŵ⁻¹(f̂, D)
  shp = size(f̂)
  f̂ = vec(f̂)
  û = zeros(ComplexF64, length(f̂))
  û[D.Wqr.jb] = D.Wqr.W \ f̂[D.Wqr.ib]
  return reshape(û, shp)
end