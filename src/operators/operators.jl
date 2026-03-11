
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