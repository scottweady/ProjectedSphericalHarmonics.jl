
using FFTW

"""
    psh(u, D; parity=:even)

PSH transform of function `u` on disk `D`.

# Arguments
- `u` : function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- PSH coefficients
"""
function psh(u, D; parity=:even)

  # Basis functions of given parity
  Yr = getfield(D.Yr, parity)

  # Compute transform
  u = reshape(u, D.shp)
  û = fft(u, 2)
  û = Yr' * (û .* D.dw)

  # Get terms of given parity
  û = û[getfield(D.az, parity)]

  return û
  
end

"""
    ipsh(û, D; parity=:even)

Inverse PSH transform

# Arguments
- `û` : PSH coefficients of function on the disk
- `D` : discretization of the disk
- `parity` : either `:even` or `:odd` expansion

# Returns
- grid values
"""
function ipsh(û, D; parity=:even)

  # Basis functions of given parity
  Yr = getfield(D.Yr, parity)

  # Compute inverse transform
  û = (Yr .* transpose(û)) * getfield(D.az, parity)
  û = reshape(û, D.shp)
  u = ifft(û, 2) * D.shp[2]

  return vec(u)

end