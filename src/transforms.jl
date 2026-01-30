
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

  if u isa Number
    u = fill(u, size(D.ζ))
  end

  u = ComplexF64.(u)
  û = similar(u)
  uₘ = fft(u, 2)

  for (nm, m) in enumerate(D.Mspan)

    y₋₁ = ylm(abs(m), m, D.r)
    y = ylm(abs(m) + 1, m, D.r)
    y₊₁ = similar(y)

    w = sqrt.(1 .- D.r.^2)
    uw = uₘ[:, nm] .* D.dw
    û[abs(m)+1, nm] = dot(y₋₁, uw)

    if abs(m) == D.Mr
      continue
    end

    û[abs(m)+2, nm] = dot(y, uw)

    for nl = abs(m) + 2 : D.Mr
      y₊₁ = D.a[nl, nm] * w .* y .+ D.am1[nl, nm] * y₋₁
      û[nl + 1, nm] = dot(y₊₁, uw)
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end
  end

  û .*= getfield(D, parity)

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
function ipsh(û, D, r; parity=:even)

  û = ComplexF64.(û)
  û .*= getfield(D, parity)
  
  # Compute transform
  u = zeros(ComplexF64, (length(r), D.shp[2]))

  for (nm, m) in enumerate(D.Mspan)

    y₋₁ = ylm(abs(m), m, r)
    y = ylm(abs(m) + 1, m, r)
    y₊₁ = similar(y)

    w = sqrt.(1 .- r.^2)
    u[:, nm] .+= û[abs(m)+1, nm] * y₋₁

    if abs(m) == D.Mr
      continue
    end

    u[:, nm] .+= û[abs(m)+2, nm] * y

    for nl = abs(m) + 2 : D.Mr
      y₊₁ = D.a[nl, nm] * w .* y .+ D.am1[nl, nm] * y₋₁
      u[:, nm] .+= û[nl + 1, nm] * y₊₁
      y₋₁, y, y₊₁ = y, y₊₁, y₋₁
    end
  end

  u = ifft(u, 2) * D.shp[2]

  return u
  
end

function ipsh(û, D; parity=:even)
  return ipsh(û, D, D.r, parity=parity)
end