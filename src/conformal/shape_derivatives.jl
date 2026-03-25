"""
    δ𝒮(u, m, D)
    
Shape derivative of 𝒮

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δ𝒮(u, m, D::Disk)

  shp = size(u)

	ζ = D.ζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒮(arg, D)
	
	for _ = 0 : m
		val .+= -𝒮(arg, D) .* fac
		fac .*= ζ
		arg ./= ζ
	end

  return reshape(val, shp)

end

"""
    δ𝒩(u, m, D)

Shape derivative of 𝒩

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δ𝒩(u, m, D::Disk)

	ζ = D.ζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒩(arg, D)
	
	for _ = 0 : m
		val .+= -3𝒩(arg, D) .* fac
		fac .*= ζ
		arg ./= ζ
	end

	return val

end

"""
    δ𝒱(u, m, D)

Shape derivative of 𝒱

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
""" 
function δ𝒱(u, m, D::Disk)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ, dζ = D.ζ, D.dζ

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * 𝒱(arg, D)

	for _ = 0 : m
		val += (1 / 2π) * fac .* sum(arg .* dζ)
		fac .*= ζ
		arg ./= ζ
	end

  val = reshape(val, shp)

	return val

end

"""
    δℬ(u, m, D)

Shape derivative of ℬ

# Arguments
- `u` : density function on the disk
- `m` : azimuthal mode number
- `D` : discretization of the disk

# Returns
- shape derivative of potential evaluated on the disk
"""
function δℬ(u, m, D::Disk)

  if isa(u, Number)
    u = fill(u, size(D.ζ))
  end

  shp = size(u)

	ζ = D.ζ
  dζ = D.dζ
  u, ζ, dζ = vec(u), vec(ζ), vec(dζ)

	fac = ζ.^0
	arg = ζ.^m .* u
	val = 2(m + 1) * ℬ(arg, D)

	for _ = 0 : m
		val += 2ℬ(arg, D) .* fac
		val += (1 / 8π) * fac .* sum(abs2.(ζ .- transpose(ζ)) .* transpose(arg .* dζ), dims=2)
		fac .*= ζ
		arg ./= ζ
	end

  return reshape(val, shp)

end
