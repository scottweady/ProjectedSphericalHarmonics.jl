using ProjectedSphericalHarmonics

function print_error(label, value, tol1=1e-6, tol2=1e-8)
  if value > tol1
      printstyled("$label$value\n"; color=:red)
  elseif value > tol2
      printstyled("$label$value\n"; color=:yellow)
  else
      printstyled("$label$value\n"; color=:green)
  end
end

include("test_integral.jl")
include("test_differential.jl")
include("test_domain.jl")
