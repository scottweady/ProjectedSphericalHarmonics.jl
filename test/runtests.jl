using ProjectedSphericalHarmonics
using Test

"Print `label` and `value`, and record a test that `value ≤ tol`."
function check(label, value, tol)
  pass = value ≤ tol
  printstyled("$label$value  (tol $tol)\n"; color = pass ? :green : :red)
  @test value ≤ tol
end

@testset "ProjectedSphericalHarmonics" begin
  @testset "Integral operators" include("test_integral.jl")
  @testset "Differential operators" include("test_differential.jl")
  @testset "Domains" include("test_domain.jl")
  @testset "Stokes mobility" include("test_stokes_mobility.jl")
  @testset "Close evaluation" include("test_close_eval.jl")
end
