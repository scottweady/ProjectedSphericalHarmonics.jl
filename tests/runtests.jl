using Test

@testset "ProjectedSphericalHarmonics.jl" begin
    include("coefficient_space_operators_test.jl")
    include("coefficient_space_operators_inplace_test.jl")
    include("disk_function_test.jl")
end
