using Test

@testset "ProjectedSphericalHarmonics.jl" begin

    @testset "EllipticOperators" begin
        include("EllipticOperators/per_frequency_operators.jl")
        include("EllipticOperators/full_fhat_operators.jl")
        include("EllipticOperators/full_fhat_operators_triangular.jl")
        include("EllipticOperators/boundary_operators.jl")
    end

    @testset "Functions" begin
        include("Functions/disk_function_test.jl")
        include("Functions/derivatives_test.jl")
        include("Functions/harmonic_function_test.jl")
    end

    @testset "TriangularArrays" begin
        include("TriangularArrays/runtests.jl")
    end

end
