using ProjectedSphericalHarmonics
using Test



@testset "Length of stuff even" begin

    Mr = 10
    D = disk(Mr)
    Mspan = vec(D.Mspan)
    lmax = Mr
    f̂empty = TriangularCoeffArray{Float64}(lmax, Mspan)


    for m in Mspan

        @test length(mode_coefficients(f̂empty, m)) == size_current_m(lmax, m)

    end

end


@testset "Length of stuff even" begin

    Mr = 11
    D = disk(Mr)
    Mspan = vec(D.Mspan)
    lmax = Mr
    f̂empty = TriangularCoeffArray{Float64}(lmax, Mspan)


    for m in Mspan

        @test length(mode_coefficients(f̂empty, m)) == size_current_m(lmax, m)

    end

end