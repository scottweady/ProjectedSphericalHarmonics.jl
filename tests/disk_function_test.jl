using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

function _make_test_function(D)
    x = real.(D.ζ)
    y = imag.(D.ζ)
    return (1 .- x.^2 .- y.^2) .* cos.(π .* x) .* sin.(2π .* y)
end

@testset "DiskFunction — even Mr (Mr=32)" begin
    let D = disk(32), lmax = 32
        f  = _make_test_function(D)
        df = DiskFunction(f, D)

        # 1. Type and shape
        @test df._coeffs[1] isa TriangularCoeffArray
        @test df._coeffs[5] isa TriangularCoeffArray
        @test ncolumns(df._coeffs[1]) == length(D.Mspan)
        for (i, m) in enumerate(Array(D.Mspan))
            @test length(column(df._coeffs[1], m)) == size_current_m(lmax, m)
            @test length(column(df._coeffs[5], m)) == size_current_m(lmax, m)
        end

        # 2. û physically solves Δu = f
        u_comp  = real.(ipsh(TriangularArrayToPSH(df._coeffs[1], D), D))
        u_ref_1 = real.(Inverse_laplacian(f, D))
        u_ref_2 = 𝒮(𝒩⁻¹(f, D), D)
        @test norm(u_comp - u_ref_1) / norm(u_ref_1) < 1e-12
        @test norm(u_comp - u_ref_2) / norm(u_ref_2) < 1e-12

        # 3. ∂ζ∂ζ̄û = f̂ᵐ / 4
        f̂ = psh(f, D)
        for (i, m) in enumerate(Array(D.Mspan))
            f̂_sparse_m = Vector{ComplexF64}(f̂[D.even[:,i], i])
            @test norm(4 .* column(df._coeffs[5], m) .- f̂_sparse_m) < 1e-14
        end

        # 4. Per-frequency consistency with direct operator call
        for (i, m) in enumerate(Array(D.Mspan))
            f̂_sparse_m = Vector{ComplexF64}(f̂[D.even[:,i], i])
            expected    = Inverse_laplacian_coef_m_sparse(f̂_sparse_m, lmax, m; aliasing=false)
            @test norm(column(df._coeffs[1], m) - expected) < 1e-14
        end
    end
end

@testset "DiskFunction — odd Mr (Mr=15)" begin
    let D = disk(15), lmax = 15
        f  = _make_test_function(D)
        df = DiskFunction(f, D)

        # Shape
        @test ncolumns(df._coeffs[1]) == length(D.Mspan)
        for (i, m) in enumerate(Array(D.Mspan))
            @test length(column(df._coeffs[1], m)) == size_current_m(lmax, m)
        end

        # Physical accuracy
        u_comp = real.(ipsh(TriangularArrayToPSH(df._coeffs[1], D), D))
        u_ref  = real.(Inverse_laplacian(f, D))
        @test norm(u_comp - u_ref) / norm(u_ref) < 1e-12

        # ∂ζ∂ζ̄û = f̂ᵐ / 4
        f̂ = psh(f, D)
        for (i, m) in enumerate(Array(D.Mspan))
            f̂_sparse_m = Vector{ComplexF64}(f̂[D.even[:,i], i])
            @test norm(4 .* column(df._coeffs[5], m) .- f̂_sparse_m) < 1e-14
        end
    end
end
