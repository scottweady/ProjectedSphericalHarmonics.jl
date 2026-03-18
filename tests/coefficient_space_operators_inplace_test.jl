using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

const LMAX_INPLACE = 31

@testset "In-place per-frequency sparse operators" begin
    mode_cases = (0, 1, 2, 3, -1, -2, -3)

    for m in mode_cases
        f̂ᵐ = randn(ComplexF64, size_current_m(LMAX_INPLACE, m))

        # Inverse Laplacian
        res_inv = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        Ĝᵐ!(res_inv, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_inv - Ĝᵐ(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_inv_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        Ĝᵐ!(res_inv_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_inv_alias - Ĝᵐ(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # First derivatives m -> m - 1
        res_dz = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_INPLACE, m))
        ∂Ĝᵐ∂ζ!(res_dz, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz - ∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dz_alias = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂Ĝᵐ∂ζ!(res_dz_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz_alias - ∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # First derivatives m -> m + 1
        res_dzbar = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_INPLACE, m))
        ∂Ĝᵐ∂ζ̄!(res_dzbar, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar - ∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dzbar_alias = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂Ĝᵐ∂ζ̄!(res_dzbar_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar_alias - ∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # Same-mode first derivatives
        res_zdz = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ζ_∂Ĝᵐ∂ζ!(res_zdz, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zdz - ζ_∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_zdz_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ζ_∂Ĝᵐ∂ζ!(res_zdz_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zdz_alias - ζ_∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_zbdzb = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ζ̄_∂Ĝᵐ∂ζ̄!(res_zbdzb, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zbdzb - ζ̄_∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_zbdzb_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ζ̄_∂Ĝᵐ∂ζ̄!(res_zbdzb_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zbdzb_alias - ζ̄_∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # Second derivatives
        res_dzbar2 = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_INPLACE, m))
        ∂²Ĝᵐ∂ζ̄²!(res_dzbar2, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar2 - ∂²Ĝᵐ∂ζ̄²(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dzbar2_alias = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂²Ĝᵐ∂ζ̄²!(res_dzbar2_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar2_alias - ∂²Ĝᵐ∂ζ̄²(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_dz2 = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_INPLACE, m))
        ∂²Ĝᵐ∂ζ²!(res_dz2, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz2 - ∂²Ĝᵐ∂ζ²(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dz2_alias = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂²Ĝᵐ∂ζ²!(res_dz2_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz2_alias - ∂²Ĝᵐ∂ζ²(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_cross = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ∂²Ĝᵐ∂ζ∂ζ̄!(res_cross, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_cross - ∂²Ĝᵐ∂ζ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_cross_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ∂²Ĝᵐ∂ζ∂ζ̄!(res_cross_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_cross_alias - ∂²Ĝᵐ∂ζ∂ζ̄(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13
    end
end
