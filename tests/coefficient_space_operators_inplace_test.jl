using Test
using ProjectedSphericalHarmonics

const LMAX_INPLACE = 31

@testset "In-place per-frequency sparse operators" begin
    mode_cases = (0, 1, 2, 3, -1, -2, -3)

    for m in mode_cases
        f̂ᵐ = randn(ComplexF64, size_current_m(LMAX_INPLACE, m))

        # Inverse Laplacian
        res_inv = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        Inverse_laplacian_coef_m_sparse!(res_inv, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_inv - Inverse_laplacian_coef_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_inv_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        Inverse_laplacian_coef_m_sparse!(res_inv_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_inv_alias - Inverse_laplacian_coef_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # First derivatives m -> m - 1
        res_dz = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_INPLACE, m))
        ∂ζΔ⁻¹_m_sparse!(res_dz, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz - ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dz_alias = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂ζΔ⁻¹_m_sparse!(res_dz_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz_alias - ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # First derivatives m -> m + 1
        res_dzbar = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_INPLACE, m))
        ∂ζ̄Δ⁻¹_m_sparse!(res_dzbar, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar - ∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dzbar_alias = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂ζ̄Δ⁻¹_m_sparse!(res_dzbar_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar_alias - ∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # Same-mode first derivatives
        res_zdz = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ζ∂ζΔ⁻¹_m_sparse!(res_zdz, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zdz - ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_zdz_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ζ∂ζΔ⁻¹_m_sparse!(res_zdz_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zdz_alias - ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_zbdzb = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ζ̄∂ζ̄Δ⁻¹_m_sparse!(res_zbdzb, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zbdzb - ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_zbdzb_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ζ̄∂ζ̄Δ⁻¹_m_sparse!(res_zbdzb_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_zbdzb_alias - ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        # Second derivatives
        res_dzbar2 = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_INPLACE, m))
        ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(res_dzbar2, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar2 - ∂ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dzbar2_alias = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂ζ̄∂ζ̄Δ⁻¹_m_sparse!(res_dzbar2_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dzbar2_alias - ∂ζ̄∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_dz2 = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_INPLACE, m))
        ∂ζ∂ζΔ⁻¹_m_sparse!(res_dz2, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz2 - ∂ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_dz2_alias = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_INPLACE, m; aliasing=true))
        ∂ζ∂ζΔ⁻¹_m_sparse!(res_dz2_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_dz2_alias - ∂ζ∂ζΔ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13

        res_cross = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m))
        ∂ζ∂ζ̄Δ⁻¹_m_sparse!(res_cross, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_cross - ∂ζ∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=false)) < 1e-13

        res_cross_alias = fill(1.0 + 1.0im, size_current_m(LMAX_INPLACE, m; aliasing=true))
        ∂ζ∂ζ̄Δ⁻¹_m_sparse!(res_cross_alias, f̂ᵐ, LMAX_INPLACE, m)
        @test norm(res_cross_alias - ∂ζ∂ζ̄Δ⁻¹_m_sparse(f̂ᵐ, LMAX_INPLACE, m; aliasing=true)) < 1e-13
    end
end
