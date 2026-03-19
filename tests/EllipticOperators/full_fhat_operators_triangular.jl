using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

# ── Helpers ───────────────────────────────────────────────────────────────────

function _sparse_col(f̂_dense, m)
    # Extract sparse (l+m even parity) coefficients for frequency m from the
    # dense PSH matrix (column ordering: m≥0 → m+1, m<0 → end-|m|+1).
    MTHETA = size(f̂_dense, 2)
    col    = m >= 0 ? m + 1 : MTHETA - abs(m) + 1
    return copy(f̂_dense[abs(m) + 1:2:end, col])
end

function _physical(û::TriangularCoeffArray, D)
    u = zeros(ComplexF64, size(D.ζ))
    ipsh!(u, û, D)
    return u
end

# ── Tests ─────────────────────────────────────────────────────────────────────
# NOTE: TriangularCoeffArray stores only even-parity (l+m even) modes.
# All triangular operators take no explicit lmax — it is inferred from f̂.lmax.
# For same-frequency operators, result.data[i] corresponds to f̂.Mspan[i].
# For frequency-shifting operators (shift ±k), the output for input frequency m
# lives at mode_coefficients(result, m ± k), not at result.data[i].

for MR in (10, 11)
    MR = 11
    D    = disk(MR, MR)
    X    = real.(D.ζ)
    Y    = imag.(D.ζ)
    lmax = D.Mr
    D.Mspan


    f = exp.(-X .* Y) .* cos.(X .^ 2) .+ im .* exp.(-X .* Y) .* sin.(Y .^ 2)

    f̂_dense = psh(f, D)
    f̂_tri   = psh_triangular(f, D)
    tri_lmax = f̂_tri.lmax   # inferred from the triangular array itself

    # ── Ĝ (same-frequency) ───────────────────────────────────────────────────
    @testset "Ĝ per-frequency vs Ĝᵐ (MR=$MR)" begin
        result = Ĝ(f̂_tri)
        for (i, m) in enumerate(f̂_tri.Mspan)
            ref_m = Ĝᵐ(_sparse_col(f̂_dense, m), tri_lmax, m; aliasing=false)
            @test norm(result.data[i] - ref_m) < 1e-12
        end
    end

    @testset "Ĝ! in-place vs out-of-place (MR=$MR)" begin
        res_op = Ĝ(f̂_tri)
        res_ip = similar(f̂_tri)
        Ĝ!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ζ_∂Ĝ∂ζ (same-frequency) ──────────────────────────────────────────────
    @testset "ζ_∂Ĝ∂ζ per-frequency vs ζ_∂Ĝᵐ∂ζ (MR=$MR)" begin
        result = ζ_∂Ĝ∂ζ(f̂_tri)
        for (i, m) in enumerate(f̂_tri.Mspan)
            ref_m = ζ_∂Ĝᵐ∂ζ(_sparse_col(f̂_dense, m), tri_lmax, m; aliasing=false)
            @test norm(result.data[i] - ref_m) < 1e-12
        end
    end

    @testset "ζ_∂Ĝ∂ζ! in-place vs out-of-place (MR=$MR)" begin
        res_op = ζ_∂Ĝ∂ζ(f̂_tri)
        res_ip = similar(f̂_tri)
        ζ_∂Ĝ∂ζ!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ζ̄_∂Ĝ∂ζ̄ (same-frequency) ─────────────────────────────────────────────
    @testset "ζ̄_∂Ĝ∂ζ̄ per-frequency vs ζ̄_∂Ĝᵐ∂ζ̄ (MR=$MR)" begin
        result = ζ̄_∂Ĝ∂ζ̄(f̂_tri)
        for (i, m) in enumerate(f̂_tri.Mspan)
            ref_m = ζ̄_∂Ĝᵐ∂ζ̄(_sparse_col(f̂_dense, m), tri_lmax, m; aliasing=false)
            @test norm(result.data[i] - ref_m) < 1e-12
        end
    end

    @testset "ζ̄_∂Ĝ∂ζ̄! in-place vs out-of-place (MR=$MR)" begin
        res_op = ζ̄_∂Ĝ∂ζ̄(f̂_tri)
        res_ip = similar(f̂_tri)
        ζ̄_∂Ĝ∂ζ̄!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── r_∂Ĝ∂r (same-frequency) ──────────────────────────────────────────────
    @testset "r_∂Ĝ∂r per-frequency vs r_∂Ĝᵐ∂r (MR=$MR)" begin
        result = r_∂Ĝ∂r(f̂_tri)
        for (i, m) in enumerate(f̂_tri.Mspan)
            ref_m = r_∂Ĝᵐ∂r(_sparse_col(f̂_dense, m), tri_lmax, m; aliasing=false)
            @test norm(result.data[i] - ref_m) < 1e-12
        end
    end

    @testset "r_∂Ĝ∂r! in-place vs out-of-place (MR=$MR)" begin
        res_op = r_∂Ĝ∂r(f̂_tri)
        res_ip = similar(f̂_tri)
        r_∂Ĝ∂r!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ∂²Ĝ∂ζ∂ζ̄ (same-frequency) ────────────────────────────────────────────
    # ∂²Ĝ∂ζ∂ζ̄ = ¼ · I on the PSH coefficient space, so compare directly.
    @testset "∂²Ĝ∂ζ∂ζ̄ vs f̂/4 (MR=$MR)" begin
        result = ∂²Ĝ∂ζ∂ζ̄(f̂_tri)
        @test norm(result - f̂_tri / 4) < 1e-14
    end

    @testset "∂²Ĝ∂ζ∂ζ̄! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂²Ĝ∂ζ∂ζ̄(f̂_tri)
        res_ip = similar(f̂_tri)
        ∂²Ĝ∂ζ∂ζ̄!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ∂Ĝ∂ζ (shift −1): output at mode_coefficients(result, m-1) ───────────
    # Use in-place reference to match the triangular output column size exactly.
    @testset "∂Ĝ∂ζ per-frequency vs ∂Ĝᵐ∂ζ (MR=$MR)" begin
        result = ∂Ĝ∂ζ(f̂_tri)
        for m in f̂_tri.Mspan
            size_current_m(tri_lmax, m-1) == 0 && continue
            res_col = mode_coefficients(result, m-1)
            ref_col = zeros(ComplexF64, length(res_col))
            ∂Ĝᵐ∂ζ!(ref_col, _sparse_col(f̂_dense, m), tri_lmax, m)
            @test norm(res_col - ref_col) < 1e-12
        end
    end

    @testset "∂Ĝ∂ζ! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂Ĝ∂ζ(f̂_tri)
        res_ip = similar(f̂_tri)
        fill!(res_ip, 0)
        ∂Ĝ∂ζ!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ∂Ĝ∂ζ̄ (shift +1): output at mode_coefficients(result, m+1) ──────────
    @testset "∂Ĝ∂ζ̄ per-frequency vs ∂Ĝᵐ∂ζ̄ (MR=$MR)" begin
        result = ∂Ĝ∂ζ̄(f̂_tri)
        for m in f̂_tri.Mspan
            size_current_m(tri_lmax, m+1) == 0 && continue
            res_col = mode_coefficients(result, m+1)
            ref_col = zeros(ComplexF64, length(res_col))
            ∂Ĝᵐ∂ζ̄!(ref_col, _sparse_col(f̂_dense, m), tri_lmax, m)
            @test norm(res_col - ref_col) < 1e-12
        end
    end

    @testset "Modified ∂Ĝ∂ζ̄! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂Ĝ∂ζ̄(f̂_tri)
        res_ip = similar(f̂_tri)
        fill!(res_ip, 0)
        ∂Ĝ∂ζ̄!(res_ip, f̂_tri)
        println(norm(res_op))
        println(norm(res_ip))
        @test norm(res_op - res_ip) < 1e-14
    end

    
    @testset "∂Ĝ∂ζ̄! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂Ĝ∂ζ̄(f̂_tri)
        res_ip = similar(f̂_tri)
        fill!(res_ip, 0)
        ∂Ĝ∂ζ̄!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end


    # ── ∂²Ĝ∂ζ̄² (shift +2): output at mode_coefficients(result, m+2) ─────────
    @testset "∂²Ĝ∂ζ̄² per-frequency vs ∂²Ĝᵐ∂ζ̄² (MR=$MR)" begin
        result = ∂²Ĝ∂ζ̄²(f̂_tri)
        for m in f̂_tri.Mspan
            size_current_m(tri_lmax, m+2) == 0 && continue
            res_col = mode_coefficients(result, m+2)
            ref_col = zeros(ComplexF64, length(res_col))
            ∂²Ĝᵐ∂ζ̄²!(ref_col, _sparse_col(f̂_dense, m), tri_lmax, m)
            @test norm(res_col - ref_col) < 1e-12
        end
    end

    @testset "∂²Ĝ∂ζ̄²! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂²Ĝ∂ζ̄²(f̂_tri)
        res_ip = similar(f̂_tri)
        fill!(res_ip, 0)
        ∂²Ĝ∂ζ̄²!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end

    # ── ∂²Ĝ∂ζ² (shift −2): output at mode_coefficients(result, m-2) ─────────
    @testset "∂²Ĝᵐ∂ζ² per-frequency vs ∂²Ĝᵐ∂ζ² (MR=$MR)" begin
        result = ∂²Ĝ∂ζ²(f̂_tri)
        for m in f̂_tri.Mspan
            size_current_m(tri_lmax, m-2) == 0 && continue
            res_col = mode_coefficients(result, m-2)
            ref_col = zeros(ComplexF64, length(res_col))
            ∂²Ĝᵐ∂ζ²!(ref_col, _sparse_col(f̂_dense, m), tri_lmax, m)
            @test norm(res_col - ref_col) < 1e-12
        end
    end

    @testset "∂²Ĝ∂ζ²! in-place vs out-of-place (MR=$MR)" begin
        res_op = ∂²Ĝ∂ζ²(f̂_tri)
        res_ip = similar(f̂_tri)
        fill!(res_ip, 0)
        ∂²Ĝ∂ζ²!(res_ip, f̂_tri)
        @test norm(res_op - res_ip) < 1e-14
    end
end
