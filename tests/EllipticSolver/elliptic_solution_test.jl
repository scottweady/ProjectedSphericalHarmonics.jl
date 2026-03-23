using Test
using ProjectedSphericalHarmonics
using LinearAlgebra

# ─── Helpers ──────────────────────────────────────────────────────────────────

function _es_make_density(D)
    x = real.(D.ζ)
    y = imag.(D.ζ)
    return (1 .- x.^2 .- y.^2) .* cos.(π .* x) .* sin.(2π .* y)
end

# ─── Tests ────────────────────────────────────────────────────────────────────

@testset "EllipticSolution" begin

for Mr in [16, 17]

    D     = disk(Mr, Mr)
    Mspan = vec(Array(D.Mspan))
    Nf    = length(Mspan)

    f     = _es_make_density(D)
    f̂_tri = psh_triangular(f, D)

    # Harmonic: m=1 and m=2 modes — enough to exercise all derivative paths
    ĥ      = zeros(ComplexF64, Nf)
    ĥ[2]   = 0.8 + 0.3im    # m=1
    ĥ[3]   = 0.4 + 0.2im    # m=2
    zero_ĥ = zeros(ComplexF64, Nf)
    zero_μ = TriangularCoeffArray{Float64}(D.Mr, Mspan)

    sol        = EllipticSolution(ĥ, f̂_tri)       # density + harmonic
    sol_zero_h = EllipticSolution(zero_ĥ, f̂_tri)  # density only
    sol_harm   = EllipticSolution(ĥ, zero_μ)       # harmonic only

    # DiskFunction reference with all derivative slots pre-computed
    df = DiskFunction(f̂_tri, D; derivatives=[(1,0),(0,1),(2,0)])

    @testset "Mr=$Mr" begin

        # ── Mspan accessor ────────────────────────────────────────────────────
        @testset "Mspan accessor" begin
            @test sol.Mspan === sol.density.Mspan
        end

        # ── evaluate (0,0): density-only matches DiskFunction ────────────────
        @testset "evaluate (0,0) — density only" begin
            u_sol = evaluate(sol_zero_h, D)
            u_ref = evaluate(df, D)
            @test isapprox(u_sol, u_ref; rtol=1e-12)
        end

        # ── evaluate (0,0): additivity (sol = sol_zero_h + sol_harm) ─────────
        @testset "evaluate (0,0) — additivity" begin
            u_full = evaluate(sol, D)
            u_dens = evaluate(sol_zero_h, D)
            u_harm = evaluate(sol_harm, D)
            @test isapprox(u_full, u_dens .+ u_harm; rtol=1e-12)
        end

        # ── evaluate! fills in-place consistently with evaluate ───────────────
        @testset "evaluate! matches evaluate" begin
            u_out = zeros(ComplexF64, length(D.r), Nf)
            evaluate!(u_out, sol, D)
            @test isapprox(u_out, evaluate(sol, D); rtol=1e-12)
        end

        # ── derivative (1,0): density part matches DiskFunction ───────────────
        @testset "evaluate! (1,0) — density only" begin
            u_ref = evaluate(df, 1, 0, D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol_zero_h, D; derivative=(1,0))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── derivative (0,1): density part matches DiskFunction ───────────────
        @testset "evaluate! (0,1) — density only" begin
            u_ref = evaluate(df, 0, 1, D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol_zero_h, D; derivative=(0,1))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── derivative (2,0): density part matches DiskFunction ───────────────
        @testset "evaluate! (2,0) — density only" begin
            u_ref = evaluate(df, 2, 0, D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol_zero_h, D; derivative=(2,0))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── derivative (1,0): coefficient-space vs grid-space reference ───────
        @testset "evaluate! (1,0) — with harmonic" begin
            u_sol = evaluate(sol, D)
            u_ref = ∂ζ(u_sol, D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol, D; derivative=(1,0))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── derivative (0,1): coefficient-space vs grid-space reference ───────
        @testset "evaluate! (0,1) — with harmonic" begin
            u_sol = evaluate(sol, D)
            u_ref = ∂ζ̄(u_sol, D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol, D; derivative=(0,1))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── derivative (2,0): coefficient-space vs ∂ζ applied twice on grid ──
        @testset "evaluate! (2,0) — with harmonic" begin
            u_sol = evaluate(sol, D)
            u_ref = ∂ζ(∂ζ(u_sol, D), D)
            u_out = zeros(ComplexF64, size(u_ref)...)
            evaluate!(u_out, sol, D; derivative=(2,0))
            @test isapprox(u_out, u_ref; rtol=1e-12)
        end

        # ── unsupported derivative throws ArgumentError ───────────────────────
        @testset "unsupported derivative throws" begin
            u_out = zeros(ComplexF64, length(D.r), Nf)
            @test_throws ArgumentError evaluate!(u_out, sol, D; derivative=(3,0))
            @test_throws ArgumentError evaluate!(u_out, sol, D; derivative=(1,1))
        end

        # ── arithmetic ────────────────────────────────────────────────────────
        @testset "arithmetic" begin
            α = 2.5

            @test isapprox(evaluate(α * sol, D),   α .* evaluate(sol, D);  rtol=1e-12)
            @test isapprox(evaluate(sol * α, D),   α .* evaluate(sol, D);  rtol=1e-12)
            @test isapprox(evaluate(sol / α, D),   evaluate(sol, D) ./ α;  rtol=1e-12)
            @test isapprox(evaluate(-sol, D),     .-evaluate(sol, D);       rtol=1e-12)

            @test isapprox(evaluate(sol + sol_zero_h, D),
                           evaluate(sol, D) .+ evaluate(sol_zero_h, D); rtol=1e-12)
            @test isapprox(evaluate(sol - sol_zero_h, D),
                           evaluate(sol, D) .- evaluate(sol_zero_h, D); rtol=1e-12)

            # lmul! mutates both fields in-place
            sol_copy = deepcopy(sol)
            lmul!(α, sol_copy)
            @test isapprox(sol_copy.harmonic, α .* sol.harmonic; rtol=1e-15)
            @test isapprox(evaluate(sol_copy, D), α .* evaluate(sol, D); rtol=1e-12)
        end

    end  # Mr=$Mr

end  # for Mr

end  # @testset EllipticSolution
