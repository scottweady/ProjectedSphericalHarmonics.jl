
# ─── HarmonicFunction ∂ζ / ∂ζ̄ ────────────────────────────────────────────────
# Reference: evaluate the harmonic function on the grid, then apply grid-space ∂ζ/∂ζ̄.


@testset "Harmonic Function Construction - even Mr" begin


    D = disk(14, 14)
    g = vec(1.0 .+ cis.(D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 10*conj.(D.ζ).^3

    u_grid = EvaluateHarmonicFunction(h.û, D)
    @test norm(u_grid - h_exact   ) < 1e-12

end


@testset "Harmonic Function Construction - odd Mr" begin


    D = disk(15, 15)
    g = vec(1.0 .+ cis.(D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 10*conj.(D.ζ).^3

    u_grid = EvaluateHarmonicFunction(h.û, D)
    @test norm(u_grid - h_exact   ) < 1e-12


end




@testset "HarmonicFunction ∂ζ — even Mr (Mr=14)" begin

    D = disk(14, 14)
    g = vec(1.0 .+ cis.(D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 10*conj.(D.ζ).^3
    ∂ζh_exact = 1.0 .+ 0*D.ζ

    hdz     = psh(∂ζ(h, D),D)
    ref_hdz = psh(∂ζh_exact, D)

    @test size(hdz) == size(D.ζ)
    @test norm(hdz .- ref_hdz) / norm(ref_hdz) < 1e-12


end

@testset "HarmonicFunction ∂ζ — odd Mr (Mr=15)" begin
    D = disk(15, 15)
    g = vec(1.0 .+ cis.(D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 10*conj.(D.ζ).^3
    ∂ζh_exact = 1.0 .+ 0*D.ζ

    hdz     = psh(∂ζ(h, D),D)
    ref_hdz = psh(∂ζh_exact, D)

    @test size(hdz) == size(D.ζ)
    @test norm(hdz .- ref_hdz) / norm(ref_hdz) < 1e-12
end

@testset "HarmonicFunction ∂ζ̄ — even Mr (Mr=14)" begin
    D = disk(14, 14)
    g = vec(1.0 .+ cis.(D.θ) + 2*cis.(-D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 2*conj.(D.ζ) + 10*conj.(D.ζ).^3
    ∂ζ̄h_exact = 2.0 .+ 10*3*conj.(D.ζ).^2

    # ∂ĥ = similar(h.û )
    # ProjectedSphericalHarmonics.∂ζ̄_HarmonicFunction!(∂ĥ, h.û, D)    
    
    # similar(g)

    hdzb     = psh(∂ζ̄(h, D), D)

    ref_hdzb = psh(∂ζ̄h_exact, D)

    @test size(hdzb) == size(D.ζ)
    @test norm(hdzb .- ref_hdzb) / norm(ref_hdzb) < 1e-12
end

@testset "HarmonicFunction ∂ζ̄ — odd Mr (Mr=15)" begin
    D = disk(15, 15)
    g = vec(1.0 .+ cis.(D.θ) + 2*cis.(-D.θ) + 10*cis.(-3*D.θ)   )
    h = HarmonicFunction(g, D)
    h_exact = 1.0 .+ D.ζ + 2*conj.(D.ζ) + 10*conj.(D.ζ).^3
    ∂ζ̄h_exact = 2.0 .+ 10*3*conj.(D.ζ).^2

    # ∂ĥ = similar(h.û )
    # ProjectedSphericalHarmonics.∂ζ̄_HarmonicFunction!(∂ĥ, h.û, D)    
    
    # similar(g)

    hdzb     = psh(∂ζ̄(h, D), D)

    ref_hdzb = psh(∂ζ̄h_exact, D)

    @test size(hdzb) == size(D.ζ)
    @test norm(hdzb .- ref_hdzb) / norm(ref_hdzb) < 1e-12
end



