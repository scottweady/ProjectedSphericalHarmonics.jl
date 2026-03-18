using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

# ─── Helpers ─────────────────────────────────────────────────────────────────

function _test_function_even(D)
    return exp.(-abs2.((D.ζ .- (0.3 - 0.3im)) / 0.1))
end

function _test_function_odd(D)
    return exp.(-abs2.((D.ζ .- (0.3 - 0.3im)) / 0.1)) .* D.w
end

# ─── Tests ───────────────────────────────────────────────────────────────────

@testset "NodalToTriangularArray- P = :even (even MR=32)" begin
    let MR = 64
        D     = disk(MR)
        lmax  = MR
        Mspan = vec(Array(D.Mspan))
        u     = _test_function_even(D)

        û_tri = NodalToTriangularArray(u, D)

        @test û_tri isa TriangularCoeffArray
        @test ordering(û_tri) === :fft
        @test parity(û_tri)   === :even

        # Column count matches D.Mspan
        @test ncolumns(û_tri) == length(Mspan)

        # Each column has the correct number of even-parity modes
        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(û_tri, m)) == size_current_m(lmax, m)
        end

        # psh! in-place version produces the same result
        v̂ = similar(û_tri)
        fill!(v̂, 0)
        psh!(v̂, u, D)
        @test norm(v̂ - û_tri) < 1e-12 * norm(û_tri)
    end
end

@testset "NodalToTriangularArray - P = :even  (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        lmax  = MR
        Mspan = vec(Array(D.Mspan))
        u     = _test_function_even(D)

        û_tri = NodalToTriangularArray(u, D)

        @test û_tri isa TriangularCoeffArray
        @test ncolumns(û_tri) == length(Mspan)

        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(û_tri, m)) == size_current_m(lmax, m)
        end

        v̂ = similar(û_tri)
        fill!(v̂, 0)
        psh!(v̂, u, D)
        @test norm(v̂ - û_tri) < 1e-12 * norm(û_tri)
    end
end


@testset "NodalToTriangularArray- P = :odd (even MR=32)" begin
    MR = 32
    D     = disk(MR)
    lmax  = MR
    Mspan = vec(Array(D.Mspan))
    u     = _test_function_odd(D)

    û_tri = NodalToTriangularArray(u, D; parity = :odd)

    @test û_tri isa TriangularCoeffArray
    @test ordering(û_tri) === :fft
    @test parity(û_tri)   === :odd

    # Column count matches D.Mspan
    @test ncolumns(û_tri) == length(Mspan)

    @test length(û_tri) == sum(D.odd)

    # Each column has the correct number of even-parity modes
    for (i, m) in enumerate(Mspan)
        @test length(mode_coefficients(û_tri, m)) == sum(D.odd[:,i])
    end

    # psh! in-place version produces the same result
    v̂ = similar(û_tri)
    fill!(v̂, 0)
    psh!(v̂, u, D)
    @test norm(v̂ - û_tri) < 1e-12 * norm(û_tri)

end

@testset "NodalToTriangularArray - P = :even  (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        lmax  = MR
        Mspan = vec(Array(D.Mspan))
        u     = _test_function_even(D)

        û_tri = NodalToTriangularArray(u, D)

        @test û_tri isa TriangularCoeffArray
        @test ncolumns(û_tri) == length(Mspan)

        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(û_tri, m)) == size_current_m(lmax, m)
        end

        v̂ = similar(û_tri)
        fill!(v̂, 0)
        psh!(v̂, u, D)
        @test norm(v̂ - û_tri) < 1e-12 * norm(û_tri)
    end
end


@testset "NodalToTriangularArray - P = :odd  (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        lmax  = MR
        Mspan = vec(Array(D.Mspan))
        u     = _test_function_odd(D)

        û_tri = NodalToTriangularArray(u, D; parity = :odd)

        @test û_tri isa TriangularCoeffArray
        @test ncolumns(û_tri) == length(Mspan)

        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(û_tri, m)) == sum(D.odd[:,i])
        end

        v̂ = similar(û_tri)
        fill!(v̂, 0)
        psh!(v̂, u, D)
        @test norm(v̂ - û_tri) < 1e-12 * norm(û_tri)
    end
end



@testset "TriangularArrayToPSH — round-trip - P = :even (even MR=32)" begin
    let MR = 32
        D   = disk(MR)
        u   = _test_function_even(D)

        û_tri = NodalToTriangularArray(u, D)
        û_psh = TriangularArrayToPSH(û_tri, D)
        û_ref = psh(u, D)

        # Per-frequency coefficients match the direct PSH transform
        for (i, m) in enumerate(vec(Array(D.Mspan)))
            col_ref = û_ref[D.even[:, i], i]
            col_tri = mode_coefficients(û_tri, m)
            @test norm(col_ref - col_tri) < 1e-12 * norm(col_ref)
        end

        # Nodal reconstruction is consistent
        u_rec = real.(ipsh(û_psh, D))
        u_ref = real.(ipsh(û_ref, D))
        @test norm(u_rec - u_ref) < 1e-12 * norm(u_ref)
    end
end

@testset "TriangularArrayToPSH — round-trip - P = :even  (odd MR=15)" begin
    let MR = 15
        D   = disk(MR)
        u   = _test_function_even(D)

        û_tri = NodalToTriangularArray(u, D)
        û_psh = TriangularArrayToPSH(û_tri, D)
        û_ref = psh(u, D)

        for (i, m) in enumerate(vec(Array(D.Mspan)))
            col_ref = û_ref[D.even[:, i], i]
            col_tri = mode_coefficients(û_tri, m)
            @test norm(col_ref - col_tri) < 1e-12 * norm(col_ref)
        end
    end
end




@testset "TriangularArrayToPSH — round-trip - P = :odd (even MR=32)" begin
    MR = 32
    D   = disk(MR)
    u   = _test_function_odd(D)

    û_tri = NodalToTriangularArray(u, D; parity = :odd)
    û_psh = TriangularArrayToPSH(û_tri, D)
    û_ref = psh(u, D; parity = :odd)

    # Per-frequency coefficients match the direct PSH transform
    for (i, m) in enumerate(vec(Array(D.Mspan)))
        col_ref = û_ref[D.odd[:, i], i]
        col_tri = mode_coefficients(û_tri, m)
        @test norm(col_ref - col_tri) <= 1e-12 * norm(col_ref)
    end

    # Nodal reconstruction is consistent
    u_rec = real.(ipsh(û_psh, D; parity = :odd))
    u_ref = real.(ipsh(û_ref, D; parity = :odd))
    @test norm(u_rec - u_ref) < 1e-12 * norm(u_ref)

end

@testset "TriangularArrayToPSH — round-trip - P = :odd (even MR=11)" begin
    MR = 11
    D   = disk(MR)
    u   = _test_function_odd(D)

    û_tri = NodalToTriangularArray(u, D; parity = :odd)
    û_psh = TriangularArrayToPSH(û_tri, D)
    û_ref = psh(u, D; parity = :odd)

    # Per-frequency coefficients match the direct PSH transform
    for (i, m) in enumerate(vec(Array(D.Mspan)))
        col_ref = û_ref[D.odd[:, i], i]
        col_tri = mode_coefficients(û_tri, m)
        @test norm(col_ref - col_tri) <= 1e-12 * norm(col_ref)
    end

    # Nodal reconstruction is consistent
    u_rec = real.(ipsh(û_psh, D; parity = :odd))
    u_ref = real.(ipsh(û_ref, D; parity = :odd))
    @test norm(u_rec - u_ref) < 1e-12 * norm(u_ref)

end

# ─── ipsh! tests ─────────────────────────────────────────────────────────────

@testset "ipsh! from TriangularCoeffArray — P = :even (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        u     = _test_function_even(D)
        û_tri = NodalToTriangularArray(u, D)
        û_psh = TriangularArrayToPSH(û_tri, D)

        u_ref = ipsh(û_psh, D; parity = :even)
        u_out = zeros(ComplexF64, size(D.ζ))
        ipsh!(u_out, û_tri, D)

        @test norm(u_out - u_ref) < 1e-12 * norm(u_ref)
    end
end

@testset "ipsh! from TriangularCoeffArray — P = :even (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        u     = _test_function_even(D)
        û_tri = NodalToTriangularArray(u, D)
        û_psh = TriangularArrayToPSH(û_tri, D)

        u_ref = ipsh(û_psh, D; parity = :even)
        u_out = zeros(ComplexF64, size(D.ζ))
        ipsh!(u_out, û_tri, D)

        @test norm(u_out - u_ref) < 1e-12 * norm(u_ref)
    end
end

@testset "ipsh! from TriangularCoeffArray — P = :odd (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        u     = _test_function_odd(D)
        û_tri = NodalToTriangularArray(u, D; parity = :odd)
        û_psh = TriangularArrayToPSH(û_tri, D)

        u_ref = ipsh(û_psh, D; parity = :odd)
        u_out = zeros(ComplexF64, size(D.ζ))
        ipsh!(u_out, û_tri, D)

        @test norm(u_out - u_ref) < 1e-12 * norm(u_ref)
    end
end

@testset "ipsh! from TriangularCoeffArray — P = :odd (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        u     = _test_function_odd(D)
        û_tri = NodalToTriangularArray(u, D; parity = :odd)
        û_psh = TriangularArrayToPSH(û_tri, D)

        u_ref = ipsh(û_psh, D; parity = :odd)
        u_out = zeros(ComplexF64, size(D.ζ))
        ipsh!(u_out, û_tri, D)

        @test norm(u_out - u_ref) < 1e-12 * norm(u_ref)
    end
end