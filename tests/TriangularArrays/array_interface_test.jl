using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

# ─── Helpers ─────────────────────────────────────────────────────────────────

function _make_tca_random(MR::Int; parity::Symbol = :even, ordering::Symbol = :fft)
    D     = disk(MR)
    Mspan = vec(Array(D.Mspan))
    lmax  = MR
    data  = if parity == :even
        [rand(ComplexF64, length(abs(m):2:lmax))     for m in Mspan]
    else
        [rand(ComplexF64, length(abs(m)+1:2:lmax))   for m in Mspan]
    end
    return TriangularCoeffArray(Mspan, data; parity = parity, ordering = ordering)
end

# ─── Tests ───────────────────────────────────────────────────────────────────

@testset "TriangularCoeffArray — array interface (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)

        # size and length are consistent
        @test size(A) == (length(A),)
        @test length(A) == sum(length, A.data)

        # getindex / setindex! round-trip
        v_orig = A[1]
        A[1]   = 99.0 + 7.0im
        @test A[1] == 99.0 + 7.0im
        A[1]   = v_orig

        # Linear indexing matches flat concatenation
        flat = reduce(vcat, A.data)
        for k in eachindex(A)
            @test A[k] == flat[k]
        end

        # IndexStyle
        @test Base.IndexStyle(typeof(A)) === IndexLinear()
    end
end

@testset "TriangularCoeffArray — array interface (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR)

        flat = reduce(vcat, A.data)
        for k in eachindex(A)
            @test A[k] == flat[k]
        end
    end
end

@testset "TriangularCoeffArray — parity / ordering accessors (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR

        A_even_fft = TriangularCoeffArray{Float64}(lmax, Mspan; parity = :even, ordering = :fft)
        A_odd_nat  = TriangularCoeffArray{Float64}(lmax, Mspan; parity = :odd,  ordering = :natural)

        @test parity(A_even_fft)   === :even
        @test ordering(A_even_fft) === :fft
        @test parity(A_odd_nat)    === :odd
        @test ordering(A_odd_nat)  === :natural
    end
end

@testset "TriangularCoeffArray — mode_coefficients (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)

        # mode_coefficients returns the same object as A.data[col]
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        for (i, m) in enumerate(Mspan)
            @test mode_coefficients(A, m) === A.data[i]
        end
    end
end

@testset "TriangularCoeffArray — mode_coefficients (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR)
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        for (i, m) in enumerate(Mspan)
            @test mode_coefficients(A, m) === A.data[i]
        end
    end
end

@testset "TriangularCoeffArray — ncolumns (even MR=32 and odd MR=15)" begin
    for MR in (32, 15)
        A = _make_tca_random(MR)
        D = disk(MR)
        @test ncolumns(A) == length(D.Mspan)
    end
end

@testset "TriangularCoeffArray — mode_coefficients odd parity column sizes (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        A = _make_tca_random(MR; parity = :odd)

        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(A, m)) == sum(D.odd[:, i])
        end
    end
end

@testset "TriangularCoeffArray — mode_coefficients odd parity column sizes (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        A = _make_tca_random(MR; parity = :odd)

        for (i, m) in enumerate(Mspan)
            @test length(mode_coefficients(A, m)) == sum(D.odd[:, i])
        end
    end
end

@testset "TriangularCoeffArray — mode_coefficients :natural ordering (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR
        data  = [rand(ComplexF64, length(abs(m):2:lmax)) for m in Mspan]
        A_fft = TriangularCoeffArray(Mspan, data; parity = :even, ordering = :fft)

        # Build a natural-ordered array manually
        using FFTW: fftshift
        nat_Mspan = fftshift(Mspan)
        nat_data  = [copy(v) for v in fftshift(A_fft.data)]
        A_nat = TriangularCoeffArray(nat_Mspan, nat_data; parity = :even, ordering = :natural)

        # Both should return the same coefficients for each m
        for m in Mspan
            @test mode_coefficients(A_fft, m) == mode_coefficients(A_nat, m)
        end
    end
end
