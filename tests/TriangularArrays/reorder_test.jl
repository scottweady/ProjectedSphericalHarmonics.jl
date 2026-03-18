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

@testset "circshift_fft_to_natural (even MR=32)" begin
    let MR = 32
        A   = _make_tca_random(MR)
        Anat = circshift_fft_to_natural(A)

        # ordering changed
        @test ordering(Anat) === :natural
        @test parity(Anat)   === parity(A)

        # Mspan is the fftshift of the original
        using FFTW: fftshift
        @test Anat.Mspan == fftshift(A.Mspan)

        # mode_coefficients returns the same values for every m
        for m in A.Mspan
            @test mode_coefficients(Anat, m) == mode_coefficients(A, m)
        end
    end
end

@testset "circshift_fft_to_natural (odd MR=15)" begin
    let MR = 15
        A    = _make_tca_random(MR)
        Anat = circshift_fft_to_natural(A)

        @test ordering(Anat) === :natural
        for m in A.Mspan
            @test mode_coefficients(Anat, m) == mode_coefficients(A, m)
        end
    end
end

@testset "circshift_natural_to_fft (even MR=32)" begin
    let MR = 32
        A    = _make_tca_random(MR)
        Anat = circshift_fft_to_natural(A)
        Afft = circshift_natural_to_fft(Anat)

        # Round-trip restores ordering and Mspan
        @test ordering(Afft) === :fft
        @test Afft.Mspan == A.Mspan

        # Round-trip restores all coefficients
        for m in A.Mspan
            @test mode_coefficients(Afft, m) == mode_coefficients(A, m)
        end
    end
end

@testset "circshift_natural_to_fft (odd MR=15)" begin
    let MR = 15
        A    = _make_tca_random(MR)
        Anat = circshift_fft_to_natural(A)
        Afft = circshift_natural_to_fft(Anat)

        @test ordering(Afft) === :fft
        @test Afft.Mspan == A.Mspan
        for m in A.Mspan
            @test mode_coefficients(Afft, m) == mode_coefficients(A, m)
        end
    end
end

@testset "reorder parity preserved (even MR=32)" begin
    let MR = 32
        for par in (:even, :odd)
            A    = _make_tca_random(MR; parity = par)
            Anat = circshift_fft_to_natural(A)
            Afft = circshift_natural_to_fft(Anat)
            @test parity(Anat) === par
            @test parity(Afft) === par
        end
    end
end
