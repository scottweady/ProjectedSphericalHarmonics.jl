using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

# ─── Helpers ─────────────────────────────────────────────────────────────────

function _make_tca(MR::Int; parity::Symbol = :even, ordering::Symbol = :fft)
    D    = disk(MR)
    Mspan = vec(Array(D.Mspan))
    lmax  = MR
    return TriangularCoeffArray{Float64}(lmax, Mspan; parity = parity, ordering = ordering)
end

# ─── Tests ───────────────────────────────────────────────────────────────────

@testset "TriangularCoeffArray — struct - even expansion - (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR

        A = TriangularCoeffArray{Float64}(lmax, Mspan)

        # Type parameters
        @test A isa TriangularCoeffArray
        @test eltype(A) == ComplexF64

        # Mspan stored correctly
        @test A.Mspan == Mspan

        # Column lengths match expected mode count
        for (i, m) in enumerate(Mspan)
            expected = sum(D.even[:,i])
            @test length(A.data[i]) == expected
        end

        # total length matches expected length

        @test length(A) == sum(D.even)

        # Offsets are consistent
        @test A._offsets[1] == 0
        for i in eachindex(A.data)
            @test A._offsets[i+1] == A._offsets[i] + length(A.data[i])
        end
        @test A._offsets[end] == length(A)

        # All zeros after construction
        @test all(iszero, A)
    end
end

@testset "TriangularCoeffArray — struct - even expansion - (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR

        A = TriangularCoeffArray{Float64}(lmax, Mspan)

        @test A isa TriangularCoeffArray
        @test eltype(A) == ComplexF64

        for (i, m) in enumerate(Mspan)
            expected = sum(D.even[:,i])
            @test length(A.data[i]) == expected
        end

        @test length(A) == sum(D.even)

        @test A._offsets[end] == length(A)
        @test all(iszero, A)
    end
end

@testset "TriangularCoeffArray — struct - odd expansion - (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR

        A = TriangularCoeffArray{Float64}(lmax, Mspan; parity = :odd)

        # Type parameters
        @test A isa TriangularCoeffArray
        @test eltype(A) == ComplexF64

        # Mspan stored correctly
        @test A.Mspan == Mspan

        # Column lengths match expected mode count
        for (i, m) in enumerate(Mspan)
            expected = sum(D.odd[:,i])
            @test length(A.data[i]) == expected
        end

        # total length matches expected length

        @test length(A) == sum(D.odd)

        # Offsets are consistent
        @test A._offsets[1] == 0
        for i in eachindex(A.data)
            @test A._offsets[i+1] == A._offsets[i] + length(A.data[i])
        end
        @test A._offsets[end] == length(A)

        # All zeros after construction
        @test all(iszero, A)
    end
end

@testset "TriangularCoeffArray — struct - odd expansion - (odd MR=15)" begin
    let MR = 15
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR

        A = TriangularCoeffArray{Float64}(lmax, Mspan; parity = :odd)

        @test A isa TriangularCoeffArray
        @test eltype(A) == ComplexF64

        for (i, m) in enumerate(Mspan)
            expected = sum(D.odd[:,i])
            @test length(A.data[i]) == expected
        end

        @test length(A) == sum(D.odd)

        @test A._offsets[end] == length(A)
        @test all(iszero, A)
    end
end




@testset "TriangularCoeffArray — similar / copy / fill! (even MR=32)" begin
    let MR = 32
        A = _make_tca(MR)
        fill!(A, 1.0 + 2.0im)

        B = similar(A)
        @test B isa TriangularCoeffArray
        @test length(B) == length(A)
        @test B.Mspan == A.Mspan

        C = copy(A)
        @test C isa TriangularCoeffArray
        @test length(C) == length(A)
        @test norm(C - A) == 0.0

        # Mutating C does not affect A
        fill!(C, 0.0)
        @test !all(iszero, A)
    end
end

@testset "TriangularCoeffArray — similar / copy / fill! (odd MR=15)" begin
    let MR = 15
        A = _make_tca(MR)
        fill!(A, 3.0 - 1.0im)

        C = copy(A)
        @test norm(C - A) == 0.0

        fill!(C, 0.0)
        @test !all(iszero, A)
    end
end

@testset "TriangularCoeffArray — data constructor - even expansion (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR
        data  = [rand(ComplexF64, length(abs(m):2:lmax)) for m in Mspan]

        A = TriangularCoeffArray(Mspan, data; parity = :even, ordering = :fft)
        @test A isa TriangularCoeffArray
        @test length(A) == sum(length, data)

        # Round-trip: data is preserved
        for (i, v) in enumerate(data)
            @test A.data[i] == v
        end
    end
end



@testset "TriangularCoeffArray — data constructor - odd expansion (even MR=32)" begin
    let MR = 32
        D     = disk(MR)
        Mspan = vec(Array(D.Mspan))
        lmax  = MR
        data  = [rand(ComplexF64, length(abs(m)+1:2:lmax)) for m in Mspan]

        A = TriangularCoeffArray(Mspan, data; parity = :odd, ordering = :fft)
        @test A isa TriangularCoeffArray
        @test length(A) == sum(length, data)

        # Round-trip: data is preserved
        for (i, v) in enumerate(data)
            @test A.data[i] == v
        end
    end
end
