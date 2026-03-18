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

function _flat(A::TriangularCoeffArray)
    return reduce(vcat, A.data)
end

# ─── Tests ───────────────────────────────────────────────────────────────────

@testset "TriangularCoeffArray — arithmetic (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)
        α = 2.5 + 1.0im

        # +
        C = A + B
        @test norm(_flat(C) - (_flat(A) + _flat(B))) == 0.0

        # -
        D_ = A - B
        @test norm(_flat(D_) - (_flat(A) - _flat(B))) == 0.0

        # unary -
        @test norm(_flat(-A) + _flat(A)) == 0.0

        # scalar * array and array * scalar
        @test norm(_flat(α * A) - α .* _flat(A)) == 0.0
        @test norm(_flat(A * α) - α .* _flat(A)) == 0.0

        # array / scalar
        @test norm(_flat(A / α) - _flat(A) ./ α) < 1e-14 * norm(_flat(A))

        # zero
        Z = zero(A)
        @test all(iszero, Z)
        @test length(Z) == length(A)
    end
end

@testset "TriangularCoeffArray — arithmetic (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)

        C = A + B
        @test norm(_flat(C) - (_flat(A) + _flat(B))) == 0.0

        @test norm(_flat(A - B) - (_flat(A) - _flat(B))) == 0.0
        @test norm(_flat(-A) + _flat(A)) == 0.0
    end
end

@testset "TriangularCoeffArray — Krylov primitives (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)
        α = 3.0 - 2.0im
        β = 0.5 + 0.5im

        # axpy!: y += α*x
        y = copy(B)
        x = copy(A)
        axpy!(α, x, y)
        @test norm(_flat(y) - (_flat(B) + α .* _flat(A))) < 1e-14 * norm(_flat(B))

        # axpby!: y = α*x + β*y
        y = copy(B)
        x = copy(A)
        axpby!(α, x, β, y)
        @test norm(_flat(y) - (α .* _flat(A) + β .* _flat(B))) < 1e-14 * norm(_flat(B))

        # rmul!
        C = copy(A)
        rmul!(C, α)
        @test norm(_flat(C) - α .* _flat(A)) < 1e-14 * norm(_flat(A))

        # lmul!
        C = copy(A)
        lmul!(α, C)
        @test norm(_flat(C) - α .* _flat(A)) < 1e-14 * norm(_flat(A))
    end
end

@testset "TriangularCoeffArray — Krylov primitives (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)
        α = -1.0im

        y = copy(B)
        axpy!(α, A, y)
        @test norm(_flat(y) - (_flat(B) + α .* _flat(A))) < 1e-14 * norm(_flat(B))
    end
end

@testset "TriangularCoeffArray — copyto! (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)

        # TCA → TCA
        B = similar(A)
        copyto!(B, A)
        @test norm(_flat(B) - _flat(A)) == 0.0

        # AbstractVector → TCA
        v = rand(ComplexF64, length(A))
        C = similar(A)
        copyto!(C, v)
        @test _flat(C) == v
    end
end

@testset "TriangularCoeffArray — convert (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)

        # Construct so the prototype is registered
        N = length(A.Mspan)
        P = parity(A)
        O = ordering(A)
        T = Float64

        v = _flat(A)
        B = convert(TriangularCoeffArray{T, N, P, O}, v)
        @test norm(_flat(B) - v) == 0.0
    end
end

@testset "TriangularCoeffArray — broadcasting (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)

        # Element-wise broadcast
        C = A .* 2
        @test norm(_flat(C) - 2 .* _flat(A)) == 0.0

        C = A .+ B
        @test norm(_flat(C) - (_flat(A) .+ _flat(B))) == 0.0
    end
end

@testset "TriangularCoeffArray — broadcasting (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR)
        B = _make_tca_random(MR)
        C = A .* 3
        @test norm(_flat(C) - 3 .* _flat(A)) == 0.0
    end
end

@testset "TriangularCoeffArray — arithmetic P=:odd (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR; parity = :odd)
        B = _make_tca_random(MR; parity = :odd)
        α = 1.5 - 0.5im

        @test norm(_flat(A + B) - (_flat(A) + _flat(B))) == 0.0
        @test norm(_flat(A - B) - (_flat(A) - _flat(B))) == 0.0
        @test norm(_flat(-A)    + _flat(A))               == 0.0
        @test norm(_flat(α * A) - α .* _flat(A))          == 0.0
        @test norm(_flat(A / α) - _flat(A) ./ α) < 1e-14 * norm(_flat(A))

        Z = zero(A)
        @test all(iszero, Z)
        @test length(Z) == length(A)
    end
end

@testset "TriangularCoeffArray — arithmetic P=:odd (odd MR=15)" begin
    let MR = 15
        A = _make_tca_random(MR; parity = :odd)
        B = _make_tca_random(MR; parity = :odd)

        @test norm(_flat(A + B) - (_flat(A) + _flat(B))) == 0.0
        @test norm(_flat(A - B) - (_flat(A) - _flat(B))) == 0.0
    end
end

@testset "TriangularCoeffArray — Krylov primitives P=:odd (even MR=32)" begin
    let MR = 32
        A = _make_tca_random(MR; parity = :odd)
        B = _make_tca_random(MR; parity = :odd)
        α = 2.0 + 1.0im
        β = -0.5im

        y = copy(B)
        axpy!(α, A, y)
        @test norm(_flat(y) - (_flat(B) + α .* _flat(A))) < 1e-14 * norm(_flat(B))

        y = copy(B)
        axpby!(α, A, β, y)
        @test norm(_flat(y) - (α .* _flat(A) + β .* _flat(B))) < 1e-14 * norm(_flat(B))

        C = copy(A)
        rmul!(C, α)
        @test norm(_flat(C) - α .* _flat(A)) < 1e-14 * norm(_flat(A))
    end
end
