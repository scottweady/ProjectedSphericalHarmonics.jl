using Test
using LinearAlgebra
using ProjectedSphericalHarmonics

# ─── helpers ──────────────────────────────────────────────────────────────────

function _make_density(D)
    x, y = real.(D.ζ), imag.(D.ζ)
    return (1 .- x.^2 .- y.^2) .* exp.(-2*cos.(x))
end

# Reference: reconstruct grid-space u from DiskFunction coefficients, then
# apply the PSH grid-space derivative. Used as ground truth.
function _ref_∂ζ(df, D)
    u = ipsh(TriangularArrayToPSH(df._coeffs[1], D), D)
    return ∂ζ(u, D)
end
function _ref_∂ζ̄(df, D)
    u = ipsh(TriangularArrayToPSH(df._coeffs[1], D), D)
    return ∂ζ̄(u, D)
end

# ─── DiskFunction ∂ζ / ∂ζ̄ ────────────────────────────────────────────────────

@testset "DiskFunction ∂ζ — even Mr (Mr=40)" begin
    let D = disk(40)
        f  = _make_density(D)
        df = DiskFunction(f, D)

        dz     = psh(∂ζ(df, D), D)
        ref_dz = psh(_ref_∂ζ(df, D), D)

        for i in eachindex(D.Mspan)
            err = norm(dz[:, i] - ref_dz[:,i])
            if err >= 1e-12
                println("Frequency m = $(D.Mspan[i]) : ", err)
            end
        end

        @test size(∂ζ(df, D)) == size(D.ζ)
        @test norm(dz .- ref_dz) / norm(ref_dz) < 1e-12
    end
end

@testset "DiskFunction ∂ζ — odd Mr (Mr=41)" begin
    let D = disk(41)
        f  = _make_density(D)
        df = DiskFunction(f, D)

        dz     = ∂ζ(df, D)
        ref_dz = _ref_∂ζ(df, D)

        @test size(dz) == size(D.ζ)
        @test norm(dz .- ref_dz) / norm(ref_dz) < 1e-12
    end
end

@testset "DiskFunction ∂ζ̄ — even Mr (Mr=40)" begin
    let D = disk(40)
        f  = _make_density(D)
        df = DiskFunction(f, D)

        dzb     = ∂ζ̄(df, D)
        ref_dzb = _ref_∂ζ̄(df, D)

        @test size(dzb) == size(D.ζ)
        @test norm(dzb .- ref_dzb) / norm(ref_dzb) < 1e-12
    end
end

@testset "DiskFunction ∂ζ̄ — odd Mr (Mr=41)" begin
    let D = disk(41)
        f  = _make_density(D)
        df = DiskFunction(f, D)

        dzb     = ∂ζ̄(df, D)
        ref_dzb = _ref_∂ζ̄(df, D)

        @test size(dzb) == size(D.ζ)
        @test norm(dzb .- ref_dzb) / norm(ref_dzb) < 1e-12
    end
end
