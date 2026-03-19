using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

# ── Shared setup ──────────────────────────────────────────────────────────────

const MR     = 32
const MTHETA = 32
const D      = disk(MR, MTHETA)
const ZETA   = D.ζ
const X      = real.(ZETA)
const Y      = imag.(ZETA)
const lmax   = MR

# ── Helpers ───────────────────────────────────────────────────────────────────

function mode_column_index(m)
    return m >= 0 ? m + 1 : size(D.Mspan, 2) - abs(m) + 1
end

function sparse_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:2:end, mode_column_index(m)])
end

function same_mode_reference_coefficients(uhat, m)
    col = mode_column_index(m)
    return copy(uhat[D.even[:, col], col])
end

function assembled_r_dot_grad_reference(uhat)
    reference = zeros(ComplexF64, size(uhat))
    max_mode  = min(div(size(uhat, 2) - 1, 2), size(uhat, 1) - 2)

    mode0 = @view uhat[1:2:end, 1]
    reference[1:2:end, 1] .= ζ_∂Ĝᵐ∂ζ(mode0, lmax, 0; aliasing=false) .+
                             ζ̄_∂Ĝᵐ∂ζ̄(mode0, lmax, 0; aliasing=false)

    for m in 1:max_mode
        mode_pos = @view uhat[m+1:2:end, m+1]
        reference[m+1:2:end, m+1] .= ζ_∂Ĝᵐ∂ζ(mode_pos, lmax, m; aliasing=false) .+
                                     ζ̄_∂Ĝᵐ∂ζ̄(mode_pos, lmax, m; aliasing=false)

        mode_neg = @view uhat[m+1:2:end, end-(m-1)]
        reference[m+1:2:end, end-(m-1)] .= ζ_∂Ĝᵐ∂ζ(mode_neg, lmax, -m; aliasing=false) .+
                                           ζ̄_∂Ĝᵐ∂ζ̄(mode_neg, lmax, -m; aliasing=false)
    end

    return reference
end

function mode_test_function(l, m)
    return 2 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 4 .* ylm.(l + 2, m, ZETA)
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Inverse_laplacian wrapper" begin
    mode_cases = ((8, 2), (10, -2), (2, 0), (4, 0), (10, 0))

    for (l, m) in mode_cases
        u               = mode_test_function(l, m)
        u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
        @test norm(ipsh(Inverse_laplacian(psh(u, D)), D) - u_inv_reference) < 1e-12
    end

    u               = exp.(-X .* Y) .* cos.(X .^ 2) .+ im .* exp.(-X .* Y) .* sin.(Y .^ 2)
    u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
    @test norm(ipsh(Inverse_laplacian(psh(u, D)), D) - u_inv_reference) < 1e-12
end

@testset "Grid-space ∂ζ and ∂ζ̄ operators" begin
    mode_cases = ((8, 2), (9, 3), (10, 2), (10, -2), (11, -3), (10, 0))
    h        = 1e-6
    ε        = 10h
    interior = abs.(ZETA) .< 1 - ε

    for (l, m) in mode_cases
        μ       = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA) .+ 2 .* ylm.(l + 4, m, ZETA)
        ∂ζ_μ    = ∂ζ(μ, D)[interior]
        ∂ζ̄_μ   = ∂ζ̄(μ, D)[interior]
        ζ_int   = ZETA[interior]
        x_int   = real.(ζ_int)
        y_int   = imag.(ζ_int)

        ζ_x⁺  = (x_int .+ h) .+ im .* y_int
        ζ_x⁻  = (x_int .- h) .+ im .* y_int
        μ_x⁺  = ylm.(abs(m), m, ζ_x⁺) .+ 2 .* ylm.(l + 2, m, ζ_x⁺) .+ 2 .* ylm.(l + 4, m, ζ_x⁺)
        μ_x⁻  = ylm.(abs(m), m, ζ_x⁻) .+ 2 .* ylm.(l + 2, m, ζ_x⁻) .+ 2 .* ylm.(l + 4, m, ζ_x⁻)
        ∂x_fd = (μ_x⁺ .- μ_x⁻) ./ (2h)

        ζ_y⁺  = x_int .+ im .* (y_int .+ h)
        ζ_y⁻  = x_int .+ im .* (y_int .- h)
        μ_y⁺  = ylm.(abs(m), m, ζ_y⁺) .+ 2 .* ylm.(l + 2, m, ζ_y⁺) .+ 2 .* ylm.(l + 4, m, ζ_y⁺)
        μ_y⁻  = ylm.(abs(m), m, ζ_y⁻) .+ 2 .* ylm.(l + 2, m, ζ_y⁻) .+ 2 .* ylm.(l + 4, m, ζ_y⁻)
        ∂y_fd = (μ_y⁺ .- μ_y⁻) ./ (2h)

        @testset "mode (l=$l, m=$m)" begin
            @test norm(∂ζ_μ .+ ∂ζ̄_μ .- ∂x_fd) / norm(∂x_fd) < 1e-4
            @test norm(im .* (∂ζ_μ .- ∂ζ̄_μ) .- ∂y_fd) / norm(∂y_fd) < 1e-4
        end
    end
end

@testset "Aggregate radial coefficient-space operator" begin
    mode_cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))

    for (l, m) in mode_cases
        u         = 3 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        uhat      = psh(u, D)
        reference = assembled_r_dot_grad_reference(uhat)
        computed  = r_dot_∇Δ⁻¹(uhat)
        @test norm(computed - reference) < 1e-10
    end

    u         = exp.(-X .* Y) .* cos.(X .^ 2) .+ im .* exp.(-X .* Y) .* sin.(Y .^ 2)
    uhat      = psh(u, D)
    reference = assembled_r_dot_grad_reference(uhat)
    computed  = r_dot_∇Δ⁻¹(uhat)
    @test norm(computed - reference) < 1e-10
end
