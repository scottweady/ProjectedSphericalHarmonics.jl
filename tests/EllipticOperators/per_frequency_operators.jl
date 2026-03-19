using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

# ── Shared setup ──────────────────────────────────────────────────────────────

const MR      = 32
const MTHETA  = 32
const D       = disk(MR, MTHETA)
const ZETA    = D.ζ
const X       = real.(ZETA)
const Y       = imag.(ZETA)
const lmax    = MR
const LMAX_IP = 31   # used by in-place tests

# ── Helpers ───────────────────────────────────────────────────────────────────

function mode_column_index(m)
    return m >= 0 ? m + 1 : size(D.Mspan, 2) - abs(m) + 1
end

function dense_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:end, mode_column_index(m)])
end

function sparse_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:2:end, mode_column_index(m)])
end

function derivative_reference_coefficients(uhat, m, which::Symbol)
    target_m = which === :∂ζ ? m - 1 : m + 1
    col = mode_column_index(target_m)
    return copy(uhat[D.even[:, col], col])
end

function second_derivative_reference_coefficients(uhat, m, which::Symbol)
    target_m = which === :∂ζ̄∂ζ̄ ? m + 2 : which === :∂ζ∂ζ ? m - 2 : m
    col = mode_column_index(target_m)
    return copy(uhat[D.even[:, col], col])
end

function same_mode_reference_coefficients(uhat, m)
    col = mode_column_index(m)
    return copy(uhat[D.even[:, col], col])
end

function embed_dense_mode!(dest, coeffs, m; trim_aliasing=true)
    used = trim_aliasing ? length(coeffs) - 2 : length(coeffs)
    if m >= 0
        dest[m + 1:m + used, m + 1] .= coeffs[1:used]
    else
        col = mode_column_index(m)
        dest[abs(m) + 1:abs(m) + used, col] .= conj.(coeffs[1:used])
    end
    return dest
end

function embed_sparse_mode!(dest, coeffs, m; trim_aliasing=true)
    used = trim_aliasing ? length(coeffs) - 1 : length(coeffs)
    rows = abs(m) + 1:2:abs(m) + 2 * (used - 1) + 1
    col = mode_column_index(m)
    values = m >= 0 ? coeffs[1:used] : conj.(coeffs[1:used])
    dest[rows, col] .= values
    return dest
end

function mode_test_function(l, m)
    return 2 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 4 .* ylm.(l + 2, m, ZETA)
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "Coefficient-space inverse Laplacian by mode" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))

    for (l, m) in cases
        u = mode_test_function(l, m)
        u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
        uhat = psh(u, D)

        @testset "mode (l=$l, m=$m)" begin
            dense_mode    = dense_mode_coefficients(uhat, m)
            dense_inverse = Inverse_laplacian_coef_m(dense_mode, lmax, m)
            dense_coeffs  = zeros(ComplexF64, size(uhat))
            embed_dense_mode!(dense_coeffs, dense_inverse, m)
            @test norm(ipsh(dense_coeffs, D) - u_inv_reference) < 1e-12

            sparse_mode    = sparse_mode_coefficients(uhat, m)
            sparse_inverse = Ĝᵐ(sparse_mode, lmax, m)
            sparse_coeffs  = zeros(ComplexF64, size(uhat))
            embed_sparse_mode!(sparse_coeffs, sparse_inverse, m)
            @test norm(ipsh(sparse_coeffs, D) - u_inv_reference) < 1e-12
        end
    end
end

@testset "Sparse matrix inverse Laplacian representation" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (11, 0))

    for (l, m) in cases
        uhat        = rand(size(D.ζ)...)
        sparse_mode = sparse_mode_coefficients(uhat, m)

        matrix          = inverse_laplacian_matrix_sparse(lmax, m)
        inverse_sparse  = Ĝᵐ(sparse_mode, lmax, m; aliasing=false)
        @test norm(inverse_sparse - matrix * sparse_mode) < 1e-14

        rectangular     = inverse_laplacian_matrix_sparse(lmax, m; rectangular=true)
        aliased_inverse = Ĝᵐ(sparse_mode, lmax, m; aliasing=true)
        @test norm(aliased_inverse - rectangular * sparse_mode) < 1e-14
    end
end

@testset "Sparse matrix zeta-dz inverse Laplacian representation" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (11, 0))

    for (l, m) in cases
        uhat        = rand(ComplexF64, size(D.ζ)...)
        sparse_mode = sparse_mode_coefficients(uhat, m)

        matrix           = ζ∂ζΔ⁻¹_matrix_sparse(lmax, m)
        operator_values  = ζ_∂Ĝᵐ∂ζ(sparse_mode, lmax, m; aliasing=false)
        @test norm(operator_values - matrix * sparse_mode) < 1e-14

        rectangular              = ζ∂ζΔ⁻¹_matrix_sparse(lmax, m; rectangular=true)
        operator_values_aliased  = ζ_∂Ĝᵐ∂ζ(sparse_mode, lmax, m; aliasing=true)
        @test norm(operator_values_aliased - rectangular * sparse_mode) < 1e-14
    end
end

@testset "Derivative coefficient-space operators" begin
    mode_cases = ((8, 2), (9, 3), (10, 2), (10, -2), (11, -3), (10, 0))

    for (l, m) in mode_cases
        μ           = ylm.(abs(m), m, ZETA) .+ 10 .* ylm.(l + 2, m, ZETA) .+ 2 .* ylm.(l + 4, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ∂ζ_reference  = derivative_reference_coefficients(psh(∂ζ(Δinv_μ, D), D), m, :∂ζ)
        ∂ζ_sparse     = ∂Ĝᵐ∂ζ(μhat_sparse, lmax, m; aliasing=false)
        ∂ζ̄_reference = derivative_reference_coefficients(psh(∂ζ̄(Δinv_μ, D), D), m, :∂ζ̄)
        ∂ζ̄_sparse    = ∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)

        @test norm(∂ζ_sparse  - ∂ζ_reference)  < 1e-10
        @test norm(∂ζ̄_sparse - ∂ζ̄_reference) < 1e-10
    end
end

@testset "Double derivative coefficient-space operators" begin
    mode_cases = ((8, 2), (9, 3), (10, 2), (10, -2), (11, -3), (10, 0))

    for (l, m) in mode_cases
        μ           = 4 .* ylm.(l + 4, m, ZETA) .+ ylm.(abs(m), m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ζ∂ζ_reference  = same_mode_reference_coefficients(psh(ZETA .* ∂ζ(Δinv_μ, D), D), m)
        ζ∂ζ_sparse     = ζ_∂Ĝᵐ∂ζ(μhat_sparse, lmax, m; aliasing=false)
        @test norm(ζ∂ζ_sparse - ζ∂ζ_reference) < 1e-10

        ζ̄∂ζ̄_reference = same_mode_reference_coefficients(psh(conj.(ZETA) .* ∂ζ̄(Δinv_μ, D), D), m)
        ζ̄∂ζ̄_sparse    = ζ̄_∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)
        @test norm(ζ̄∂ζ̄_sparse - ζ̄∂ζ̄_reference) < 1e-10
    end
end

@testset "Second-derivative coefficient-space operators" begin
    mode_cases = ((6, 0), (5, 1), (6, 2), (7, 3), (10, 0))

    for (l, m) in mode_cases
        μ           = 3 .* ylm.(m, m, ZETA) .+ 2 .* ylm.(l, m, ZETA) .+ ylm.(l + 2, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ∂ζ̄∂ζ̄_reference = second_derivative_reference_coefficients(psh(∂ζ̄(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ̄∂ζ̄)
        ∂ζ̄∂ζ̄_sparse    = ∂²Ĝᵐ∂ζ̄²(μhat_sparse, lmax, m; aliasing=false)
        @test norm(∂ζ̄∂ζ̄_sparse - ∂ζ̄∂ζ̄_reference) < 1e-10

        ∂ζ∂ζ_reference = second_derivative_reference_coefficients(psh(∂ζ(∂ζ(Δinv_μ, D), D), D), m, :∂ζ∂ζ)
        ∂ζ∂ζ_sparse    = ∂²Ĝᵐ∂ζ²(μhat_sparse, lmax, m; aliasing=false)
        @test norm(∂ζ∂ζ_sparse - ∂ζ∂ζ_reference) < 1e-10

        ∂ζ∂ζ̄_reference = second_derivative_reference_coefficients(psh(∂ζ(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ∂ζ̄)
        ∂ζ∂ζ̄_sparse    = ∂²Ĝᵐ∂ζ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)
        @test norm(∂ζ∂ζ̄_sparse - ∂ζ∂ζ̄_reference) < 1e-10
    end
end

@testset "Edge-case frequency boundary mappings — single derivatives" begin
    edge_cases = (
        (8,  0, :∂ζ),
        (8,  0, :∂ζ̄),
        (7,  1, :∂ζ),
        (7, -1, :∂ζ̄),
    )

    for (l, m_in, op) in edge_cases
        m_out = op === :∂ζ ? m_in - 1 : m_in + 1
        μ           = ylm.(abs(m_in), m_in, ZETA) .+ 2 .* ylm.(l + 2, m_in, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m_in)
        reference   = derivative_reference_coefficients(
                          psh(op === :∂ζ ? ∂ζ(Δinv_μ, D) : ∂ζ̄(Δinv_μ, D), D), m_in, op)
        sparse_result = op === :∂ζ ?
            ∂Ĝᵐ∂ζ(μhat_sparse, lmax, m_in; aliasing=false) :
            ∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m_in; aliasing=false)

        @testset "$(op) m=$m_in → m=$m_out" begin
            @test norm(sparse_result - reference) < 1e-10
        end
    end
end

@testset "Edge-case frequency boundary mappings — double derivatives" begin
    cases_∂ζ∂ζ = ((7, 1), (6, 0))
    cases_∂ζ̄∂ζ̄ = ((7, -1), (6, 0))

    for (l, m) in cases_∂ζ∂ζ
        μ           = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)
        reference   = second_derivative_reference_coefficients(psh(∂ζ(∂ζ(Δinv_μ, D), D), D), m, :∂ζ∂ζ)
        sparse_result = ∂²Ĝᵐ∂ζ²(μhat_sparse, lmax, m; aliasing=false)

        @testset "∂ζ∂ζ m=$m → m=$(m-2)" begin
            @test norm(sparse_result - reference) < 1e-10
        end
    end

    for (l, m) in cases_∂ζ̄∂ζ̄
        μ           = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)
        reference   = second_derivative_reference_coefficients(psh(∂ζ̄(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ̄∂ζ̄)
        sparse_result = ∂²Ĝᵐ∂ζ̄²(μhat_sparse, lmax, m; aliasing=false)

        @testset "∂ζ̄∂ζ̄ m=$m → m=$(m+2)" begin
            @test norm(sparse_result - reference) < 1e-10
        end
    end
end

@testset "Modified Poisson system singular values" begin
    bessel_zeros = (2.40482555769577, 11.791534439014281)

    for α in bessel_zeros
        singular_values = svd(helmholtz_matrix(lmax, 0, α^2)).S
        @test singular_values[end] < 1e-14
        @test singular_values[end - 1] > 1e-14
    end
end

# ── In-place per-frequency sparse operators ───────────────────────────────────

@testset "In-place per-frequency sparse operators" begin
    mode_cases = (0, 1, 2, 3, -1, -2, -3)

    for m in mode_cases
        f̂ᵐ = randn(ComplexF64, size_current_m(LMAX_IP, m))

        res = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m))
        Ĝᵐ!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - Ĝᵐ(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m; aliasing=true))
        Ĝᵐ!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - Ĝᵐ(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_IP, m))
        ∂Ĝᵐ∂ζ!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, ∂ζ_indexing_sparse(LMAX_IP, m; aliasing=true))
        ∂Ĝᵐ∂ζ!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_IP, m))
        ∂Ĝᵐ∂ζ̄!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, ∂ζ̄_indexing_sparse(LMAX_IP, m; aliasing=true))
        ∂Ĝᵐ∂ζ̄!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m))
        ζ_∂Ĝᵐ∂ζ!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ζ_∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m; aliasing=true))
        ζ_∂Ĝᵐ∂ζ!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ζ_∂Ĝᵐ∂ζ(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m))
        ζ̄_∂Ĝᵐ∂ζ̄!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ζ̄_∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m; aliasing=true))
        ζ̄_∂Ĝᵐ∂ζ̄!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ζ̄_∂Ĝᵐ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_IP, m))
        ∂²Ĝᵐ∂ζ̄²!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ∂²Ĝᵐ∂ζ̄²(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, ∂ζ̄∂ζ̄_indexing_sparse(LMAX_IP, m; aliasing=true))
        ∂²Ĝᵐ∂ζ̄²!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ∂²Ĝᵐ∂ζ̄²(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_IP, m))
        ∂²Ĝᵐ∂ζ²!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ∂²Ĝᵐ∂ζ²(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, ∂ζ∂ζ_indexing_sparse(LMAX_IP, m; aliasing=true))
        ∂²Ĝᵐ∂ζ²!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ∂²Ĝᵐ∂ζ²(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13

        res = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m))
        ∂²Ĝᵐ∂ζ∂ζ̄!(res, f̂ᵐ, LMAX_IP, m)
        @test norm(res - ∂²Ĝᵐ∂ζ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=false)) < 1e-13

        res_alias = fill(1.0 + 1.0im, size_current_m(LMAX_IP, m; aliasing=true))
        ∂²Ĝᵐ∂ζ∂ζ̄!(res_alias, f̂ᵐ, LMAX_IP, m)
        @test norm(res_alias - ∂²Ĝᵐ∂ζ∂ζ̄(f̂ᵐ, LMAX_IP, m; aliasing=true)) < 1e-13
    end
end
