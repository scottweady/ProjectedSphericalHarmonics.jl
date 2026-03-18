using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

const MR = 32
const MTHETA = 32
const D = disk(MR, MTHETA)
const ZETA = D.ζ
const X = real.(ZETA)
const Y = imag.(ZETA)
const lmax = MR 

function mode_column_index(m)
    return m >= 0 ? m + 1 : size(D.Mspan, 2) - abs(m) + 1
end

function dense_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:end, mode_column_index(m)])
end

function sparse_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:2:end, mode_column_index(m)])
end

function sparse_output_mode_coefficients(uhat, m)
    return sparse_mode_coefficients(uhat, m)
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

function sparse_same_mode_projection(uhat)
    projected = zeros(ComplexF64, size(uhat))
    max_mode = min(div(size(uhat, 2) - 1, 2), size(uhat, 1) - 2)

    projected[1:2:end, 1] .= same_mode_reference_coefficients(uhat, 0)

    for m in 1:max_mode
        projected[m+1:2:end, m+1] .= same_mode_reference_coefficients(uhat, m)
        projected[m+1:2:end, end-(m-1)] .= same_mode_reference_coefficients(uhat, -m)
    end

    return projected
end

function assembled_r_dot_grad_reference(uhat)
    reference = zeros(ComplexF64, size(uhat))
    max_mode = min(div(size(uhat, 2) - 1, 2), size(uhat, 1) - 2)

    mode0 = @view uhat[1:2:end, 1]
    reference[1:2:end, 1] .= ζ_∂Ĝᵐ∂ζ(mode0, lmax, 0; aliasing=false) .+
                             ζ̄_∂Ĝᵐ∂ζ̄(mode0, lmax, 0; aliasing=false)

    for m in 1:max_mode
        mode_pos = @view uhat[m+1:2:end, m+1]
        reference[m+1:2:end, m+1] .= ζ_∂Ĝᵐ∂ζ(mode_pos, lmax, m; aliasing=false) .+
                                     ζ̄_∂Ĝᵐ∂ζ̄(mode_pos, lmax, m; aliasing=false)

        mode_neg = @view uhat[m+1:2:end, end-(m-1)]
        reference[m+1:2:end, end-(m-1)] .= ζ_∂Ĝᵐ∂ζ(mode_neg, lmax, -m; aliasing=false) .+
                                           ζ̄_∂Ĝᵐ∂ζ̄(mode_neg, lmax, -m; aliasing=false)
    end

    return reference
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

@testset "Coefficient-space inverse Laplacian by mode" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))

    for (l, m) in cases
        u = mode_test_function(l, m)
        u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
        uhat = psh(u, D)

        @testset "mode (l=$l, m=$m)" begin
            dense_mode = dense_mode_coefficients(uhat, m)
            dense_inverse = Inverse_laplacian_coef_m(dense_mode, lmax, m)
            dense_coeffs = zeros(ComplexF64, size(uhat))
            embed_dense_mode!(dense_coeffs, dense_inverse, m)
            @test norm(ipsh(dense_coeffs, D) - u_inv_reference) < 1e-12

            sparse_mode = sparse_mode_coefficients(uhat, m)
            sparse_inverse = Ĝᵐ(sparse_mode, lmax, m)
            sparse_coeffs = zeros(ComplexF64, size(uhat))
            embed_sparse_mode!(sparse_coeffs, sparse_inverse, m)
            @test norm(ipsh(sparse_coeffs, D) - u_inv_reference) < 1e-12
        end
    end
end

@testset "Inverse_laplacian wrapper" begin
    
    mode_cases = ((8, 2), (10, -2), (2,0), (4,0), (10,0))

    for (l, m) in mode_cases
        u = mode_test_function(l, m)
        u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
        Inverse_laplacian(psh(u, D))
        ipsh(Inverse_laplacian(psh(u, D)), D)
        @test norm(ipsh(Inverse_laplacian(psh(u, D)), D) - u_inv_reference) < 1e-12
    end

    u = exp.(-X .* Y) .* cos.(X .^ 2) .+ im .* exp.(-X .* Y) .* sin.(Y .^ 2)
    u_inv_reference = 𝒮(𝒩⁻¹(u, D), D)
    @test norm(ipsh(Inverse_laplacian(psh(u, D)), D) - u_inv_reference) < 1e-12
end


@testset "Sparse matrix inverse Laplacian representation" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (11, 0))

    for (l, m) in cases
        # l = 10
        # m = -2
        println("Testing mode (l=$l, m=$m)")
        # u = mode_test_function(l, m)
        uhat = rand(size(D.ζ)...)#psh(u, D)
        sparse_mode = sparse_mode_coefficients(uhat, m)

        matrix = inverse_laplacian_matrix_sparse(lmax, m)
        inverse_sparse = Ĝᵐ(sparse_mode, lmax, m; aliasing=false)
        @test norm(inverse_sparse - matrix * sparse_mode) < 1e-14

        rectangular = inverse_laplacian_matrix_sparse(lmax, m; rectangular=true)
        aliased_inverse = Ĝᵐ(sparse_mode, lmax, m; aliasing=true)
        @test norm(aliased_inverse - rectangular * sparse_mode) < 1e-14
    end
end

@testset "Sparse matrix zeta-dz inverse Laplacian representation" begin
    cases = ((8, 2), (9, 3), (10, -2), (11, -3), (11, 0))

    for (l, m) in cases
        println("Testing zeta-dz mode (l=$l, m=$m)")
        uhat = rand(ComplexF64, size(D.ζ)...)
        sparse_mode = sparse_mode_coefficients(uhat, m)

        matrix = ζ∂ζΔ⁻¹_matrix_sparse(lmax, m)
        operator_values = ζ_∂Ĝᵐ∂ζ(sparse_mode, lmax, m; aliasing=false)
        @test norm(operator_values - matrix * sparse_mode) < 1e-14

        rectangular = ζ∂ζΔ⁻¹_matrix_sparse(lmax, m; rectangular=true)
        operator_values_aliased = ζ_∂Ĝᵐ∂ζ(sparse_mode, lmax, m; aliasing=true)
        @test norm(operator_values_aliased - rectangular * sparse_mode) < 1e-14
    end
end


@testset "Derivative coefficient-space operators" begin
    mode_cases = ((8, 2), (9, 3), (10,2) ,(10, -2), (11, -3), (10, 0))


    for (l, m) in mode_cases
        println("Testing mode (l=$l, m=$m)")
        μ = ylm.(abs(m), m, ZETA) .+ 10 .* ylm.(l + 2, m, ZETA) .+ 2 .* ylm.(l + 4, m, ZETA)
        μhat = psh(μ, D)
        Δinv_μ = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ∂ζ_reference = derivative_reference_coefficients(psh(∂ζ(Δinv_μ, D), D), m, :∂ζ)
        ∂ζ_sparse = ∂Ĝᵐ∂ζ(μhat_sparse, lmax, m; aliasing=false)

        ∂ζ̄_reference = derivative_reference_coefficients(psh(∂ζ̄(Δinv_μ, D), D), m, :∂ζ̄)
        ∂ζ̄_sparse = ∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)
        

        @test norm(∂ζ_sparse - ∂ζ_reference) < 1e-10
        @test norm(∂ζ̄_sparse - ∂ζ̄_reference) < 1e-10
    end
end

@testset "Grid-space ∂ζ and ∂ζ̄ operators" begin
    mode_cases = ((8, 2), (9, 3), (10, 2), (10, -2), (11, -3), (10, 0))
    h = 1e-6
    ε = 10h
    interior = abs.(ZETA) .< 1 - ε

    for (l, m) in mode_cases
        μ = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA) .+ 2 .* ylm.(l + 4, m, ZETA)

        ∂ζ_μ  = ∂ζ(μ, D)[interior]
        ∂ζ̄_μ = ∂ζ̄(μ, D)[interior]
        ζ_int = ZETA[interior]
        x_int = real.(ζ_int)
        y_int = imag.(ζ_int)

        # ∂x = ∂ζ + ∂ζ̄
        ζ_x⁺ = (x_int .+ h) .+ im .* y_int
        ζ_x⁻ = (x_int .- h) .+ im .* y_int
        μ_x⁺ = ylm.(abs(m), m, ζ_x⁺) .+ 2 .* ylm.(l + 2, m, ζ_x⁺) .+ 2 .* ylm.(l + 4, m, ζ_x⁺)
        μ_x⁻ = ylm.(abs(m), m, ζ_x⁻) .+ 2 .* ylm.(l + 2, m, ζ_x⁻) .+ 2 .* ylm.(l + 4, m, ζ_x⁻)
        ∂x_fd = (μ_x⁺ .- μ_x⁻) ./ (2h)

        # ∂y = i*(∂ζ - ∂ζ̄)
        ζ_y⁺ = x_int .+ im .* (y_int .+ h)
        ζ_y⁻ = x_int .+ im .* (y_int .- h)
        μ_y⁺ = ylm.(abs(m), m, ζ_y⁺) .+ 2 .* ylm.(l + 2, m, ζ_y⁺) .+ 2 .* ylm.(l + 4, m, ζ_y⁺)
        μ_y⁻ = ylm.(abs(m), m, ζ_y⁻) .+ 2 .* ylm.(l + 2, m, ζ_y⁻) .+ 2 .* ylm.(l + 4, m, ζ_y⁻)
        ∂y_fd = (μ_y⁺ .- μ_y⁻) ./ (2h)

        @testset "mode (l=$l, m=$m)" begin
            @test norm(∂ζ_μ .+ ∂ζ̄_μ .- ∂x_fd) / norm(∂x_fd) < 1e-4
            @test norm(im .* (∂ζ_μ .- ∂ζ̄_μ) .- ∂y_fd) / norm(∂y_fd) < 1e-4
        end
    end
end

@testset "Double derivative coefficient-space operators" begin
    mode_cases = ((8, 2), (9, 3), (10,2) ,(10, -2), (11, -3), (10, 0))


    for (l, m) in mode_cases
        μ = 4 .* ylm.(l + 4, m, ZETA) .+ ylm.(abs(m), m, ZETA)
        μhat = psh(μ, D)
        Δinv_μ = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ζ∂ζ_reference = same_mode_reference_coefficients(psh(ZETA .* ∂ζ(Δinv_μ, D), D), m)
        ζ∂ζ_sparse = ζ_∂Ĝᵐ∂ζ(μhat_sparse, lmax, m; aliasing=false)
        @test norm(ζ∂ζ_sparse - ζ∂ζ_reference) < 1e-10

        ζ̄∂ζ̄_reference = same_mode_reference_coefficients(psh(conj.(ZETA) .* ∂ζ̄(Δinv_μ, D), D), m)
        ζ̄∂ζ̄_sparse = ζ̄_∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)
        @test norm(ζ̄∂ζ̄_sparse - ζ̄∂ζ̄_reference) < 1e-10
    end
end

@testset "Second-derivative coefficient-space operators" begin
    mode_cases = ((6, 0), (5, 1), (6, 2), (7, 3), (10, 0))

    for (l, m) in mode_cases
        μ = 3 .* ylm.(m, m, ZETA) .+ 2 .* ylm.(l, m, ZETA) .+ ylm.(l + 2, m, ZETA)
        μhat = psh(μ, D)
        Δinv_μ = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        ∂ζ̄∂ζ̄_reference = second_derivative_reference_coefficients(psh(∂ζ̄(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ̄∂ζ̄)
        ∂ζ̄∂ζ̄_sparse = ∂²Ĝᵐ∂ζ̄²(μhat_sparse, lmax, m; aliasing=false)

        @test norm(∂ζ̄∂ζ̄_sparse - ∂ζ̄∂ζ̄_reference) < 1e-10

        ∂ζ∂ζ_reference = second_derivative_reference_coefficients(psh(∂ζ(∂ζ(Δinv_μ, D), D), D), m, :∂ζ∂ζ)
        ∂ζ∂ζ_sparse = ∂²Ĝᵐ∂ζ²(μhat_sparse, lmax, m; aliasing=false)

        @test norm(∂ζ∂ζ_sparse - ∂ζ∂ζ_reference) < 1e-10

        ∂ζ∂ζ̄_reference = second_derivative_reference_coefficients(psh(∂ζ(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ∂ζ̄)
        ∂ζ∂ζ̄_sparse = ∂²Ĝᵐ∂ζ∂ζ̄(μhat_sparse, lmax, m; aliasing=false)

        @test norm(∂ζ∂ζ̄_sparse - ∂ζ∂ζ̄_reference) < 1e-10
    end
end

@testset "Aggregate radial coefficient-space operator" begin
    mode_cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))

    for (l, m) in mode_cases
        u = 3 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        uhat = psh(u, D)
        reference = assembled_r_dot_grad_reference(uhat)
        computed = r_dot_∇Δ⁻¹(uhat)

        @test norm(computed - reference) < 1e-10
    end

    u = exp.(-X .* Y) .* cos.(X .^ 2) .+ im .* exp.(-X .* Y) .* sin.(Y .^ 2)
    uhat = psh(u, D)
    reference = assembled_r_dot_grad_reference(uhat)
    computed = r_dot_∇Δ⁻¹(uhat)

    @test norm(computed - reference) < 1e-10
end

@testset "neumann_traceĜ" begin
    mode_cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))
    # l,m = 8,2

    for (l, m) in mode_cases
        f = 2 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 3 .* ylm.(l + 2, m, ZETA)
        fhat = psh(f, D)
        mode_coeffs = sparse_mode_coefficients(fhat, m)

        inverse_mode = Ĝᵐ(mode_coeffs, lmax, m; aliasing=false)
        inverse_coeffs = zeros(ComplexF64, size(fhat))
        embed_sparse_mode!(inverse_coeffs, inverse_mode, m)
        boundary_values = vec(∂n(ipsh(inverse_coeffs, D), D))
        boundary_coeffs = fft(boundary_values) / length(boundary_values)


        @test abs(neumann_traceĜ(mode_coeffs, lmax, m) - boundary_coeffs[mode_column_index(m)]) < 1e-10
    end

    f = 1 .+ ylm.(4, 0, ZETA) .+ ylm.(6, 2, ZETA) .+ ylm.(7, -3, ZETA)
    fhat = psh(f, D)
    inverse_coeffs = zeros(ComplexF64, size(fhat))
    inverse_coeffs[1:2:end, 1] .= Ĝᵐ(@view(fhat[1:2:end, 1]), lmax, 0; aliasing=false)
    embed_sparse_mode!(inverse_coeffs, Ĝᵐ(@view(fhat[3:2:end, 3]), lmax, 2; aliasing=false), 2)
    embed_sparse_mode!(inverse_coeffs, Ĝᵐ(@view(fhat[4:2:end, end-2]), lmax, -3; aliasing=false), -3)

    boundary_values = vec(∂n(ipsh(inverse_coeffs, D), D))
    boundary_coeffs = fft(boundary_values) / length(boundary_values)

    expected = zeros(ComplexF64, size(fhat, 2))
    expected[1] = boundary_coeffs[1]
    expected[3] = boundary_coeffs[3]
    expected[end-2] = boundary_coeffs[end-2]

    @test norm(neumann_traceĜ(fhat) - expected) < 1e-10
end






@testset "Edge-case frequency boundary mappings — single derivatives" begin
    # These test the specific boundary crossings where m transitions across zero
    # or from positive to negative, which have different index arithmetic.

    edge_cases = (
        # (l, m_input, m_output, op)
        (8, 0, -1, :∂ζ),    # ∂ζ: m=0 → m=-1 (crosses into negative frequencies)
        (8, 0,  1, :∂ζ̄),   # ∂ζ̄: m=0 → m=+1
        (7, 1,  0, :∂ζ),    # ∂ζ: m=1 → m=0
        (7, -1, 0, :∂ζ̄),   # ∂ζ̄: m=-1 → m=0 (crosses from negative into zero)
        (7, 1, 0, :∂ζ),  # ∂ζ: m=1 → m=0 (redundant but explicit)
    )

    for (l, m_in, _, op) in edge_cases
        m_out = op === :∂ζ ? m_in - 1 : m_in + 1
        μ = ylm.(abs(m_in), m_in, ZETA) .+ 2 .* ylm.(l + 2, m_in, ZETA)
        μhat       = psh(μ, D)
        Δinv_μ     = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m_in)

        reference = derivative_reference_coefficients(psh(op === :∂ζ ? ∂ζ(Δinv_μ, D) : ∂ζ̄(Δinv_μ, D), D), m_in, op)

        sparse_result = op === :∂ζ ?
            ∂Ĝᵐ∂ζ(μhat_sparse, lmax, m_in; aliasing=false) :
            ∂Ĝᵐ∂ζ̄(μhat_sparse, lmax, m_in; aliasing=false)

        @testset "$(op) m=$m_in → m=$m_out" begin
            @test norm(sparse_result - reference) < 1e-10
        end
    end
end

@testset "Edge-case frequency boundary mappings — double derivatives" begin
    # Test double derivatives that cross or straddle zero frequency.

    cases_∂ζ∂ζ = (
        (7, 1),   # ∂ζ∂ζ: m=1 → m=-1 (crosses zero twice)
        (6, 0),   # ∂ζ∂ζ: m=0 → m=-2
    )

    cases_∂ζ̄∂ζ̄ = (
        (7, -1),  # ∂ζ̄∂ζ̄: m=-1 → m=1 (crosses zero twice)
        (6, 0),   # ∂ζ̄∂ζ̄: m=0 → m=2
    )

    for (l, m) in cases_∂ζ∂ζ
        m_out = m - 2
        μ = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        reference = second_derivative_reference_coefficients(psh(∂ζ(∂ζ(Δinv_μ, D), D), D), m, :∂ζ∂ζ)
        sparse_result = ∂²Ĝᵐ∂ζ²(μhat_sparse, lmax, m; aliasing=false)

        @testset "∂ζ∂ζ m=$m → m=$m_out" begin
            @test norm(sparse_result - reference) < 1e-10
        end
    end

    for (l, m) in cases_∂ζ̄∂ζ̄
        m_out = m + 2
        μ = ylm.(abs(m), m, ZETA) .+ 2 .* ylm.(l + 2, m, ZETA)
        μhat        = psh(μ, D)
        Δinv_μ      = ipsh(Inverse_laplacian(μhat), D)
        μhat_sparse = sparse_mode_coefficients(μhat, m)

        reference = second_derivative_reference_coefficients(psh(∂ζ̄(∂ζ̄(Δinv_μ, D), D), D), m, :∂ζ̄∂ζ̄)
        sparse_result = ∂²Ĝᵐ∂ζ̄²(μhat_sparse, lmax, m; aliasing=false)

        @testset "∂ζ̄∂ζ̄ m=$m → m=$m_out" begin
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
