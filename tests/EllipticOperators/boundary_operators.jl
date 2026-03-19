using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

# ── Shared setup ──────────────────────────────────────────────────────────────

const MR     = 32
const MTHETA = 32
const D      = disk(MR, MTHETA)
const ZETA   = D.ζ
const lmax   = MR

# ── Helpers ───────────────────────────────────────────────────────────────────

function mode_column_index(m)
    return m >= 0 ? m + 1 : size(D.Mspan, 2) - abs(m) + 1
end

function sparse_mode_coefficients(uhat, m)
    return copy(@view uhat[abs(m) + 1:2:end, mode_column_index(m)])
end

function embed_sparse_mode!(dest, coeffs, m; trim_aliasing=true)
    used   = trim_aliasing ? length(coeffs) - 1 : length(coeffs)
    rows   = abs(m) + 1:2:abs(m) + 2 * (used - 1) + 1
    col    = mode_column_index(m)
    values = m >= 0 ? coeffs[1:used] : conj.(coeffs[1:used])
    dest[rows, col] .= values
    return dest
end

# ── Tests ─────────────────────────────────────────────────────────────────────

@testset "neumann_traceĜ" begin
    mode_cases = ((8, 2), (9, 3), (10, -2), (11, -3), (10, 0))

    for (l, m) in mode_cases
        f           = 2 .* ylm.(abs(m), m, ZETA) .+ ylm.(l, m, ZETA) .+ 3 .* ylm.(l + 2, m, ZETA)
        fhat        = psh(f, D)
        mode_coeffs = sparse_mode_coefficients(fhat, m)

        inverse_mode   = Ĝᵐ(mode_coeffs, lmax, m; aliasing=false)
        inverse_coeffs = zeros(ComplexF64, size(fhat))
        embed_sparse_mode!(inverse_coeffs, inverse_mode, m)
        boundary_values = vec(∂n(ipsh(inverse_coeffs, D), D))
        boundary_coeffs = fft(boundary_values) / length(boundary_values)

        @test abs(neumann_traceĜ(mode_coeffs, lmax, m) - boundary_coeffs[mode_column_index(m)]) < 1e-10
    end

    f    = 1 .+ ylm.(4, 0, ZETA) .+ ylm.(6, 2, ZETA) .+ ylm.(7, -3, ZETA)
    fhat = psh(f, D)
    inverse_coeffs = zeros(ComplexF64, size(fhat))
    inverse_coeffs[1:2:end, 1] .= Ĝᵐ(@view(fhat[1:2:end, 1]), lmax, 0; aliasing=false)
    embed_sparse_mode!(inverse_coeffs, Ĝᵐ(@view(fhat[3:2:end, 3]),     lmax,  2; aliasing=false),  2)
    embed_sparse_mode!(inverse_coeffs, Ĝᵐ(@view(fhat[4:2:end, end-2]), lmax, -3; aliasing=false), -3)

    boundary_values = vec(∂n(ipsh(inverse_coeffs, D), D))
    boundary_coeffs = fft(boundary_values) / length(boundary_values)

    expected = zeros(ComplexF64, size(fhat, 2))
    expected[1]     = boundary_coeffs[1]
    expected[3]     = boundary_coeffs[3]
    expected[end-2] = boundary_coeffs[end-2]

    @test norm(neumann_traceĜ(fhat) - expected) < 1e-10
end
