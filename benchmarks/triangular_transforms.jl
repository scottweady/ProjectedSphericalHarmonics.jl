# benchmarks/triangular_transforms.jl
#
# Compares the triangular-array transforms (psh! / ipsh!) against the
# classical path (psh → TriangularArrayToPSH / NodalToTriangularArray → ipsh)
# for increasing MR.  Run with:
#
#   julia --project benchmarks/triangular_transforms.jl

using BenchmarkTools
using LinearAlgebra
using Printf
using ProjectedSphericalHarmonics

# ─── helpers ──────────────────────────────────────────────────────────────────

function _test_function_even(D)
    return exp.(-abs2.((D.ζ .- (0.3 - 0.3im)) / 0.1))
end

function _test_function_odd(D)
    return exp.(-abs2.((D.ζ .- (0.3 - 0.3im)) / 0.1)) .* D.w
end

# ─── forward transform benchmarks ─────────────────────────────────────────────

function bench_forward(MR; parity = :even)
    D     = disk(MR)
    u     = parity == :even ? _test_function_even(D) : _test_function_odd(D)
    Mspan = vec(Array(D.Mspan))

    # triangular path: psh! in-place
    # û_tri = TriangularCoeffArray{Float64}(D.Mr, Mspan; parity = parity, ordering = :fft)
    # b_tri = @benchmark psh!($û_tri, $u, $D) samples=200 evals=3 seconds=10

    # triangular path: psh! out of place
    b_tri = @benchmark psh_triangular($u, $D; parity = $parity) samples=200 evals=3 seconds=10


    # classical path: psh then extract triangular
    b_cls = @benchmark psh($u, $D; parity = $parity) samples=200 evals=3 seconds=10

    return b_tri, b_cls
end

# ─── inverse transform benchmarks ─────────────────────────────────────────────

function bench_inverse(MR; parity = :even)
    D     = disk(MR)
    u_raw = parity == :even ? _test_function_even(D) : _test_function_odd(D)
    û_tri = NodalToTriangularArray(u_raw, D; parity = parity)
    û_psh = TriangularArrayToPSH(û_tri, D)
    u_out = zeros(ComplexF64, size(D.ζ))

    # triangular path: ipsh! in-place
    b_tri = @benchmark ipsh!($u_out, $û_tri, $D) samples=200 evals=3 seconds=10

    # classical path: ipsh on full square array
    b_cls = @benchmark ipsh($û_psh, $D; parity = $parity) samples=200 evals=3 seconds=10

    return b_tri, b_cls
end

# ─── printing ─────────────────────────────────────────────────────────────────

function fmt_time(t_ns)
    t_us = t_ns / 1e3
    t_ms = t_ns / 1e6
    t_ms >= 1.0 && return @sprintf("%.3f ms", t_ms)
    return @sprintf("%.3f μs", t_us)
end

function print_row(MR, b_tri, b_cls)
    t_tri  = median(b_tri).time
    t_cls  = median(b_cls).time
    a_tri  = median(b_tri).allocs
    a_cls  = median(b_cls).allocs
    m_tri  = median(b_tri).memory / 1024
    m_cls  = median(b_cls).memory / 1024
    ratio  = t_cls / t_tri
    @printf("  MR = %4d  |  tri: %10s  %5d allocs  %7.1f KiB  |  cls: %10s  %5d allocs  %7.1f KiB  |  speedup: %.2fx\n",
            MR,
            fmt_time(t_tri), a_tri, m_tri,
            fmt_time(t_cls), a_cls, m_cls,
            ratio)
end

# ─── main ─────────────────────────────────────────────────────────────────────

MR_values = [16, 32, 64, 128, 256, 512]

for parity in (:even, :odd)
    println()
    println("=" ^ 110)
    println("  FORWARD TRANSFORM (psh!)  —  parity = :$parity")
    println("=" ^ 110)
    println("  triangular path = psh_triangular(u, D; parity)")
    println("  classical path  = psh(u, D; parity)")
    println("-" ^ 110)
    # for MR in MR_values
    #     b_tri, b_cls = bench_forward(MR; parity = parity)
    #     print_row(MR, b_tri, b_cls)
    # end

    println()
    println("=" ^ 110)
    println("  INVERSE TRANSFORM (ipsh!)  —  parity = :$parity")
    println("=" ^ 110)
    println("  triangular path = ipsh!(u_out, û_tri, D)")
    println("  classical path  = ipsh(û_psh, D; parity)")
    println("-" ^ 110)
    for MR in MR_values
        b_tri, b_cls = bench_inverse(MR; parity = parity)
        print_row(MR, b_tri, b_cls)
    end
end

println()
