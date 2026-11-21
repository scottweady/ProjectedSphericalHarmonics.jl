using Random

println("Testing evaluation at arbitrary points...")

Random.seed!(3)

# Error relative to the size `scale` of the expansion (odd expansions vanish on the rim)
maxrel(A, B, scale = maximum(abs, B)) = maximum(abs, A - B) / scale

# Direct sum Σ û_ℓ^m y_ℓ^m(ζ) over the modes of the parity
function direct_sum(û, D, ζ, parity)
  mask = getproperty(D, parity)
  u = zeros(ComplexF64, length(ζ))
  for (nm, m) in enumerate(D.Mspan), l = abs(m) : D.Mℓ
    mask[l + 1, nm] && (u .+= û[l + 1, nm] .* ylm(l, m, ζ))
  end
  return u
end

# Mℓ ≠ Mₘ, and a domain (fields forwarded to its disk; points are in the reference disk, D.ζ)
for (label, D) in [("disk(12, 9)", disk(12, 9)), ("ellipse", ellipse(1.5, 1.0, 10))]
  u = exp.(D.ζ) .* (1 .+ abs2.(D.ζ)) .+ cos.(3 .* real.(D.ζ))
  ζ = sqrt.(rand(200)) .* cis.(2π .* rand(200))
  ζrim = cis.(2π .* rand(20))
  for parity in (:even, :odd)
    û = psh(u, D; parity=parity)
    û₀ = copy(û)
    scale = maximum(abs, ipsh(û, D; parity=parity))

    # On the grid points, against the FFT-based inverse transform
    check("  ipsh at grid points vs ipsh ($label, $parity): ",
          maxrel(ipsh(û, D, vec(D.ζ); parity=parity), vec(ipsh(û, D; parity=parity)), scale), 1e-12)

    # At random interior points and on the rim, against the direct sum
    check("  ipsh at interior points vs direct sum ($label, $parity): ",
          maxrel(ipsh(û, D, ζ; parity=parity), direct_sum(û, D, ζ, parity), scale), 1e-13)
    check("  ipsh at rim points vs direct sum ($label, $parity): ",
          maxrel(ipsh(û, D, ζrim; parity=parity), direct_sum(û, D, ζrim, parity), scale), 1e-13)

    # Points outside are clamped to the rim, where the odd modes vanish (and ~ √(1 - |ζ|²) near it,
    # so the rounding in |ζrim| ≈ 1 shows at the 1e-8 level in the odd values at ζrim)
    rimvals = parity == :even ? ipsh(û, D, ζrim) : zeros(length(ζrim))
    check("  clamping to the rim ($label, $parity): ",
          maxrel(ipsh(û, D, 1.5 .* ζrim; parity=parity), rimvals, scale), 1e-14)

    # The rim values match trace (radial evaluation) on the grid's angles
    if parity == :even
      θ = vec(D.θ[1, :])
      check("  ipsh on the rim vs radial ipsh ($label): ",
            maxrel(ipsh(û, D, cis.(θ)), vec(ipsh(û, D, [1.0]))), 1e-13)
    end

    # Matrix form against vector form, with columns in the order of the retained modes
    idx = findall(vec(getproperty(D, parity)))
    parity == :even && @test idx == D.idx_even
    A = ipsh_matrix(D, [ζ; ζrim]; parity=parity)
    @test size(A) == (220, length(idx))
    check("  ipsh_matrix vs ipsh ($label, $parity): ",
          maxrel(A * vec(û)[idx], ipsh(û, D, [ζ; ζrim]; parity=parity), scale), 1e-13)

    # Azimuthal orders of the columns: y_ℓ^m(ζ e^{-iα}) = y_ℓ^m(ζ) e^{-imα} (at interior points, as
    # the odd modes are not differentiable at the rim)
    ms = mode_orders(D; parity=parity)
    @test length(ms) == length(idx)
    @test maxrel(A[1 : 200, :] .* cis.(-ms' .* 0.7), ipsh_matrix(D, ζ .* cis(-0.7); parity=parity)) < 1e-12

    # û is not modified, by either point evaluation
    ipsh(û, D, [1.0]; parity=parity)
    @test û == û₀
  end
end

# Real coefficients and tuples
let D = disk(8)
  û = real.(psh(abs2.(D.z), D))
  ζ = [0.3 + 0.2im, -0.5im]
  @test ipsh(û, D, ζ) ≈ abs2.(ζ)
  @test all(ipsh((û, 2û), D, ζ) .≈ (abs2.(ζ), 2abs2.(ζ)))
end
