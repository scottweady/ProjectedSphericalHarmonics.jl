
using ProjectedSphericalHarmonics

# Discretize disk
M = 32
D = psh_disk(M)

# Get grid points
ζ = D.ζ
r = abs.(ζ)

# Modes to test
mspan = collect(0 : 5)

for (nm, m) in enumerate(mspan)

    # Argument
    u = ζ.^m

    # Numerical solution
    v_num = 𝒱(u, D)

    # Exact solution
    aₘ = 1 / (4 * (m + 1))
    bₘ = m > 0 ? -1 / (4 * m) : -1 / 4
    v_exact = (aₘ .* abs2.(ζ) .+ bₘ) .* ζ.^m

    # Compute error
    δv = (v_num .- v_exact) ./ ζ.^m
    err = maximum(abs.(δv))
    println("(m, error) = ", "($m, $err)")

end
