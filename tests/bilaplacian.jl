
using ProjectedSphericalHarmonics

# Discretize disk
M = 32
D = psh_disk(M)

# Get grid points
ζ = D.ζ
r = abs.(ζ)

# Modes to test
mspan = collect(0 : 5)

# Integration constant
κ = 0

for (nm, m) in enumerate(mspan)

    # Argument
    u = ζ.^m

    # Numerical solution
    v_num = ℬ(u, D, κ=κ)

    # Exact solution
    aₘ =  1 / (32 * (m + 1) * (m + 2))

    if m == 0
        bₘ, cₘ = -1/16 + κ/8, -5/64 + κ/16
    elseif m == 1
        bₘ, cₘ = -1/32, 3/64 - κ/16
    else
        bₘ, cₘ =  -1 / (16 * m * (m + 1)), 1 / (32 * m * (m - 1))
    end

    v_exact = (aₘ .* abs2.(ζ).^2 .+ bₘ .* abs2.(ζ) .+ cₘ) .* ζ.^m

    # Compute error
    δv = (v_num .- v_exact) ./ ζ.^m
    err = maximum(abs.(δv))
    println("(m, error) = ", "($m, $err)")

end
