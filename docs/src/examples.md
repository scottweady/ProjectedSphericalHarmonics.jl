# Examples

The scripts in `examples/` check the package against exact solutions. They have their own environment: from the
package root, run `julia --project=examples -e 'using Pkg; Pkg.instantiate()'` once, then
`julia --project=examples examples/<name>.jl`. Below, each script's key lines are followed by its output, from a
laptop. Each script takes 8–20 s, most of it compilation.

## Eigenfunctions of the single layer (`eigenfunctions.jl`)

Applies `𝒮` to ``y_5^3/w`` and `𝒮⁻¹` to ``y_5^3``, and compares with the eigenvalue relation
``\mathcal{S}[y_\ell^m/w] = (\lambda_\ell^m/4)\, y_\ell^m``:

```julia
D = disk(64, 16)
ζ, w = D.ζ, D.w

l, m = 5, 3
u = ylm(l, m, ζ)

err = 𝒮(u ./ w, D) - (λlm(l, m) * u / 4.0)
err = 𝒮⁻¹(u, D) - (4.0 / λlm(l, m)) * (u ./ w)
```

```
Max error in 𝒮 for (l,m) = (5,3): 4.4e-16
Max error in 𝒮⁻¹ for (l,m) = (5,3): 4.2e-9
```

The inverse's error comes from dividing by ``w`` near the rim, where ``w`` is small, which amplifies rounding.

## The Poisson problem (`poisson.jl`)

Solves ``\Delta u = -1`` in the disk with ``u = 0`` on the rim, against ``u = (1 - r^2)/4``:

```julia
D = disk(64, 16)
u = Δ⁻¹(-1, 0, D)                  # right-hand side f = -1, boundary data g = 0
err = maximum(abs.(u .- 0.25 * (1 .- abs2.(D.ζ))))
```

```
Max error in Δ⁻¹ for f = -1, g = 0: 6.0e-15
```

## Drag and torque on a disk (`stokes.jl`)

The force on a disk translating at unit speed and the torque on one rotating at unit rate, in its plane, under a
stress-free surface (the default viscosities). Both are ``16/3``. Then the same disk moving out of its plane in free
space: broadside, with drag ``16\mu``, and tilting about a diameter, with torque ``32\mu/3``. The normal force
density is ``f_z = 2\mu\, \mathcal{S}^{-1}[u_z]``.

```julia
D = disk(128, 64)
x, y = real.(D.ζ), imag.(D.ζ)

# in-plane, under a stress-free surface
f = 𝒮_st⁻¹((1.0, 0.0), D)
F = real(sum(f[1] .* D.dζ))                            # 16/3

f = 𝒮_st⁻¹((-y, x), D)
T = real(sum((-y .* f[1] + x .* f[2]) .* D.dζ))        # 16/3

# out of the plane, free space with viscosity μ
μ = 1.0
f = 𝒮_st⁻¹((0.0, 0.0, 1.0), D; μ=μ)
Fz = real(sum(f[3] .* D.dζ))                           # 16μ

f = 𝒮_st⁻¹((0.0, 0.0, y), D; μ=μ)
Tx = real(sum(y .* f[3] .* D.dζ))                      # 32μ/3
```

```
force on a translating disk: 5.333333333336814
torque on a rotating disk: 5.333333333338558
force on a disk moving broadside: 16.000000000010445
torque on a tilting disk: 10.666666666676285
```

## An ellipse, two ways (`ellipse.jl`)

Two quantities for an ellipse with ``a = 1.5``, ``b = 1``, at ``M = 32``: the charge density at unit potential,
and the drag when it translates along its major axis. The ellipse is discretized twice: by `ellipse` (the affine
map) and by `domain` from 127 samples of its boundary (the conformal map, experimental). The references are the
closed forms ``\sigma\sqrt{1 - x^2/a^2 - y^2/b^2} = 2/(b\,K(1 - b^2/a^2))`` and the drag ``4\pi/A``, with ``A``
from complete elliptic integrals:

```julia
θ = range(0, 2π, 128)[1:end-1]
γ = a * cos.(θ) + im * b * sin.(θ)

for Ω in (ellipse(a, b, M), domain(γ, M))
    σ = 𝒮⁻¹(1.0, Ω)                                     # charge density at unit potential
    x, y = real.(Ω.z), imag.(Ω.z)
    err = σ .* sqrt.(1 .- (x/a).^2 .- (y/b).^2) .- 2 / (b * ellipk(1 - (b/a)^2))

    f = 𝒮_st⁻¹((1.0, 0.0), Ω)                           # translating along the major axis
    F = integral(f[1], Ω)
end
```

```
Using angle discretization
Error in single layer potential: 8.8e-11
Error in drag force on an ellipse: 9.4e-14
Using conformal discretization
Error in single layer potential: 6.6e-5
Error in drag force on an ellipse: 6.1e-9
```

The affine map is exact, so `ellipse` is limited only by the expansion. The conformal map is computed numerically
from the boundary samples, which limits `domain`. The pointwise density, measured against the ``1/w`` form, is the
most sensitive to the map's accuracy near the boundary.

## A second-kind equation by GMRES (`concentration.jl`)

Solves ``\sigma + 2\beta\, \mathcal{S}[\sigma] = 2\beta`` in coefficient space with the package's `gmres` wrapper,
for ``\beta = 0.1, 1, 10, 100``. As ``\beta \to \infty`` the solution tends to that of ``\mathcal{S}[\sigma] = 1``,
``\sigma_0 = 4/(\pi w)``:

```julia
D = disk(128, 64)
σ₀ = 𝒮⁻¹(1, D)                                          # the β → ∞ limit

function L̂!(b̂, σ̂)                                       # σ + 2β 𝒮[σ], on PSH coefficients
    σ̂ = reshape(σ̂, size(D.ζ))
    σ̂w = psh(ipsh(σ̂, D) .* D.w, D)
    b̂ .= vec(σ̂ + 2β * D.K̂_S .* σ̂w)                      # 𝒮 is diagonal: λ/4 on the even modes
end

σ̂ = gmres(L̂!, psh(2β, D))
σ = ipsh(σ̂, D)
```

```
β = 0.1,  |σ(0) - σ₀(0)| = 1.09    (GMRES: 3 iterations)
β = 1,    |σ(0) - σ₀(0)| = 0.314   (5 iterations)
β = 10,   |σ(0) - σ₀(0)| = 0.0879  (8 iterations)
β = 100,  |σ(0) - σ₀(0)| = 0.0218  (17 iterations)
```

The difference falls roughly like ``1/\beta``, while the iteration count grows with ``\beta``, as the operator
approaches the first-kind ``\mathcal{S}``.

## Stability of a growing droplet (`microbial_droplet.jl`)

A linear stability calculation that combines most of the Laplace operators. A droplet grows from a uniform
concentration with buoyant forcing (``\mathrm{Ra} = \beta = 1``, ``\gamma = 0``). Its base state is built from:
- the volume potentials `𝒱` and `ℬ`;
- `𝒩⁻¹` for the pressure;
- `𝒮` for the stream function;
- `∂n` for the normal velocity of the rim.

The perturbation for each azimuthal mode ``m`` comes from the shape derivatives `δ𝒩`, `δ𝒮`, `δ𝒱` and `δℬ`:

```julia
D = disk(32)

# base state
σ₀ = 2β
f₀ = (γ / 2β) * σ₀ .+ (Ra / 16) * 𝒱(σ₀, D)
p₀ = -𝒩⁻¹(f₀, D)                                        # pressure
ψ₀ = 𝒮(p₀, D) .+ (Ra / 16) * ℬ(σ₀, D)
U₀ = -∂n(ψ₀, D)                                         # normal velocity of the rim

# perturbation of mode m
f₁ = δ𝒩(p₀, m, D) .+ (Ra / 16) * δ𝒱(σ₀, m, D)
p₁ = -𝒩⁻¹(f₁, D)
ψ₁ = 𝒮(p₁, D) .+ δ𝒮(p₀, m, D) .+ (Ra / 16) * δℬ(σ₀, m, D)
U₁ = -(m + 1) * U₀ .- ∂n(ψ₁, D)
σₘ = real(U₁[1])                                        # growth rate of mode m
```

The script prints ``\sigma_m`` for ``m = 0, \dots, 8``. For ``m \ge 2`` these agree to 2–5e-6 with the closed form
``\sigma_b = \frac{\beta\,\mathrm{Ra}}{96}\left(\frac{m\,\lambda_1^0\lambda_m^m}{4} - \frac{9}{2(m - 1)(2m +
1)}\right)``, which the script computes but does not print (``-0.0015625`` at ``m = 2``, ``0.0064174`` at
``m = 3``). The rates for ``m = 0`` and ``1``, about 1e-6, are zero up to the same error.

```
(m, σₘ) = (2, -0.001560448350202179)
(m, σₘ) = (3, 0.006420022296074067)
(m, σₘ) = (4, 0.009660193515409412)
…
(m, σₘ) = (8, 0.015975572486573913)
```
