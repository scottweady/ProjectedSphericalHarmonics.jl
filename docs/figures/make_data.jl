# The data behind the figures of ProjectedSphericalHarmonics' documentation, as CSV files in docs/figures/data/
# (plot_figures.py reads them). Every error is measured here against a closed form, a manufactured solution or an
# upsampled direct sum; the timings are the minimum of a few repeats (about 7 minutes in all, 8 threads).
#   julia -t 8 --project=docs docs/figures/make_data.jl          (from the package root)
ENV["OMP_NUM_THREADS"] = string(Threads.nthreads())       # FMM3D's OpenMP threads, as many as Julia's
using ProjectedSphericalHarmonics, LinearAlgebra, Printf, Random
using SpecialFunctions: ellipk, ellipe
const PSH = ProjectedSphericalHarmonics

const DATA = joinpath(@__DIR__, "data")
mkpath(DATA)
cell(v) = v isa AbstractString ? "\"" * v * "\"" : string(v)      # labels quoted (they hold commas)
csv(name, header, rows) = open(joinpath(DATA, name), "w") do io
    println(io, join(header, ","))
    for r in rows
        println(io, join(cell.(r), ","))
    end
    println("wrote data/", name, " (", length(rows), " rows)")
end
loadavg() = (s = read(`uptime`, String); strip(s[findlast("load average", s).start:end]))
T0 = time()

# Closed forms on the ellipse with semi-axes a ≥ b (examples/ellipse.jl), m = 1 − b²/a²: the density of 𝒮⁻¹(1) is
# σ = 2/(b K(m) ω_e), ω_e = √(1 − x²/a² − y²/b²), and the drag on the plate translating along a is 4π/A,
# A = ((2a² − b²)K(m) − a²E(m))/(a(a² − b²)) (16/3 on the unit disk, the limit a → b)
σ_exact(a, b) = 2 / (b * ellipk(1 - (b / a)^2))
drag_exact(a, b) = a ≈ b ? 16a / 3 : (m = 1 - (b / a)^2; 4π * a * (a^2 - b^2) / ((2a^2 - b^2) * ellipk(m) - a^2 * ellipe(m)))
ωe(z, a, b) = sqrt(max(1 - (real(z) / a)^2 - (imag(z) / b)^2, 0.0))

# Even PSH coefficients of g = f·ω on the grid, and g at points ζ of the unit disk (Σ ĝ_ℓ^m y_ℓ^m(ζ))
function synth(ĝ, Ω, ζ)
    u = zeros(ComplexF64, size(ζ))
    for (mi, m) in enumerate(Ω.Mspan), l in abs(m):Ω.Mℓ
        Ω.even[l + 1, mi] && (u .+= ĝ[l + 1, mi] .* ylm(l, m, ζ))
    end
    return real.(u)
end

"Random real density g = f·ω with decaying coefficients (test/test_close_eval.jl); its even PSH coefficients."
function random_density(D, rng)
    c = zeros(ComplexF64, D.shp)
    for l in 0:D.Mℓ, mi in eachindex(D.Mspan)
        D.even[l + 1, mi] && (c[l + 1, mi] = randn(rng, ComplexF64) * exp(-0.25l))
    end
    return psh(real.(ipsh(c, D)), D)
end

"Stokes single layer (μ = 1, body frame of the ellipse (a, b)) of f = g/ω, ĝ the even coefficients, at `targets`, by
direct sum over an (Nr, Nθ) polar grid of the ellipse."
function ellipse_direct(a, b, ĝ, D, Nr, Nθ, targets)
    r, θ, ζ, dζ = PSH.diskpts(Nr, Nθ)
    # on the tensor grid: the radial profile of each order m, times e^{imθ}
    tensor(c) = real.(sum(sum(c[l + 1, mi] .* ylm(l, m, r .+ 0im) for l in abs(m):D.Mℓ if D.even[l + 1, mi]) .* cis.(m .* θ)
                          for (mi, m) in enumerate(D.Mspan)))
    wt = a * b .* real.(dζ) ./ sqrt.(1 .- abs2.(ζ))
    fx, fy = vec(tensor(ĝ[1]) .* wt), vec(tensor(ĝ[2]) .* wt)
    stokeslet_sum(vec(a .* real.(ζ) .+ im .* b .* imag.(ζ)), fx, fy, targets)
end

"Σⱼ G(x − yⱼ) Fⱼ, G = (I/ρ + r rᵀ/ρ³)/4π, at the complex `targets` from the point forces (Fx, Fy) at `src`."
function stokeslet_sum(src, Fx, Fy, targets)
    u = zeros(2, length(targets))
    xs, ys = real.(src), imag.(src)
    Threads.@threads for t in eachindex(targets)
        X, Y = real(targets[t]), imag(targets[t])
        ux = uy = 0.0
        @inbounds for j in eachindex(xs)
            dx, dy = X - xs[j], Y - ys[j]
            r2 = dx^2 + dy^2
            ir = 1 / sqrt(r2); ir3 = ir / r2
            d = dx * Fx[j] + dy * Fy[j]
            ux += Fx[j] * ir + dx * d * ir3
            uy += Fy[j] * ir + dy * d * ir3
        end
        u[1, t], u[2, t] = ux / 4π, uy / 4π
    end
    return u
end

# ---- hero (a): a projected spherical harmonic on the disk, with the nodes of disk(16) ----------------------------
t0 = time()
ℓ, m = 9, 3
rs = range(0, 1, length=121); θs = range(0, 2π, length=361)
csv("hero_ylm.csv", ("r", "theta", "y"), [(r, θ, real(ylm(ℓ, m, r * cis(θ) + 0im))) for θ in θs for r in rs])
D = disk(16)
csv("hero_nodes.csv", ("x", "y"), [(real(z), imag(z)) for z in vec(D.z)])

# ---- hero (b): a suspension of ellipses under applied forces and torques -----------------------------------------
# 8 ellipses a = 1, b = 0.45, M = 16, placed at random without overlap (edge gaps ≥ 0.2), forces and torques of unit
# size; suspension_mobility_solve (Ewald, eps = gmrestol = 1e-10), then the flow on a grid, by direct sums over the
# nodes of every body, and by the closed-form exterior map near a body (|ξ| < 1.6 in its normalized frame)
a, b, M = 1.0, 0.45, 16
Ω = ellipse(a, b, M)
rng = MersenneTwister(7)
outline(c, θ, n=120) = [c + cis(θ) * (a * cos(t) + im * b * sin(t)) for t in range(0, 2π, length=n)]
xcm = ComplexF64[]; θcm = Float64[]
while length(xcm) < 8
    c = complex(8.4 * (rand(rng) - 0.5), 4.4 * (rand(rng) - 0.5)); θ = π * rand(rng)
    if all(minimum(abs.(outline(c, θ) .- transpose(outline(x, t)))) > 0.2 &&
           abs(c - x) > a + b + 0.3 for (x, t) in zip(xcm, θcm))
        push!(xcm, c); push!(θcm, θ)
    end
end
F = [[cos(φ), sin(φ), 0.5 * randn(rng)] for φ in 2π .* rand(rng, 8)]
f, U = suspension_mobility_solve(xcm, θcm, nothing, nothing, F, Ω; eps=1e-10, gmrestol=1e-10)
ĝ = [psh(f[i] .* Ω.w, Ω) for i in eachindex(f)]            # (x, y) coefficients of f·ω, body frame
geom = PSH.exterior_geometry(Ω)
xs = range(-6.0, 6.0, length=241); ys = range(-3.6, 3.6, length=145)
grid = [complex(x, y) for y in ys for x in xs]
ξ(i, z) = (w = cis(-θcm[i]) * (z - xcm[i]); real(w) / a + im * imag(w) / b)
inside = [any(abs(ξ(i, z)) < 1 for i in eachindex(xcm)) for z in grid]
ug = zeros(2, length(grid))
for i in eachindex(xcm)
    zw = xcm[i] .+ cis(θcm[i]) .* vec(Ω.z)                    # world-frame nodes and point forces of body i
    Fx, Fy = PSH.rotate(real.(vec(f[i][1])), real.(vec(f[i][2])), θcm[i])
    dz = real.(vec(Ω.dz))
    near = [!inside[k] && abs(ξ(i, grid[k])) < 1.6 for k in eachindex(grid)]
    far = findall(.!inside .& .!near); near = findall(near)
    ug[:, far] .+= stokeslet_sum(zw, Fx .* dz, Fy .* dz, grid[far])
    T = PSH.exterior_map(geom, Ω, [ξ(i, z) for z in grid[near]])
    v = real.(T * vcat(vec(ĝ[i][1])[Ω.idx_even], vec(ĝ[i][2])[Ω.idx_even]))
    n = length(near)
    ux, uy = PSH.rotate(v[1:n], v[n+1:end], θcm[i])
    ug[1, near] .+= ux; ug[2, near] .+= uy
end
ug[:, inside] .= NaN
csv("hero_flow.csv", ("x", "y", "ux", "uy"), [(real(z), imag(z), ug[1, k], ug[2, k]) for (k, z) in enumerate(grid)])
# |f|·ω on a polar grid of each body (bounded: f has a 1/ω edge singularity)
ζb = [r * cis(t) + 0im for t in range(0, 2π, length=121) for r in range(0, 1, length=31)]
rows = []
for i in eachindex(xcm)
    gx, gy = synth(ĝ[i][1], Ω, ζb), synth(ĝ[i][2], Ω, ζb)
    for (k, ζ) in enumerate(ζb)
        z = xcm[i] + cis(θcm[i]) * (a * real(ζ) + im * b * imag(ζ))
        push!(rows, (i, abs(ζ), mod(angle(ζ), 2π), real(z), imag(z), hypot(gx[k], gy[k])))
    end
end
csv("hero_bodies.csv", ("body", "r", "theta", "x", "y", "fw"), rows)
csv("hero_motion.csv", ("body", "xc", "yc", "theta", "Fx", "Fy", "T", "Ux", "Uy", "omega"),
    [(i, real(xcm[i]), imag(xcm[i]), θcm[i], F[i]..., U[i]...) for i in eachindex(xcm)])
@printf("hero (b): |u| ≤ %.3f off the bodies, |f|ω ≤ %.3f; hero total %.0f s\n",
        maximum(filter(!isnan, hypot.(ug[1, :], ug[2, :]))), maximum(r[end] for r in rows), time() - t0)

# ---- convergence (a, b): 𝒮⁻¹(1) and the drag of a translating plate on ellipses, in M ----------------------------
# by ellipse(a, b, M) (affine map of the disk) and by domain(γ, M) (conformal map from 256 boundary samples)
t0 = time()
lap = []; sto = []
cases = [("ellipse", 1.5, 1.0, 4:4:48), ("ellipse", 4.0, 1.0, 4:4:48),
         ("domain", 1.5, 1.0, 4:4:32), ("domain", 2.0, 1.0, 4:4:32)]
for (kind, a, b, Ms) in cases, M in Ms
    local Ω
    θ = range(0, 2π, length=257)[1:end-1]
    Ω = kind == "ellipse" ? ellipse(a, b, M) : domain(a .* cos.(θ) .+ im .* b .* sin.(θ), M)
    label = @sprintf("%s, a/b = %g", kind, a / b)
    σ = real.(𝒮⁻¹(1.0, Ω))
    push!(lap, (M, label, maximum(abs.(σ .* ωe.(Ω.z, a, b) .- σ_exact(a, b))) / σ_exact(a, b)))
    fx = 𝒮_st⁻¹((1.0, 0.0), Ω)[1]
    push!(sto, (M, label, abs(real(integral(fx, Ω)) - drag_exact(a, b)) / drag_exact(a, b)))
end
for M in 4:4:48           # the unit disk: drag and torque 16/3
    Mm = stokes_mobility_matrix(disk(M))
    push!(sto, (M, "disk, drag", abs(real(Mm[1, 1]) - 16 / 3) / (16 / 3)))
    push!(sto, (M, "disk, torque", abs(real(Mm[3, 3]) - 16 / 3) / (16 / 3)))
end
csv("conv_laplace.csv", ("M", "case", "error"), lap)
csv("conv_stokes.csv", ("M", "case", "error"), sto)
@printf("convergence (a, b): %.0f s\n", time() - t0)

# ---- convergence (c): the Poisson solver Δ⁻¹(f, g, D) against u = eˣ sin y + cos(2x + y), in M --------------------
t0 = time()
uex(z) = exp(real(z)) * sin(imag(z)) + cos(2real(z) + imag(z))
rows = []
for M in 4:2:32
    local D = disk(M)
    u = Δ⁻¹(-5 .* cos.(2 .* real.(D.z) .+ imag.(D.z)), uex.(cis.(D.θ)), D)
    push!(rows, (M, "Δ⁻¹(f, g, D)", maximum(abs.(u .- uex.(D.z))) / maximum(abs.(uex.(D.z)))))
end
csv("conv_poisson.csv", ("M", "case", "error"), rows)
@printf("convergence (c): %.0f s\n", time() - t0)

# ---- convergence (d): cross velocity between two close bodies against the edge gap -------------------------------
# Two bodies side by side, long axes parallel (as test/test_close_eval.jl), M = 16, random smooth densities; the
# velocity of suspension_velocity_eval (eps = 1e-13) at the nodes of body 2, with the closed form (close_eval=true)
# and with the node quadrature only, against the self term plus a direct sum over a 400 × 16384 polar grid of body 1
# (checked against 200 × 4096: the column ref_check)
t0 = time()
rows = []
M = 16
for (name, a, b) in (("disk", 1.0, 1.0), ("ellipse, a/b = 4", 2.0, 0.5))
    local Ω, rng, xcm, θcm, f
    Ω = a == b ? disk(M) : ellipse(a, b, M)
    rng = MersenneTwister(1)
    f̃ = [(random_density(Ω, rng), random_density(Ω, rng)) for _ in 1:2]
    f = [map(c -> real.(ipsh(c, Ω)) ./ Ω.w .+ 0im, fi) for fi in f̃]
    u_self = 𝒮_st(f[2], Ω)
    for gap in (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
        xcm = [0.0 + 0.0im, 2b + gap + 0.3im]; θcm = [π / 2, -π / 2]
        targets = cis(-θcm[1]) .* (xcm[2] .+ cis(θcm[2]) .* vec(Ω.z) .- xcm[1])
        ref = map(((Nr, Nθ),) -> begin
            u12 = ellipse_direct(a, b, f̃[1], Ω, Nr, Nθ, targets)
            ux, uy = PSH.rotate(u12[1, :], u12[2, :], θcm[1] - θcm[2])
            vcat(vec(real.(u_self[1])) .+ ux, vec(real.(u_self[2])) .+ uy)
        end, ((400, 16384), (200, 4096)))
        scale = maximum(abs, ref[1])
        e(u) = maximum(abs, vcat(vec(real.(u[2][1])), vec(real.(u[2][2]))) .- ref[1]) / scale
        ec = e(suspension_velocity_eval(xcm, θcm, f, Ω; eps=1e-13))
        en = e(suspension_velocity_eval(xcm, θcm, f, Ω; eps=1e-13, close_eval=false))
        push!(rows, (name, gap, ec, en, maximum(abs, ref[1] .- ref[2]) / scale))
    end
end
csv("conv_close.csv", ("case", "gap", "closed_form", "node_quadrature", "ref_check"), rows)
@printf("convergence (d): %.0f s\n", time() - t0)

# ---- suspension: timings against the number of bodies, disk(16, 16) ------------------------------------------------
# Bodies on a square lattice of spacing 3 (edge gaps about 1), jittered by ±0.2, random orientations. (a) one cross-
# body Stokeslet sum over all N·561 nodes at eps = 1e-6: the Ewald plan (StokesEwald) and its apply (stokes_ewald!)
# timed separately, FMM3D's stfmm3d, and the direct sum st3ddir for N ≤ 32; minimum of 3 repeats. (b) one
# suspension_mobility_solve (applied forces and torques, no background flow) at eps = gmrestol = 1e-8 by both fast
# sums, S_self and the mobility matrix precomputed; minimum of 2 repeats for N ≤ 64, 1 above
t0 = time()
D = disk(16, 16); Nz = length(D.z)
S = suspension_self_matrix(D); Mm = stokes_mobility_matrix(D)
function lattice(N, rng; h=3.0)
    n = ceil(Int, sqrt(N))
    [complex(h * (k % n), h * (k ÷ n)) + 0.4complex(rand(rng) - 0.5, rand(rng) - 0.5) for k in 0:N-1], 2π .* rand(rng, N)
end
best(f, n) = minimum(@elapsed(f()) for _ in 1:n)
# warm up (compilation)
let (x, t) = lattice(4, MersenneTwister(0)), F = [[1.0, 0.0, 0.0] for _ in 1:4]
    src = PSH.suspension_sources!(zeros(3, 4Nz), x, t, D); stk = randn(3, 4Nz)
    PSH.stokes_ewald!(similar(src), PSH.StokesEwald(src; tol=1e-6), stk); PSH.stfmm3d(1e-6, src; stoklet=stk, ppreg=1)
    PSH.st3ddir(src, src; stoklet=stk, ppregt=1)
    for meth in (:ewald, :fmm)
        suspension_mobility_solve(x, t, nothing, nothing, F, D; eps=1e-8, gmrestol=1e-8, M=Mm, S_self=S, method=meth)
    end
end
println("timings start, load ", loadavg())
sums = []; solves = []
for N in (2, 4, 8, 16, 32, 64, 128, 256)
    local rng, xcm, θcm, F
    rng = MersenneTwister(N)
    xcm, θcm = lattice(N, rng)
    src = PSH.suspension_sources!(zeros(3, N * Nz), xcm, θcm, D)
    stk = zeros(3, N * Nz); stk[1:2, :] = randn(rng, 2, N * Nz)
    E = PSH.StokesEwald(src; tol=1e-6)
    push!(sums, (N, N * Nz, "Ewald plan", best(() -> PSH.StokesEwald(src; tol=1e-6), 3)))
    push!(sums, (N, N * Nz, "Ewald apply", best(() -> PSH.stokes_ewald!(similar(src), E, stk), 3)))
    push!(sums, (N, N * Nz, "FMM3D", best(() -> PSH.stfmm3d(1e-6, src; stoklet=stk, ppreg=1), 3)))
    N <= 32 && push!(sums, (N, N * Nz, "direct", best(() -> PSH.st3ddir(src, src; stoklet=stk, ppregt=1), 2)))
    F = [[randn(rng), randn(rng), 0.3randn(rng)] for _ in 1:N]
    npairs = length(PSH.close_eval_plan(xcm, θcm, D))
    for meth in (:ewald, :fmm)
        log, io = mktemp()
        t = minimum(1:(N <= 64 ? 2 : 1)) do _
            redirect_stdout(io) do       # verbose=true prints the iteration count
                @elapsed suspension_mobility_solve(xcm, θcm, nothing, nothing, F, D; eps=1e-8, gmrestol=1e-8, M=Mm,
                                                   S_self=S, method=meth, verbose=true)
            end
        end
        close(io)
        its = parse(Int, match(r"in (\d+) iterations", read(log, String)).captures[1]); rm(log)
        push!(solves, (N, N * Nz, meth == :ewald ? "Ewald" : "FMM3D", t, its, npairs))
    end
    @printf("N = %3d: Ewald plan %.3f s, apply %.4f s, FMM3D %.3f s; solve Ewald %.2f s, FMM3D %.2f s (%d its)\n", N,
            sums[end-2-(N<=32)][4], sums[end-1-(N<=32)][4], sums[end-(N<=32)][4], solves[end-1][4], solves[end][4], solves[end][5])
end
csv("susp_sum.csv", ("N", "points", "method", "time"), sums)
csv("susp_solve.csv", ("N", "points", "method", "time", "iterations", "close_pairs"), solves)
@printf("suspension: %.0f s, load after %s\n", time() - t0, loadavg())
@printf("total: %.0f s\n", time() - T0)
