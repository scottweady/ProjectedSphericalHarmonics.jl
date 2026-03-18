
using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra
using Revise


# Discretization
println("Discretizing...")
Mr, Mθ = 100, 100
lmax = Mr
D = disk(Mr);  # We should change it to a fast transform...

X = real.(D.ζ)
Y = imag.(D.ζ)
θ = D.θ
r = D.r

x_boundary = cos.(θ)
y_boundary = sin.(θ)

Mspan = vec(Array(D.Mspan))

k = 60
n = 10

# Build per-frequency system matrices
Problem_matrices = [helmholtz_matrix(lmax, m, k^2) for m in Mspan];


# cond.(Problem_matrices[1]), cond.(Problem_matrices[end])

f =  k^2*exp.(-abs2.(D.ζ .- (0.4 + 0.3*im)))
g = y_boundary .* cos.(n * x_boundary)

# Transform right-hand side into TriangularCoeffArray
f̂_tri = NodalToTriangularArray(f, D)
ĝ = fft(g) / length(g)

Mspan = vec(Array(D.Mspan))


# Build per-frequency right-hand side: [boundary_coeff ; even-parity coefficients]
b̂ = [[ ĝ[i] ; mode_coefficients(f̂_tri, Mspan[i]) ] for i in eachindex(Mspan)];

# Solve per frequency
Solution_vector = [Problem_matrices[i] \ b̂[i] for i in eachindex(Mspan)];

# Reconstruct û: apply Δ⁻¹ to the interior part, then add boundary contribution
û_tri = TriangularCoeffArray{Float64}(lmax, Mspan)

for (i, m) in enumerate(Mspan)
    res = mode_coefficients(û_tri, m)
    Ĝᵐ!(res, Solution_vector[i][2:end], lmax, m)
    res[1] += Solution_vector[i][1]
end





# Convert TriangularCoeffArray back to full PSH matrix for ipsh
û_psh = TriangularArrayToPSH(û_tri, D)

u = ipsh(û_psh, D)


R_samples = collect(0.0:0.01:1.0)
index_r = findfirst(x -> x > 0.25, R_samples)
Θ_samples = [D.θ..., 2π]

u_interp = ipsh(û_psh, D, R_samples)
u_interp = hcat(u_interp, u_interp[:,1])
X_samples = (R_samples' .* cos.(Θ_samples))'
Y_samples = (R_samples' .* sin.(Θ_samples))'


zs = real.(u_interp)
levels = Makie.get_tickvalues(Makie.LinearTicks(20), extrema(zs[1:index_r,:])...)



fig = Figure()
ax1 = Axis(fig[1, 1], aspect=DataAspect())
ax2 = Axis(fig[1, 2], aspect=DataAspect())
srf1 = surface!(ax1, X_samples, Y_samples, fill(0f0, size(zs)); color=zs, colorrange=extrema(zs), shading=NoShading, colormap=:coolwarm)
srf2 = surface!(ax2, X_samples, Y_samples, fill(0f0, size(zs)); color=zs, colorrange=extrema(zs), shading=NoShading, colormap=:coolwarm)

xlims!(ax2, -0.2, 0.2)
ylims!(ax2, -0.2, 0.2)

ctr = contour!(ax2, X_samples[1:index_r,:], Y_samples[1:index_r,:], max.(0.000, zs[1:index_r,:]); color=:black, levels=levels, labels=false)
ctr = contour!(ax2, X_samples[1:index_r,:], Y_samples[1:index_r,:], min.(0.000, zs[1:index_r,:]); color=:black, levels=levels, labels=false, linestyle=:dash)

display(fig)
maximum(abs.(u_interp))
#20.732932219270985
# fig, ax, srf = surface(X_samples, Y_samples, fill(0f0, size(zs)); color=zs, colorrange=extrema(zs), shading=NoShading, axis=(; type=Axis, aspect=DataAspect()), colormap=:coolwarm)




# xlims!(ax, -0.2, 0.2)
# ylims!(ax, -0.2, 0.2)

# ctr = contour!(ax, X_samples[1:index_r,:], Y_samples[1:index_r,:], max.(0.000, zs[1:index_r,:]); color=:black, levels=levels, labels=false)
# ctr = contour!(ax, X_samples[1:index_r,:], Y_samples[1:index_r,:], min.(0.000, zs[1:index_r,:]); color=:black, levels=levels, labels=false, linestyle=:dash)

# fig
