
using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra


# Discretization
println("Discretizing...")
Mr, Mθ = 200, 200
lmax = Mr
D = disk(Mr);

X = real.(D.ζ)
Y = imag.(D.ζ)
θ = D.θ
r = D.r

x_boundary = cos.(θ)
y_boundary = sin.(θ)


k = 60
n = 10

# u_manufactured = Y.*cos.(n*X) 

# f = -n^2*u_manufactured + k^2*u_manufactured

f = k^2*exp.( - abs2.(D.ζ .-(0.4 + 0.3*im)) )
# f = cos.(40*sin.(3*X.*Y) .+ 1)
g = y_boundary.*cos.(n*x_boundary)
#g = x_boundary.*cos.(n*y_boundary) 

f̂ = psh(f, D)
ĝ = fft(g)/length(g)



sum(D.even[:,1])
ModifiedPoissonSystemMatrix(lmax, 0, k^2) 


Problem_matrices = [ModifiedPoissonSystemMatrix(lmax, m, k^2) for m in Array(D.Mspan)];


b̂ = [ [ ĝ[i] ; f̂[D.even[:,i], i]] for i in eachindex(Array(D.Mspan))];

sum(length.(b̂))

#Solution_vector


Solution_vector = [ Problem_matrices[i] \ b̂[i] for i in eachindex(Array(D.Mspan))];

#reconstruct u
Gμ̂ = [ Inverse_laplacian_coef_m_sparse(Solution_vector[i][2:end], lmax, D.Mspan[i]; aliasing = false) for i in eachindex(Array(D.Mspan))]; 

û = deepcopy(Gμ̂ )
for i in eachindex(Array(D.Mspan))
    # println("Reconstructing u for m = ", Array(D.Mspan)[i])
    û[i][1] += Solution_vector[i][1] 
end

#transform û into psh space format

û_psh = zeros(ComplexF64, size(f̂))
for i in eachindex(Array(D.Mspan))
    m = Array(D.Mspan)[i]
    if m < 0
        û_psh[abs(m) + 1:2:end, end - (abs(m) - 1)] .= û[i]
    else
        û_psh[m + 1:2:end, m + 1] .= û[i]
    end
end


u = ipsh(û_psh, D)


# norm(u, Inf)
# norm(u_manufactured, Inf)
# norm(u - u_manufactured, Inf)


R_samples = collect(0.0:0.001:1.0)
index_r = findfirst(x -> x > 0.25, R_samples)
Θ_samples = [D.θ..., 2π ]

u_interp = ipsh(û_psh, D, R_samples)
u_interp = hcat(u_interp , u_interp[:,1])
X_samples = (R_samples' .* cos.(Θ_samples))'
Y_samples = (R_samples' .* sin.(Θ_samples))'



zs = real.(u_interp)
levels = Makie.get_tickvalues(Makie.LinearTicks(20), extrema(zs[1:index_r,:])...)

# fig,ax, srf = surface(X_samples[1:10,:], Y_samples[1:10,:], fill(0f0, size(zs)); color= zs[1:10,:], colorrange = extrema(zs[1:10,:]) , shading = NoShading, axis = (; type = Axis, aspect = DataAspect()), colormap = :coolwarm)

fig,ax, srf = surface(X_samples, Y_samples, fill(0f0, size(zs)); color= zs, colorrange = extrema(zs) , shading = NoShading, axis = (; type = Axis, aspect = DataAspect()), colormap = :coolwarm)

xlims!(ax, -0.2, 0.2)
ylims!(ax, -0.2, 0.2)


# ctr = contour!(ax, X_samples[1:20,:], Y_samples[1:20,:], zs[1:20,:]; color = :black, levels = levels, labels = false, linestyle = :dash)
ctr = contour!(ax, X_samples[1:index_r,:], Y_samples[1:index_r,:], max.(0.000, zs[1:index_r,:]); color = :black, levels = levels, labels = false)
ctr = contour!(ax, X_samples[1:index_r,:], Y_samples[1:index_r,:], min.(0.000, zs[1:index_r,:]); color = :black, levels = levels, labels = false, linestyle = :dash)

fig