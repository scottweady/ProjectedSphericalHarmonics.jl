
using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra
using BenchmarkTools


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

f = k^2 * exp.(-abs2.(D.ζ .- (0.4 + 0.3 * im)))
# f = cos.(40*sin.(3*X.*Y) .+ 1)
g = y_boundary .* cos.(n * x_boundary)

Problem_matrices = [helmholtz_matrix(lmax, m, k^2) for m in Array(D.Mspan)];


function BigMatrix(f, g, D)

    f̂ = psh(f, D)
    ĝ = fft(g) / length(g)

    b̂ = [[ĝ[i]; f̂[D.even[:, i], i]] for i in eachindex(Array(D.Mspan))]


    Solution_vector = [Problem_matrices[i] \ b̂[i] for i in eachindex(Array(D.Mspan))]

    #reconstruct u
    Gμ̂ = [Ĝᵐ(Solution_vector[i][2:end], lmax, D.Mspan[i]; aliasing=false) for i in eachindex(Array(D.Mspan))]

    û = deepcopy(Gμ̂)
    for i in eachindex(Array(D.Mspan))
        # println("Reconstructing u for m = ", Array(D.Mspan)[i])
        û[i][1] += Solution_vector[i][1]
    end

    #transform û into psh space format

    û_psh = zeros(ComplexF64, size(f̂))
    for i in eachindex(Array(D.Mspan))
        m = Array(D.Mspan)[i]
        if m < 0
            û_psh[abs(m)+1:2:end, end-(abs(m)-1)] .= û[i]
        else
            û_psh[m+1:2:end, m+1] .= û[i]
        end
    end

    # u = ipsh(û_psh, D)

    return nothing




end


function triangular_array(f, g, D)

    f̂_tri = NodalToTriangularArray(f, D)
    ĝ = fft(g) / length(g)
    Mspan = vec(Array(D.Mspan))

    # Build per-frequency right-hand side: [boundary_coeff ; even-parity coefficients]
    b̂ = [[ĝ[i]; mode_coefficients(f̂_tri, Mspan[i])] for i in eachindex(Mspan)]

    # Solve per frequency
    Solution_vector = [Problem_matrices[i] \ b̂[i] for i in eachindex(Mspan)]

    # Reconstruct û: apply Δ⁻¹ to the interior part, then add boundary contribution
    û_tri = TriangularCoeffArray{Float64}(lmax, Mspan)

    for (i, m) in enumerate(Mspan)
        res = mode_coefficients(û_tri, m)
        Ĝᵐ!(res, Solution_vector[i][2:end], lmax, m)
        res[1] += Solution_vector[i][1]
    end

    # Convert TriangularCoeffArray back to full PSH matrix for ipsh
    # û_psh = TriangularArrayToPSH(û_tri, D)

    # u = ipsh(û_psh, D)

    return nothing

end

BigMatrix(f, g, D)
triangular_array(f, g, D)

b1 = @benchmark BigMatrix($f, $g, $D)
b2 = @benchmark triangular_array($f, $g, $D)
