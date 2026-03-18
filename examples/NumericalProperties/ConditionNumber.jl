
using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra
using Revise
using FastAlmostBandedMatrices
 


# Discretization
# println("Discretizing...")
# Mr, Mθ = 100, 100
# lmax = Mr
# D = disk(Mr);  # We should change it to a fast transform...


lmax_list = [10, 20, 40, 80, 160, 250,500] 

condi_list = Vector{Float64}[]
scondi_list = Vector{Float64}[]

condi2_list = Vector{Float64}[]
scondi2_list = Vector{Float64}[]


for i in eachindex(lmax_list)
    lmax = lmax_list[i]
    Mspan = 0:1:lmax


    # Build per-frequency system matrices
    Problem_matrices = [helmholtz_matrix(lmax, m, 50) for m in Mspan];


    condi =  cond.(Problem_matrices)
    scondi = condskeel.(real.(Problem_matrices))

    push!(condi_list, condi)
    push!(scondi_list, scondi)
    println("ddd")


    # condi2 =  cond.(bandpart.(Problem_matrices))
    # scondi2 = condskeel.(real.(Problem_matrices))

    condi2 =  [cond(Matrix(Problem_matrices[i])[2:end, 2:end]) for i in eachindex(Problem_matrices)]
    scondi2 = [condskeel(Matrix(Problem_matrices[i])[2:end, 2:end]) for i in eachindex(Problem_matrices)]

    push!(condi2_list, condi2)
    push!(scondi2_list, scondi2)
    println("xxxxx")

end

