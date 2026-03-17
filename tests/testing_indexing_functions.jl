using Test
using LinearAlgebra
using FFTW
using ProjectedSphericalHarmonics

MR = 32
MTHETA = 32
D = disk(MR, MTHETA)
ZETA = D.ζ
X = real.(ZETA)
Y = imag.(ZETA)
lmax = MR 


for i in eachindex(D.Mspan)

    println("****************************************************")
    m = D.Mspan[i]
    n = size_current_m(lmax, m; aliasing = false)
    println(sum(D.even[:, i]) - n)

end

