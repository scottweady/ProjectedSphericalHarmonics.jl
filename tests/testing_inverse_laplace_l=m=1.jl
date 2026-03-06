using ProjectedSphericalHarmonics
using LinearAlgebra
using AssociatedLegendrePolynomials


function Plm_(l, m, x)
    if m == 0
        return Plm(l, m, x)
    elseif m > 0
        return Plm(l, m, x)
    else
        return Plm(l, abs(m), x)*(-1)^abs(m)*factorial(l-abs(m))/factorial(l+abs(m))
    end
end


# Discretize disk
Mr, Mθ = 32, 16
D = disk(Mr, Mθ)

# Get grid points and weight function
ζ = D.ζ
w = D.w
r, θ = abs.(ζ), angle.(ζ)

# Single layer operator
l, m = 1, 1
# u = ylm(l, m, ζ)

f = Plm_.(l, m, sqrt.(1 .- r.^2)) .* exp.(im * m * θ )
Ninvf = 𝒩⁻¹(f, D)
Ninvf_exact = -4/ProjectedSphericalHarmonics.λlm(l, m)/(2*l+1)*(1/(l+m+1)*Plm_.(l+1, m, sqrt.(1 .- r.^2)) ) .* exp.(im * m * θ)

norm(Ninvf - Ninvf_exact)


wNinvf = D.w .* Ninvf
wNinvf_exact = -exp.(im * m * θ ) .* 4/ProjectedSphericalHarmonics.λlm(l, m)/(2*l+1) .*(
    (l-m+2)/(2*l+3)/(l+m+1) * Plm_.(l+2, m, sqrt.(1 .- r.^2)) +
    1/(2l+3) * Plm_.(l, m, sqrt.(1 .- r.^2))
)

norm(wNinvf - wNinvf_exact)



SNinvf = 𝒮(Ninvf, D)
SNinvf_exact = -1/(2l+1) * exp.(im * m * θ ) .* (
    1/(2*l+3)/(2l+2) * Plm_.(l+2, l, sqrt.(1 .- r.^2)) +
    1/(2*l+3) * Plm_.(l, l, sqrt.(1 .- r.^2)) 
)

norm(SNinvf - SNinvf_exact)


#Testing its derivatives
polar_coord = (x,y) -> (sqrt(x^2+y^2), atan(y,x))
SNinvf_func =  (r, θ) -> -1/(2l+1) * exp(im * m * θ ) * (
    (l-m+1)/(2*l+3)/(l+m+2) * Plm_(l+2, m, sqrt(1 - r^2)) +
    1/(2*l+3) * Plm_(l, m, sqrt(1 - r^2)) 
)
SNinvf_func_cart = (x,y) -> SNinvf_func(polar_coord(x,y)...)

δx = 1e-5
δy = 1e-5
x,y = 0.5, 0.8

∂xSNinvf_approx(x, y) = (SNinvf_func_cart(x + δx, y) - SNinvf_func_cart(x - δx, y)) / (2 * δx)
∂ySNinvf_approx(x, y) = (SNinvf_func_cart(x, y + δy) - SNinvf_func_cart(x, y - δy)) / (2 * δy)

∂x∂xSNinvf_approx(x, y) = (SNinvf_func_cart(x + δx, y) - 2 * SNinvf_func_cart(x, y) + SNinvf_func_cart(x - δx, y)) / (δx^2)
∂x∂ySNinvf_approx(x, y) = (SNinvf_func_cart(x + δx, y + δy) - SNinvf_func_cart(x + δx, y - δy) - SNinvf_func_cart(x - δx, y + δy) + SNinvf_func_cart(x - δx, y - δy)) / (4 * δx * δy)
∂y∂ySNinvf_approx(x, y) = (SNinvf_func_cart(x, y + δy) - 2 * SNinvf_func_cart(x, y) + SNinvf_func_cart(x, y - δy)) / (δy^2)


#First Derivative Exact
∂zSNinvf_exact = (r, θ) -> 1/2/(2l+1)*exp(im*(l-1)*θ)*(Plm_(l+1, l-1, sqrt(1 - r^2)))
∂z̄SNinvf_exact = (r, θ) -> -1/4/(2l+1)*exp(im*(l+1)*θ)*1/(l+1)*Plm_(l+1, l+1, sqrt(1 - r^2))


#Second Derivative Exact
∂z̄∂z̄SNinvf_exact = (r, θ) -> 0

∂z∂zSNinvf_exact = (r, θ) -> -Plm_(1, -1, sqrt(1 - r^2))*exp(-im*1*θ)/2

∂z∂z̄SNinvf_exact = (r, θ) -> 1/4*exp(im*(l)*θ)*Plm_(l, l, sqrt(1 - r^2))


#First derivative test
∂zSNinvf_exact(polar_coord(x,y)...) + ∂z̄SNinvf_exact(polar_coord(x,y)...) - ∂xSNinvf_approx(x, y) 
(-∂zSNinvf_exact(polar_coord(x,y)...) + ∂z̄SNinvf_exact(polar_coord(x,y)...))/im - ∂ySNinvf_approx(x, y)


#Second derivative test
#Laplacian
4*∂z∂z̄SNinvf_exact(polar_coord(x,y)...) - (∂x∂xSNinvf_approx(x, y) + ∂y∂ySNinvf_approx(x, y))

#AntiLaplacian
2*∂z∂zSNinvf_exact(polar_coord(x,y)...) + 2*∂z̄∂z̄SNinvf_exact(polar_coord(x,y)...) -(∂x∂xSNinvf_approx(x, y) - ∂y∂ySNinvf_approx(x, y))




#cross derivative

im*∂z∂zSNinvf_exact(polar_coord(x,y)...) - im*∂z̄∂z̄SNinvf_exact(polar_coord(x,y)...)- ∂x∂ySNinvf_approx(x, y)


########
