function SingleLayer(res, μ)


    #μ equispaced grid values

    #Implements S[μ] = - 1/(2π) ∫ log|x - y| μ(y) dy
    #μ(θ) = ∑ μ̂ₘ exp(i m θ)

    #res is the fourier coefficients of S[μ]
    res .= μ
    fft!(res)
    N = length(μ̂)
    freq = fftfreq(N,N)
    res[1] = 0 
    res[2:end] .= res[2:end] ./ (2*abs.(freq[2:end]))



end


function CauchyLayer(res, μ)

    #μ equispaced grid values

    #Implements limit of C[μ] = lim_{|z| → 1⁻} 1/(2π) ∫ 1/(z - w) μ(w) dw
    #μ(θ) = ∑ μ̂ₘ exp(i m θ)

    res .= μ
    fft!(res)
    N = length(μ̂)
    freq = fftfreq(N,N)

    res .= -im*res .* (freq .> -0.5 )


end


function DoubleLayer(res, μ)
    #The double layer is the identity for the inner limit

    res .= μ
    fft!(res)

end


function LogarithmicIntegral(f, D)

    f̂ = psh_triangular(f, D; parity = even) 

    #compute inverse laplacian

    #Gf̂

    #compute dirichlet and neumann trace Gf̂trig   

    #TG
    


end