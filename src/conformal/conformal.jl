
using FFTW

function derivative_fourier(u, L, p=1)

  N = length(u)
  û = fft(u)
  k = collect(fftfreq(N, N))
  isodd(p) && (k[N÷2+1] *= N % 2)

  return (2π / L)^p * ifft((1im .* k) .^ p .* û)

end

function upsample(u, α)

  N = length(u)
  û = fft(u)
  M = floor(Int, N * α)

  û_pad = zeros(eltype(û), M)
  û_pad[1:N÷2+1]          = û[1:N÷2+1]
  û_pad[end-N+N÷2+2:end]  = û[N÷2+2:end]

  if iseven(N)
    û_pad[N÷2+1] /= 2
    û_pad[end-N+N÷2+1] = û_pad[N÷2+1]
  end

  return ifft(û_pad) * (M / N)

end

function cauchy_operator(ζ, f, γ, γₛ, w)

  num, den = zero(ComplexF64), zero(ComplexF64)
  @inbounds for k in eachindex(γ)
    Δz = γ[k] - ζ
    iszero(Δz) && return f[k]
    q = w[k] * γₛ[k] / Δz
    num += f[k] * q
    den += q
  end
  return num / den

end

function generate_holomorphic_data(τ, L, γ, w, γₛ)

  # Σⱼ (τᵢ - τⱼ)/(γᵢ - γⱼ) wⱼ γₛⱼ, with τₛᵢ wᵢ for j = i
  τₛ = derivative_fourier(τ, L)
  N = length(τ)
  h = similar(γ, ComplexF64)
  Threads.@threads for i = 1 : N
    acc = τₛ[i] * w[i]
    @inbounds for j = 1 : N
      j == i && continue
      acc += (τ[i] - τ[j]) / (γ[i] - γ[j]) * (w[j] * γₛ[j])
    end
    h[i] = -τ[i] - acc / (2π * 1im)
  end
  return h

end

# Double layer -I/2 + K on the curve γ (normals n̂, curvature κ, weights ẇ), with the limit -κ/2 of the
# kernel on the diagonal
function double_layer_matrix(γ, n̂, κ, ẇ)
  N = length(γ)
  K = Matrix{Float64}(undef, N, N)
  Threads.@threads for j = 1 : N
    @inbounds for i = 1 : N
      if i == j
        K[i, i] = -(1 / 2π) * κ[i] / 2 * ẇ[i] - 1 / 2
      else
        Δγ = γ[i] - γ[j]
        K[i, j] = (1 / 2π) * real(conj(Δγ) * n̂[j]) / abs2(Δγ) * ẇ[j]
      end
    end
  end
  return K
end

"""
    conformalmap(γ; L, α) -> f, df

Construct conformal map functions from the disk to the geometry parametrized by `γ`.

# Arguments
- `γ` : points parametrized on an equally spaced grid
- `L` : length of the parametrization interval
- `α` : oversampling factor

# Returns
- `f`, `df`
"""
function conformalmap(γ; L=1.0, α=5)

  γ = vec(γ)
  γₛ = derivative_fourier(γ, L)
  γₛₛ = derivative_fourier(γ, L, 2)

  if !(α ≈ 1)
    γ  = upsample(γ, α)
    γₛ = upsample(γₛ, α)
    γₛₛ = upsample(γₛₛ, α)
  end

  N = length(γ)

  v = abs.(γₛ)
  κ = imag.(conj.(γₛ) .* γₛₛ) ./ v .^ 3
  n̂ = -1im * γₛ ./ v

  w = ones(N) / N
  ẇ = w .* abs.(γₛ)
  b = -log.(abs.(γ))

  K = double_layer_matrix(γ, n̂, κ, ẇ)
  ϕ  = lu!(K) \ b

  h = generate_holomorphic_data(ϕ, 1.0, γ, w, γₛ)
  hₛ = derivative_fourier(h, 1.0)
  γ̃  = γ .* exp.(h)
  γ̃ₛ = γₛ .* exp.(h) + hₛ .* γ̃

  # Fresh bindings: γ is reassigned above, and closures capturing it would box it
  f, df = let γ = γ, dγ = γₛ ./ γ̃ₛ, γ̃ = γ̃, γ̃ₛ = γ̃ₛ, w = w
    (ζ -> cauchy_operator(ζ, γ, γ̃, γ̃ₛ, w)), (ζ -> cauchy_operator(ζ, dγ, γ̃, γ̃ₛ, w))
  end

  return f, df

end
