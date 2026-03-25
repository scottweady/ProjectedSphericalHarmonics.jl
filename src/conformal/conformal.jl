
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

  Δz  = γ .- ζ
  idx = findfirst(x -> abs(x) ≈ 0, Δz)
  isnothing(idx) || return f[idx]

  q = w .* γₛ ./ Δz
  return sum(f .* q) / sum(q)

end

function generate_holomorphic_data(τ, L, γ, w, γₛ)

  τₛ = derivative_fourier(τ, L)
  Δτ = τ .- transpose(τ)
  Δγ = γ .- transpose(γ)
  A  = Δτ ./ (Δγ .+ (Δγ .== 0)) .* transpose(w .* γₛ)
  A[diagind(A)] .= τₛ .* w
  return -τ .- (1 / (2π * 1im)) .* vec(sum(A, dims=2))

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

  Δγ = γ .- transpose(γ)
  K  = (1 / 2π) .* real.(conj.(Δγ) .* transpose(n̂)) ./ (abs2.(Δγ) .+ (Δγ .== 0)) .* transpose(ẇ)
  K[diagind(K)] .= -(1 / 2π) .* κ ./ 2 .* ẇ
  ϕ  = (-I/2 + K) \ b

  h = generate_holomorphic_data(ϕ, 1.0, γ, w, γₛ)
  hₛ = derivative_fourier(h, 1.0)
  γ̃  = γ .* exp.(h)
  γ̃ₛ = γₛ .* exp.(h) + hₛ .* γ̃

  f(ζ) = cauchy_operator(ζ, γ, γ̃, γ̃ₛ, w)
  df(ζ) = cauchy_operator(ζ, γₛ ./ γ̃ₛ, γ̃, γ̃ₛ, w)

  return f, df

end
