
using LinearAlgebra

"""
    Domain

Discretization of a non-circular domain, the image of the unit disk under a map ``z = f(ζ)``:
returned by `ellipse(a, b, Mℓ, Mₘ)` (an affine map) and `domain(γ, Mℓ, Mₘ)` (a conformal map).
It wraps the `Disk` of the reference grid (its fields, such as `even` and `w`, are reachable from
the `Domain`) and adds the mapped points `z`, the area weights `dz` (with the Jacobian of the map),
and the operators of the domain as matrices in coefficient space. Functions on a `Domain` are
sampled at `Ω.z`; the weight `Ω.w` is the disk's, ``\\sqrt{1 - |ζ|^2}``, which vanishes on the
boundary like the square root of the distance to it.
"""
struct Domain
  D
  f
  df
  z
  dz
  K_S
  K_N
  K_G
  K̂_S
  K̂_N
  K̂_G
  ext       # EllipseGeometry for close evaluation, or nothing
  factors   # LU factorizations of the operators, made by `solve` on first use
end

Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G, ext) =
  Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G, ext, (IdDict{Any, Any}(), ReentrantLock()))

# Factorization of the operator `M` (a matrix, or a 2×2 block of matrices assembled into one), made
# once per domain. The cache is keyed by the identity of `M`, so `M` must not be modified in place
function factorization(Ω::Domain, M)
  cache, lk = getfield(Ω, :factors)
  lock(lk) do
    get!(cache, M) do
      lu(M isa AbstractMatrix{<:AbstractMatrix} ? [M[1, 1] M[1, 2]; M[2, 1] M[2, 2]] : M)
    end
  end
end

"""
    domain(γ, Mℓ, Mₘ)

Discretization of the image of the unit disk under a conformal map.

# Arguments
- `γ` : boundary of the domain
- `Mℓ` : maximum radial degree
- `Mₘ` : maximum azimuthal order

# Returns
- Domain containing discretization information

!!! warning "Experimental"
    Conformally mapped domains are experimental: their accuracy and interface may change. The
    curve must enclose the origin (this is not checked), convergence slows as the shape gets more
    elongated, the operators are dense (moderate `M` only), and suspensions of such bodies have no
    close evaluation. For ellipses, use `ellipse`.
"""
function domain(γ, Mℓ::Int, Mₘ::Int)

  f, df = conformalmap(γ)
  D = disk(Mℓ, Mₘ)
  ζ, dζ = D.ζ, D.dζ

  z, z′ = similar(ζ, ComplexF64), similar(ζ, ComplexF64)
  Threads.@threads for k in eachindex(ζ)
    z[k], z′[k] = f(ζ[k]), df(ζ[k])
  end
  dz = abs2.(z′) .* dζ

  ζvec, zvec, z′vec = vec(ζ), vec(z), vec(z′)
  N = length(ζvec)

  # The divided difference (zᵢ - zⱼ)/(ζᵢ - ζⱼ) = ρ e^{iα} (z′ on the diagonal) relates distances and
  # directions on the domain to those on the disk. With the Jacobian Jⱼ = |z′ⱼ|² of the source point,
  # the kernels on the grid are H = J/ρ (for 1/r) and J/ρ³ (for 1/r³), and (c, s) = (cos α, sin α)
  # rotates the direction of the Stokeslet
  H, H3, c, s = (Matrix{Float64}(undef, N, N) for _ = 1 : 4)
  Threads.@threads for j = 1 : N
    Jⱼ = abs2(z′vec[j])
    for i = 1 : N
      m = i == j ? z′vec[j] : (zvec[i] - zvec[j]) / (ζvec[i] - ζvec[j])
      ρ = abs(m)
      H[i, j], H3[i, j] = Jⱼ / ρ, Jⱼ / ρ^3
      c[i, j], s[i, j] = real(m) / ρ, imag(m) / ρ
    end
  end

  even, odd = findall(vec(D.even)), findall(vec(D.odd))

  # Grid operators Q Ŝ P of disk operators Ŝ in coefficient space, built from the factors of the
  # transforms (see `_grid_operator`)
  Y, _ = _psh_factors(D)
  dw = vec(D.dw)
  K_S = _grid_operator(sparse(Diagonal(D.K̂_S[even])), even, Y, dw) .* H
  K_N = _grid_operator(sparse(Diagonal(D.K̂_N[odd])), odd, Y, dw) .* H3

  # Disk single layers weighted by the products of the components of the direction on the disk
  Ŝcc = laplace3d_angular_matrix(ϕ -> cos(ϕ)^2, D.Lspan, D.Mspan, even)
  Ŝsc = laplace3d_angular_matrix(ϕ -> cos(ϕ) * sin(ϕ), D.Lspan, D.Mspan, even)
  Ŝss = laplace3d_angular_matrix(ϕ -> sin(ϕ)^2, D.Lspan, D.Mspan, even)

  K_cc = _grid_operator(Ŝcc, even, Y, dw)
  K_sc = _grid_operator(Ŝsc, even, Y, dw)
  K_ss = _grid_operator(Ŝss, even, Y, dw)

  # Rotate the direction by α and weight by H
  K_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K_G[1, 1], K_G[1, 2], K_G[2, 2] = (similar(K_S) for _ = 1 : 3)
  K_G[2, 1] = K_G[1, 2]
  Threads.@threads for j = 1 : N
    for i = 1 : N
      cα, sα, h = c[i, j], s[i, j], H[i, j]
      cc, sc, ss = K_cc[i, j], K_sc[i, j], K_ss[i, j]
      K_G[1, 1][i, j] = K_S[i, j] + h * (cc * cα^2 - 2 * sc * cα * sα + ss * sα^2)
      K_G[1, 2][i, j] = h * (sc * (cα^2 - sα^2) + (cc - ss) * cα * sα)
      K_G[2, 2][i, j] = K_S[i, j] + h * (cc * sα^2 + 2 * sc * cα * sα + ss * cα^2)
    end
  end

  # Back to coefficient space: P K Q restricted to the modes of each parity
  K̂_S = _coefficient_operator(K_S, even, Y, dw)
  K̂_N = _coefficient_operator(K_N, odd, Y, dw)

  K̂_G = Matrix{Matrix{ComplexF64}}(undef, 2, 2)
  K̂_G[1, 1] = _coefficient_operator(K_G[1, 1], even, Y, dw)
  K̂_G[1, 2] = _coefficient_operator(K_G[1, 2], even, Y, dw)
  K̂_G[2, 1] = K̂_G[1, 2]
  K̂_G[2, 2] = _coefficient_operator(K_G[2, 2], even, Y, dw)

  return Domain(D, f, df, z, dz, K_S, K_N, K_G, K̂_S, K̂_N, K̂_G, nothing)
  
end

domain(γ, M::Int) = domain(γ, M, M)

# The PSH transform is an FFT in θ followed by a radial projection for each azimuthal order:
# P = blockdiag(Rₘ) (F ⊗ I) and Q = (Fᴴ ⊗ I) blockdiag(Yₘ), with Yₘ[i, ℓ] = y_ℓ^m(rᵢ), Rₘ = Yₘᴴ diag(dw)
# and F[m, j] = e^{-imθⱼ} (see `_psh_factors`). The two functions below use this structure in place of
# the dense matrices P and Q.

# Grid operator Q[:, idx] Ŝ P[idx, :] of a sparse operator Ŝ on the coefficients `idx`. A block of Ŝ
# mapping order m to order m + κ gives e^{iκθⱼ} times a block circulant in θ:
#   (Q Ŝ P)[(i, j), (i′, j′)] = Σ_κ e^{iκθⱼ} C_κ[i, i′, j - j′],   C_κ[:, :, Δ] = Σ_m e^{imθ_Δ} Y_{m+κ} Ŝ_{m+κ,m} Rₘ
function _grid_operator(Ŝ::SparseMatrixCSC, idx, Y, dw)

  Nr, _, Nθ = size(Y)
  N = Nr * Nθ

  # Blocks Ŝ_{m+κ,m} of each shift κ (mod Nθ), indexed by the order m of the input
  blocks = Dict{Int, Array{ComplexF64, 3}}()
  rows, vals = rowvals(Ŝ), nonzeros(Ŝ)
  for q = 1 : size(Ŝ, 2)
    l, nm = mod1(idx[q], Nr), cld(idx[q], Nr)
    for k = nzrange(Ŝ, q)
      l′, nm′ = mod1(idx[rows[k]], Nr), cld(idx[rows[k]], Nr)
      S = get!(() -> zeros(ComplexF64, Nr, Nr, Nθ), blocks, mod(nm′ - nm, Nθ))
      S[l′, l, nm] += vals[k]
    end
  end

  # C[:, :, Δ + 1] = Σ_m e^{2πi m Δ/Nθ} Y_{m+κ} S_m R_m for each shift κ, and the phases e^{iκθⱼ}
  κs = collect(keys(blocks))
  Cs = map(κs) do κ
    S = blocks[κ]
    C = zeros(ComplexF64, Nr, Nr, Nθ)
    for nm = 1 : Nθ
      nm′ = mod1(nm + κ, Nθ)
      C[:, :, nm] .= (view(Y, :, :, nm′) * view(S, :, :, nm)) * (view(Y, :, :, nm)' .* transpose(dw))
    end
    bfft!(C, 3)
  end
  E = [cis(2π * κ * (j - 1) / Nθ) for j = 1 : Nθ, κ in κs]

  # One pass over A, summing the shifts
  A = Matrix{ComplexF64}(undef, N, N)
  A4 = reshape(A, Nr, Nθ, Nr, Nθ)
  Threads.@threads for j′ = 1 : Nθ
    @inbounds for i′ = 1 : Nr, j = 1 : Nθ
      Δ = mod(j - j′, Nθ) + 1
      for i = 1 : Nr
        acc = zero(ComplexF64)
        for k in eachindex(Cs)
          acc += E[j, k] * Cs[k][i, i′, Δ]
        end
        A4[i, j, i′, j′] = acc
      end
    end
  end

  return A

end

# Coefficient operator (P K Q)[idx, idx] of a grid operator K. Each column of K, as an (Nr, Nθ) array,
# is transformed by the FFT in θ and projected radially onto the modes in `idx`; then each row of the
# result likewise by the inverse FFT and the radial basis. The columns are threaded, each with its own
# buffer, executing one FFTW plan (execution is thread-safe; planning is not)
function _coefficient_operator(K, idx, Y, dw)

  Nr, _, Nθ = size(Y)
  N = Nr * Nθ
  n = length(idx)
  nms = cld.(idx, Nr)
  Yc = similar(Y, Nr, n)                  # Yc[:, p] = y_ℓ^m(r) of the mode idx[p]
  for p = 1 : n
    Yc[:, p] .= view(Y, :, mod1(idx[p], Nr), nms[p])
  end
  Rc = Yc .* dw                           # conj(Rc[:, p]) is its radial projection

  T = _transform_columns(K, Rc, nms, Nr, Nθ, true)            # T[p, c] = (P K)[idx[p], c]
  return transpose(_transform_columns(transpose(T), Yc, nms, Nr, Nθ, false))

end

# Out[p, c] = Σᵢ conj(B[i, p]) û[i, nms[p]] (forward = true) or Σᵢ B[i, p] û[i, nms[p]], where û is
# the FFT (forward) or unnormalized inverse FFT in θ of column c of A, reshaped to (Nr, Nθ)
function _transform_columns(A, B, nms, Nr, Nθ, forward)
  n, Nc = size(B, 2), size(A, 2)
  Out = Matrix{ComplexF64}(undef, n, Nc)
  buf0 = zeros(ComplexF64, Nr, Nθ)
  flags = FFTW.ESTIMATE | FFTW.UNALIGNED
  plan = forward ? plan_fft!(buf0, 2; flags=flags) : plan_bfft!(buf0, 2; flags=flags)
  chunks = Iterators.partition(1 : Nc, cld(Nc, 4Threads.nthreads()))
  Threads.@threads for cs in collect(chunks)
    buf = similar(buf0)
    for c in cs
      copyto!(buf, view(A, :, c))
      plan * buf
      @inbounds for p = 1 : n
        nm, acc = nms[p], zero(ComplexF64)
        @simd for i = 1 : Nr
          acc += (forward ? conj(B[i, p]) : B[i, p]) * buf[i, nm]
        end
        Out[p, c] = acc
      end
    end
  end
  return Out
end

const _domain_fields = fieldnames(Domain)
Base.getproperty(Ω::Domain, s::Symbol) =
  s ∈ _domain_fields ? getfield(Ω, s) : getproperty(getfield(Ω, :D), s)