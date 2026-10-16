
using SparseArrays
using FFTW

"""
    laplace3d_SN_inverse_matrix(Lspan, Mspan)

Sparse matrix representation of the inverse of SN⁻¹ in the even PSH basis.

# Arguments
- `Lspan` : vector of radial mode numbers
- `Mspan` : vector of azimuthal mode numbers
- `idx` : vector of linear indices corresponding to even modes

# Returns
- sparse matrix representation of SN⁻¹ in the even PSH basis
"""
function laplace3d_SN⁻¹_matrix(Lspan, Mspan, idx)

    Nl, Nm = length(Lspan), length(Mspan)
    L, M = Lspan, abs.(Mspan)

    c_diag = -(L .== M) ./ ((2L .+ 1) .* (2L .+ 3)) .- (L .> M) .* 2 ./ ((2L .- 1) .* (2L .+ 3))
    c_up   = Nlm.(L, M, L .+ 2, M) .* (-(L .- M .+ 1) ./ ((2L .+ 1) .* (2L .+ 3) .* (L .+ M .+ 2)))
    c_down = Nlm.(L, M, L .- 2, M) .* (-(L .+ M) ./ ((2L .+ 1) .* (2L .- 1) .* max.(L .- M .- 1, 1)))

    LI = LinearIndices((Nl, Nm))
    Is, Js, Vs = Int[], Int[], Float64[]

    for nm in 1:Nm, nl in 1:Nl

        push!(Is, LI[nl, nm])
        push!(Js, LI[nl, nm])
        push!(Vs, c_diag[nl, nm])

        if nl + 2 <= Nl
          push!(Is, LI[nl+2, nm]); 
          push!(Js, LI[nl, nm]); 
          push!(Vs, c_up[nl, nm])
        end

        if nl >= 3
          push!(Is, LI[nl-2, nm]); 
          push!(Js, LI[nl, nm]); 
          push!(Vs, c_down[nl, nm])
        end

    end

    M = sparse(Is, Js, Vs, Nl * Nm, Nl * Nm)
    return M[idx, idx]

end

"""
    fourier_coeffs(g; N0=64, tol=1e-15, Nmax=2^16)

Fourier coefficients ĝ_k = (1/N) ∑_j g(t_j) e^{-ikt_j} of a periodic function g, in FFT
order (k ≥ 0 at k+1, k < 0 at k+N+1). The number of points N starts at N0 and is doubled
until the coefficients in the upper half of the band (N/4 ≤ |k| ≤ N/2) fall below
`tol * max|ĝ|`, so that the retained modes are free of aliasing.

# Arguments
- `g`    : function `t -> scalar` (or a tuple of such functions, sharing one N)
- `N0`   : initial number of points (power of 2)
- `tol`  : relative tolerance on the tail
- `Nmax` : maximum number of points

# Returns
- vector of Fourier coefficients (or a tuple of them)
"""
function fourier_coeffs(g::Tuple; N0=64, tol=1e-15, Nmax=2^16)

  N = N0
  while true
    ts = 2π .* (0:N-1) ./ N
    ĝ  = map(gi -> fft(gi.(ts)) ./ N, g)
    tail  = maximum(gk -> maximum(abs, view(gk, N÷4+1:3N÷4+1)), ĝ)
    scale = maximum(gk -> maximum(abs, gk), ĝ)
    (tail <= tol * scale || N >= Nmax) && return ĝ
    N *= 2
  end

end

fourier_coeffs(g; kwargs...) = fourier_coeffs((g,); kwargs...)[1]

"""
    laplace3d_angular_matrix(g, Lspan, Mspan, idx; Nθ=nothing, tol=1e-14)

Sparse PSH matrix for the operator defined by a scalar function g(t).

Computes the Fourier series g(t) = ∑_k ĝ_k e^{ikt} and assembles the matrix
whose (l,m) → (l,m-k) entries are ĝ_k * Clmn(l, m-k, k), coupling each PSH mode
to modes shifted in azimuthal order by each nonzero Fourier mode of g.

# Arguments
- `g`     : function `t -> scalar` (real or complex valued), or its Fourier
            coefficients in FFT order (e.g. from `fourier_coeffs`)
- `Lspan` : vector of radial mode numbers
- `Mspan` : vector of azimuthal mode numbers (ordered 0, 1, …, Mₘ, -Mₘ, …, -1)
- `idx`   : linear indices of the even PSH modes to retain
- `Nθ`    : Fourier quadrature points (default: chosen by the decay of ĝ, at least the
            next power of 2 ≥ 4Mₘ+4)
- `tol`   : cutoff for negligible Fourier coefficients

# Returns
- sparse matrix restricted to even modes `idx`
"""
function laplace3d_angular_matrix(g, Lspan, Mspan, idx; Nθ=nothing, tol=1e-14)

  Mₘ = length(Mspan) ÷ 2

  # Fourier coefficients: ĝ_k = (1/Nθ) ∑_j g(t_j) e^{-ikt_j}
  if isnothing(Nθ)
    ĝ = fourier_coeffs(g; N0=nextpow(2, 4Mₘ + 4), tol=min(tol, 1e-15))
  else
    ts = 2π .* (0:Nθ-1) ./ Nθ
    ĝ  = fft(g.(ts)) ./ Nθ
  end

  return laplace3d_angular_matrix(ĝ, Lspan, Mspan, idx; tol=tol)

end

function laplace3d_angular_matrix(ĝ::AbstractVector, Lspan, Mspan, idx; tol=1e-14)

  Nl, Nm = length(Lspan), length(Mspan)
  N  = Nl * Nm
  Mₘ = Nm ÷ 2
  Nθ = length(ĝ)

  # FFTW index for mode k: k ≥ 0 → k+1, k < 0 → k+Nθ+1
  k_to_fft_idx(k) = k >= 0 ? k + 1 : k + Nθ + 1

  # Mspan is ordered 0, 1, ..., Mₘ, -Mₘ, ..., -1
  m_to_nm(m) = m >= 0 ? m + 1 : m + Nm + 1

  LI = LinearIndices((Nl, Nm))
  Is, Js, Vs = Int[], Int[], ComplexF64[]

  for k in vcat(0:Nθ÷2-1, -(Nθ÷2):-1)

    abs(k) > 2Mₘ && continue    # no pair of retained orders differs by k
    ĝk = ĝ[k_to_fft_idx(k)]
    abs(ĝk) < tol && continue

    Ck = Clmn.(Lspan, Mspan, k)

    for nm in 1:Nm
      m_src = Mspan[nm] - k
      abs(m_src) > Mₘ && continue

      nm_src = m_to_nm(m_src)

      for nl in 1:Nl
        Ck_val = Ck[nl, nm_src]
        iszero(Ck_val) && continue

        push!(Is, LI[nl, nm])
        push!(Js, LI[nl, nm_src])
        push!(Vs, ĝk * Ck_val)
      end
    end

  end

  return sparse(Is, Js, Vs, N, N)[idx, idx]

end


"""
    stokes3d_single_layer_matrix(D::Disk)

Sparse matrix representation of the 3D Stokes single layer operator in the even PSH basis.

# Arguments
- `Lspan` : vector of radial mode numbers
- `Mspan` : vector of azimuthal mode numbers
- `idx` : vector of linear indices corresponding to even modes

# Returns
- sparse matrix representation of the 3D Stokes single layer operator in the even PSH basis
"""
function stokes3d_single_layer_matrix(Lspan, Mspan, idx)

  g11(θ) = 1.0 + cos(θ)^2
  g12(θ) = sin(θ) * cos(θ)
  g22(θ) = 1.0 + sin(θ)^2

  Ĝ = Matrix{SparseMatrixCSC{ComplexF64, Int}}(undef, 2, 2)
  Ĝ[1, 1] = laplace3d_angular_matrix(g11, Lspan, Mspan, idx)
  Ĝ[1, 2] = laplace3d_angular_matrix(g12, Lspan, Mspan, idx)
  Ĝ[2, 1] = Ĝ[1, 2]
  Ĝ[2, 2] = laplace3d_angular_matrix(g22, Lspan, Mspan, idx)

  return Ĝ

end