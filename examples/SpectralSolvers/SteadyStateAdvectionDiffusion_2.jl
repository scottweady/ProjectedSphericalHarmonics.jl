using ProjectedSphericalHarmonics
using FFTW
using CairoMakie
using LinearAlgebra
using LinearOperators
using Krylov


# Solver for the steady-state advection-diffusion equation on the disk with Dirichlet BC.
# The PDE in complex form:
#   4 ∂ζ∂ζ̄ u + 2 (a ∂ζ u + ā ∂ζ̄ u) = f    on D
#   u = g    on ∂D
#
# where a ∈ ℂ is the (constant) complex advection velocity.
#
# Decomposition: u = Δ⁻¹μ + u_h
# where Δ⁻¹μ has zero Dirichlet BC and u_h is harmonic with u_h|_{∂D} = g - Δ⁻¹μ|_{∂D}.
#
# Substituting (Δ(Δ⁻¹μ) = μ, Δu_h = 0):
#   μ + 2a ∂ζΔ⁻¹μ + 2ā ∂ζ̄Δ⁻¹μ + 2(a ∂ζ + ā ∂ζ̄) u_h = f
#
# Since u_h depends linearly on μ via the BC, eliminating u_h yields:
#
#   [I + 2a ∂ζΔ⁻¹ + 2ā ∂ζ̄Δ⁻¹ - 2(a ∂ζ + ā ∂ζ̄)∘SolveHarmonic∘trace∘Δ⁻¹] μ
#       = f - 2(a ∂ζ + ā ∂ζ̄) SolveHarmonic(g)
#
# The GMRES unknown μ is a TriangularCoeffArray (PSH coefficient space), NOT a
# nodal grid of size Nr × Nθ. The per-frequency operators are applied directly,
# exploiting the frequency-shift rules:
#   ∂Ĝᵐ∂ζ  : input at freq m → output at freq m-1
#   ∂Ĝᵐ∂ζ̄ : input at freq m → output at freq m+1
# To fill column m of the result:
#   ∂ζΔ⁻¹ contribution: feed mode_coefficients(μ, m+1)  [shifts m+1 → m]
#   ∂ζ̄Δ⁻¹ contribution: feed mode_coefficients(μ, m-1) [shifts m-1 → m]
# The harmonic correction only populates mode_coefficients(result, m)[1] (l = |m| mode).


# ─── Helper: reshape flat vector → TriangularCoeffArray ──────────────────────

function _flat_to_tri(v::AbstractVector, template::TriangularCoeffArray)
    result = similar(template)
    offset = 0
    for col in result.data
        n    = length(col)
        col .= v[offset+1:offset+n]
        offset += n
    end
    return result
end


# ─── Coefficient-space linear operator ───────────────────────────────────────

"""
    _apply_A_coeff(μ_tri, a, D)

Apply the advection-diffusion operator in PSH coefficient space:
    Aμ = μ + 2a ∂ζΔ⁻¹μ + 2ā ∂ζ̄Δ⁻¹μ
           - 2(a ∂ζ + ā ∂ζ̄) SolveHarmonic(trace(Δ⁻¹μ))

Terms 2–3 are assembled per-frequency using the sparse operators.
Term 4 is purely harmonic and contributes only to mode_coefficients(result, m)[1].

# Arguments
- `μ_tri` : source density as a `TriangularCoeffArray`
- `a`     : complex advection velocity
- `D`     : disk discretization

# Returns
- `TriangularCoeffArray` of the same structure as `μ_tri`
"""
function _apply_A_coeff(μ_tri::TriangularCoeffArray, a, D)
    lmax  = D.Mr
    Mθ    = D.Mθ
    Mspan = μ_tri.Mspan

    result = TriangularCoeffArray{Float64}(lmax, Mspan)

    u = DiskFunction(μ_tri, Dl; derivatives = ((1,0), (0,1)))


    # Harmonic correction: compute û_h from trace(Δ⁻¹μ), then subtract
    # 2(a ∂ζ + ā ∂ζ̄) u_h from the l = |m| mode of each column.
    N_m       = length(Mspan)
    trace_hat = zeros(ComplexF64, N_m)
    for (i, m) in enumerate(Mspan)
        trace_hat[i] = traceĜ(mode_coefficients(μ_tri, m), lmax, m)
    end

    û_h  = [trace_hat[i] / ylm(abs(Mspan[i]), Mspan[i], 1.0) for i in 1:N_m]
    dû_ζ  = zeros(ComplexF64, N_m)
    dû_ζ̄  = zeros(ComplexF64, N_m)
    ∂ζ_HarmonicFunction!(dû_ζ,  û_h, Mspan)
    ∂ζ̄_HarmonicFunction!(dû_ζ̄, û_h, Mspan)

    for (i, m) in enumerate(Mspan)
        mode_coefficients(result, m)[1] -= 2*(a * dû_ζ[i] + conj(a) * dû_ζ̄[i])
    end

    return result
end
