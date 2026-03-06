"""
    build_weight_operators(Lspan, Mspan, Nlm)

Build operators for multiplication and division by the weight √(1 - |ζ|²)
in the spherical-harmonic-on-disk coefficient space.

The weight acts as a tridiagonal coupling in `l` for each fixed `m`,
via the three-term recurrence for associated Legendre polynomials:

    √(1-r²) · ỹₗᵐ = cₗ₋₁ᵐ · ỹₗ₋₁ᵐ + cₗ₊₁ᵐ · ỹₗ₊₁ᵐ

Returns `Ŵ` (forward multiply) and `Ŵ⁻¹` (coefficient-space solve).
"""
function build_weight_operators(Lspan, Mspan)

    Nl = length(Lspan)
    Nm = length(Mspan)
    N  = Nl * Nm

    # ── Forward operator: multiplication by √(1 - |ζ|²) ──────────────

    W = _build_weight_matrix(Lspan, Mspan, Nl, Nm)

    function Ŵ(û)
        return reshape(W * vec(û), size(û))
    end

    # ── Inverse operator: division by √(1 - |ζ|²) ───────────────────
    #
    # W is Nl×Nl tridiagonal per m-block, but the top row (l = |m|)
    # couples to l = |m|-1 which doesn't exist, so there's one fewer
    # equation than unknown per block. We select the well-determined
    # rows (ib) and columns including one extra degree of freedom (jb),
    # then solve the rectangular system via QR.

    ib, jb = _select_invertible_subsystem(Lspan, Mspan, Nl)
    Wb = qr(W[ib, jb])

    function Ŵ⁻¹(f̂)
        û = zeros(ComplexF64, N)
        û[jb] = Wb \ vec(f̂)[ib]
        return reshape(û, size(f̂))
    end

    return Ŵ, Ŵ⁻¹
end


# ── Internal helpers ─────────────────────────────────────────────────

"""
Tridiagonal coupling coefficients for weight × ỹₗᵐ.
Returns (lower, upper) coefficients connecting l to l-1 and l+1.
"""
function _weight_coupling(l, m, Nlm)
    am = abs(m)
    lower = Nlm(l - 1, m, l, m) * ((l - 1) - am + 1) / (2(l - 1) + 1)
    upper = Nlm(l + 1, m, l, m) * ((l + 1) + am)     / (2(l + 1) + 1)
    return lower, upper
end

"""
Assemble the full sparse weight matrix (block-tridiagonal, one block per m).
"""
function _build_weight_matrix(Lspan, Mspan, Nl, Nm)

    Is, Js, Vs = Int[], Int[], ComplexF64[]

    for (nm, m) in enumerate(Mspan)
        offset = (nm - 1) * Nl

        for (nl, l) in enumerate(Lspan)
            abs(m) > l && continue

            row = nl + offset
            lower, upper = _weight_coupling(l, m, Nlm)

            if nl > 1
                push!(Is, row); push!(Js, row - 1); push!(Vs, lower)
            end
            if nl < Nl
                push!(Is, row); push!(Js, row + 1); push!(Vs, upper)
            end
        end
    end

    return sparse(Is, Js, Vs, Nl * Nm, Nl * Nm)
end

"""
Select row (ib) and column (jb) indices for the rectangular subsystem
used to invert the weight operator. Per m-block, we keep rows for
l = |m|...(Nl-2) and columns for l = |m|...(Nl-1), giving one extra
column that the QR solve handles as a least-norm solution.
"""
function _select_invertible_subsystem(Lspan, Mspan, Nl)

    ib = Int[]  # equation rows to keep
    jb = Int[]  # unknown columns to solve for

    for (nm, m) in enumerate(Mspan)
        offset = (nm - 1) * Nl
        l_start = abs(m) + 1  # first valid l index within this block

        # equations: all rows except the last in the valid range
        for nl in l_start:(Nl - 1)
            push!(ib, nl + offset)
            push!(jb, nl + offset)
        end

        # one extra column (the last valid l) to make the system rectangular
        push!(jb, Nl + offset)
    end

    return ib, jb
end