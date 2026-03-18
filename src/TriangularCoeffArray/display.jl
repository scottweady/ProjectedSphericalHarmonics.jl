# display.jl
# Pretty-printing for TriangularCoeffArray: compact one-line show and the
# full text/plain table view with terminal-aware truncation of rows and columns.

using Printf

# ─── Display ─────────────────────────────────────────────────────────────────

"""Infer the maximum radial degree from the stored column lengths, Mspan, and parity."""
function _infer_lmax(A::TriangularCoeffArray{T,N,P}) where {T,N,P}
    isempty(A.data) && return 0
    if P == :even
        return maximum(abs(m) + 2*(length(A.data[i]) - 1) for (i, m) in enumerate(A.Mspan))
    else  # :odd
        return maximum(abs(m) + 1 + 2*(length(A.data[i]) - 1) for (i, m) in enumerate(A.Mspan))
    end
end

"""Format a complex number in scientific notation with 4 decimal places."""
function _fmt_complex(z::Complex)
    r = @sprintf("%.3e", real(z))
    i = @sprintf("%.3e", imag(z))
    return "$(r)$(i)im"
    if iszero(imag(z))
        return r
    elseif imag(abs(z) ) > 0
        return "$(r)+$(i)im"
    else
        return "$(r)$(i)im"
    end
end

function Base.show(io::IO, A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    lmax = _infer_lmax(A)
    Mθ   = length(A.Mspan) ÷ 2
    print(io, "TriangularCoeffArray{$T}(lmax=$lmax, Mθ=$Mθ, parity=$P, ordering=$O, $(length(A)) coefficients)")
end

function Base.show(io::IO, ::MIME"text/plain", A::TriangularCoeffArray{T,N,P,O}) where {T,N,P,O}
    lmax  = _infer_lmax(A)
    nM    = length(A.Mspan)
    Mθ    = nM ÷ 2
    cw    = 16        # column width per (l, m) entry
    gutter = 6        # width of "l=XX │"

    term_rows, term_cols = displaysize(io)

    # ── Column range: symmetric around m=0, always shown ─────────────────────
    n_cols_avail = (term_cols - gutter) ÷ cw
    if 2Mθ + 1 ≤ n_cols_avail
        k_max         = Mθ
        show_col_dots = false
    else
        # Reserve one column width for the "⋯" indicator
        k_max         = max(0, (n_cols_avail - 2) ÷ 2)
        show_col_dots = (2k_max + 1) < n_cols_avail   # room for ⋯ after data cols
    end

    # ── Row range: from l=0, always shown ────────────────────────────────────
    n_rows_avail = term_rows - 4   # title + blank + header + separator
    if lmax + 1 ≤ n_rows_avail
        l_max_shown   = lmax
        show_row_dots = false
    else
        # Reserve one row for the "⋮" indicator
        l_max_shown   = max(0, n_rows_avail - 2)
        show_row_dots = true
    end

    println(io, "TriangularCoeffArray{$T} — lmax=$lmax, Mθ=$Mθ, parity=$P, ordering=$O, $(length(A)) coefficients")
    println(io)

    # Header row
    print(io, " "^gutter)
    for m in -k_max:k_max
        print(io, lpad("m=$m", cw))
    end
    show_col_dots && print(io, lpad("⋯", cw))
    println(io)

    # Separator
    n_sep = (2k_max + 1) + (show_col_dots ? 1 : 0)
    println(io, " "^gutter * "─"^(cw * n_sep))

    # Parity check: even → l+m even; odd → l+m odd
    _has_mode(l, m) = P == :even ? (abs(m) ≤ l && iseven(l + m)) :
                                   (abs(m) < l  && isodd(l + m))

    # Row index within the column for (l, m)
    _row_idx(l, m)  = P == :even ? (l - abs(m)) ÷ 2 + 1 :
                                   (l - abs(m) - 1) ÷ 2 + 1

    # Data rows
    for l in 0:l_max_shown
        print(io, "l=$(lpad(string(l), 2)) │")
        for m in -k_max:k_max
            if _has_mode(l, m) && m ∈ A.Mspan
                row = _row_idx(l, m)
                s   = _fmt_complex(mode_coefficients(A, m)[row])
                print(io, lpad(s, cw))
            else
                print(io, lpad("·", cw))
            end
        end
        show_col_dots && print(io, lpad("⋯", cw))
        println(io)
    end

    # Row truncation indicator
    if show_row_dots
        print(io, " "^gutter)
        for _ in -k_max:k_max
            print(io, lpad("⋮", cw))
        end
        show_col_dots && print(io, lpad("⋱", cw))
        println(io)
    end
end
