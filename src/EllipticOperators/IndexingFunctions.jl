export size_current_m, ∂ζ_indexing_sparse, ∂ζ̄_indexing_sparse



@inline function size_current_m(lmax, m; aliasing = false)

    #This function is only valid for even expansions
    return floor(Int, (lmax + 2 - abs(m)) / 2) + aliasing

end



function ∂ζ_indexing_sparse(lmax, m; aliasing = false)

    #if m = 0, we go to m = -1, so we need to check the parity of lmax
    size_current_m(lmax, m-1; aliasing = aliasing)

end


function ∂ζ̄_indexing_sparse(lmax, m; aliasing = false)

    #if m = 0, we go to m = -1, so we need to check the parity of lmax
    size_current_m(lmax, m+1; aliasing = aliasing)

end

function ∂ζ∂ζ_indexing_sparse(lmax, m; aliasing = false)

    #if m = 0, we go to m = -1, so we need to check the parity of lmax
    size_current_m(lmax, m-2; aliasing = aliasing)

end

function ∂ζ̄∂ζ̄_indexing_sparse(lmax, m; aliasing = false)

    #if m = 0, we go to m = -1, so we need to check the parity of lmax
    size_current_m(lmax, m+2; aliasing = aliasing)

end