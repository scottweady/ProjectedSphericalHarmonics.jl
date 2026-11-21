using Random
using LinearAlgebra: I

println("Testing coefficient resizing between disks...")

Random.seed!(2)

# Band-limited test functions: polynomials of degree 4 in x, y (even PSH), and w times them (odd PSH)
poly(z) = 1 + 2z - 0.5im * conj(z)^2 + z^2 * conj(z) + 0.25 * z * conj(z)^3
band = Dict(:even => z -> poly(z), :odd => z -> sqrt(1 - abs2(z)) * poly(z))

# Pairs of disks with Mℓ ≠ Mₘ, refining Mℓ and Mₘ together and independently, and one pair where Mℓ
# shrinks while Mₘ grows (`psh` needs Mₘ ≤ Mℓ)
pairs = [(disk(7, 4), disk(14, 8)), (disk(6, 5), disk(11, 5)), (disk(9, 5), disk(9, 9)), (disk(8), disk(16)),
         (disk(10, 4), disk(7, 6))]

for (D, D′) in pairs, parity in (:even, :odd)

  local tag = "($(D.Mℓ),$(D.Mₘ)) → ($(D′.Mℓ),$(D′.Mₘ)), $parity"
  local mask, mask′ = vec(getproperty(D, parity)), vec(getproperty(D′, parity))

  # Round trip D → D′ → D keeps exactly the modes present on both (all of them when D′ is finer)
  û = randn(ComplexF64, D.shp) .* getproperty(D, parity)
  common = (D.Lspan .<= D′.Mℓ) .& (abs.(D.Mspan) .<= D′.Mₘ)
  check("  Round trip error $tag: ", maximum(abs, resize_coefficients(resize_coefficients(û, D, D′; parity), D′, D; parity) - û .* common), 0.0)

  # Embedded coefficients of a band-limited function equal its coefficients on the finer disk
  f = band[parity]
  local err = maximum(abs, resize_coefficients(psh(f.(D.z), D; parity), D, D′; parity) - psh(f.(D′.z), D′; parity))
  check("  Embedding vs psh on finer grid $tag: ", err, 1e-12)

  # Embedded coefficients of random modes evaluate to the same function on the grid of D′
  u′ = sum(û[i] * ylm(index2lm(i, D)..., D′.z) for i in findall(mask .& vec(common)))
  local err = maximum(abs, ipsh(resize_coefficients(û, D, D′; parity), D′; parity) - u′)
  check("  Evaluation on finer grid $tag: ", err, 1e-11)

  # Matrix form agrees with the array form, in both directions, and is the transpose of the reverse map
  E, Eᵀ = embedding_matrix(D, D′; parity), embedding_matrix(D′, D; parity)
  @test size(E) == (count(mask′), count(mask))
  @test E * vec(û)[mask] == vec(resize_coefficients(û, D, D′; parity))[mask′]
  v̂ = randn(ComplexF64, D′.shp) .* getproperty(D′, parity)
  @test Eᵀ * vec(v̂)[mask′] == vec(resize_coefficients(v̂, D′, D; parity))[mask]
  @test Eᵀ == transpose(E)
  all(common) && @test Eᵀ * E == I

  # ipsh_matrix on the finer disk composed with the embedding evaluates coarse coefficients there
  Q′ = ipsh_matrix(D′; parity)[:, findall(mask′)]
  local err = maximum(abs, Q′ * E * vec(û)[mask] - vec(u′))
  check("  ipsh_matrix ∘ embedding_matrix $tag: ", err, 1e-11)
end

# Flattened and tuple inputs, and the idx_even layout used downstream
let D = disk(6, 4), D′ = disk(12, 8)
  û = randn(ComplexF64, D.shp) .* D.even
  @test resize_coefficients(vec(û), D, D′) == vec(resize_coefficients(û, D, D′))
  @test resize_coefficients((û, 2û), D, D′) == (resize_coefficients(û, D, D′), resize_coefficients(2û, D, D′))
  @test embedding_matrix(D, D′) * û[D.idx_even] == resize_coefficients(û, D, D′)[D′.idx_even]
  @test_throws DimensionMismatch resize_coefficients(zeros(3, 3), D, D′)
end
