# ProjectedSphericalHarmonics' code map: which functions of src/ each feature runs. Every setting (a kind of domain,
# or an option: the basis and transforms, Laplace operators, volume and differential operators, Stokes, shape
# derivatives, suspensions) is one small run, changed from the base case (`disk(16)` and one Laplace solve
# `𝒮⁻¹(1, D)`) in that setting only, run under line coverage; the lines it executes are mapped back to the top-level
# function definitions of src/. Written as data for src/assets/codemap/codemap.js, which draws the docs page
# "Code map".
#   julia --project=docs docs/codemap.jl [covdir]          (from the package root; a few minutes)
# The script runs itself again in a child Julia with coverage on (no package image, so the child compiles
# ProjectedSphericalHarmonics), on one thread: the threaded loops run the same lines, and with several threads
# updating the same coverage counters the runs are much slower. The child's counters accumulate over the whole run:
# it writes them before (<setting>.pre.info) and after (<setting>.info) each setting, and the lines a setting ran are
# those whose count grew. (Clearing the counters between settings with jl_clear_coverage_data does not work on
# Julia 1.12: code compiled before the clear stops counting, so a function an earlier setting compiled would show as
# not run.) The tracefiles go to `covdir` if given, and a `covdir` that already holds them skips the run.
const ROOT = dirname(@__DIR__)                       # the package root
const SRC = joinpath(ROOT, "src")
const OUT = joinpath(@__DIR__, "src", "assets", "codemap", "codemap_data.js")

# ---- the data every setting starts from: built once, before the first setting ---------------------------------
# The three kinds of domain at M = 16 (a 17 × 33 grid): the unit disk, an ellipse with semi-axes 1.5 and 1, and a
# three-lobed domain mapped conformally from 128 samples of its boundary. The domain settings build theirs again
# inside their own run, so the map shows what that costs; the options use these, the one named by their `parent`
# bound to `Ω`.
const SETUP = quote
    using ProjectedSphericalHarmonics, LinearAlgebra
    const PSH = ProjectedSphericalHarmonics
    θs = range(0, 2π, 129)[1:end-1]
    γ = (1 .+ 0.15cos.(3θs)) .* cis.(θs)               # a three-lobed boundary curve, 128 samples
    D = disk(16)
    E = ellipse(1.5, 1.0, 16)
    G = domain(γ, 16)
    DOMAINS = Dict("disk" => D, "ellipse" => E, "domain" => G)
    shear(z) = (imag.(z), zero(real.(z)))               # ux = y
    xcm = [0.0 + 0.0im, 0.3 + 2.25im]                   # two bodies, about a quarter of a radius apart
    θcm = [0.0, 0.2]
    FT = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.1]]            # forces and torques [Fx, Fy, T]
end

# ---- the settings: (id, label, group, what it is, the run) ----------------------------------------------------
# `parent` is the domain an option is run with (bound to `Ω`) and compared with; "" for the domains. `schemes`: the
# domains the option applies to, each combination checked by running it.
const ALL = ["disk", "ellipse", "domain"]
const DISK = ["disk"]
const SETTINGS = [
    # the kinds of domain
    (id="disk", label="Disk", group="Domain", base=true, parent="", schemes=String[],
     about="disk(16): the PSH grid on the unit disk (17 × 33 nodes) and its coefficient-space operators, and one Laplace solve 𝒮⁻¹(1, D), the charged disk; the base case",
     run=:(let D = disk(16); 𝒮⁻¹(1, D) end)),
    (id="ellipse", label="Ellipse", group="Domain", base=false, parent="", schemes=String[],
     about="ellipse(1.5, 1.0, 16): the disk under an affine map, its single layer and Stokes operators from the Fourier coefficients of the angular kernels (and the EllipseGeometry for close evaluation), and one solve 𝒮⁻¹(1, Ω)",
     run=:(let Ω = ellipse(1.5, 1.0, 16); 𝒮⁻¹(1, Ω) end)),
    (id="domain", label="Conformal domain", group="Domain", base=false, parent="", schemes=String[],
     about="domain(γ, 16) from 128 samples of a three-lobed boundary: the conformal map onto the disk, the operators by singularity swapping (dense, from the FFT and radial factors of the transforms), and one solve 𝒮⁻¹(1, Ω)",
     run=:(let Ω = domain(γ, 16); 𝒮⁻¹(1, Ω) end)),
    # the basis and transforms
    (id="eigen", label="Eigenfunctions", group="Basis & transforms", base=false, parent="disk", schemes=DISK,
     about="ylm and ∂ylm∂ζ on the grid, the normalizations Nlm, the eigenvalues λlm, the coefficients Clmn, and lm2index and index2lm of a Disk",
     run=:(let ζ = Ω.ζ
               ylm(3, 1, ζ); ∂ylm∂ζ(3, 1, ζ); Nlm(3, 1); Nlm(3, 1, 2, 1); λlm(3, 1); Clmn(3, 1, 2)
               index2lm(lm2index(3, 1, Ω), Ω) end)),
    (id="transforms", label="psh and ipsh", group="Basis & transforms", base=false, parent="disk", schemes=ALL,
     about="psh and ipsh of even and odd parity, of a number and of a tuple, in place (psh!, ipsh!), and ipsh at given radii",
     run=:(let u = exp.(real.(Ω.z)), û = psh(u, Ω)
               ipsh(û, Ω); ipsh(psh(u, Ω; parity=:odd), Ω; parity=:odd); ipsh(û, Ω, [0.5, 1.0])
               ipsh(psh((u, u), Ω), Ω); psh(1.0, Ω); ipsh!(psh!(ComplexF64.(u), Ω), Ω) end)),
    (id="upsample", label="upsample", group="Basis & transforms", base=false, parent="disk", schemes=ALL,
     about="upsample(u, Ω, 24, 24): grid values on a finer grid through the PSH expansion",
     run=:(upsample(exp.(real.(Ω.z)), Ω, 24, 24))),
    (id="matrices", label="Transform matrices", group="Basis & transforms", base=false, parent="disk", schemes=ALL,
     about="psh_matrix and ipsh_matrix: the dense matrices of the transforms, filled from the radial basis and the Fourier factors",
     run=:(begin psh_matrix(Ω); ipsh_matrix(Ω) end)),
    (id="quadrature", label="trace and integral", group="Basis & transforms", base=false, parent="disk", schemes=ALL,
     about="trace (of a function and of a tuple) on the boundary, integral over the domain, and the grids legpts, trigpts and diskpts",
     run=:(let u = exp.(real.(Ω.z))
               trace(u, Ω); trace((u, u), Ω); integral(u, Ω); legpts(8); trigpts(8); PSH.diskpts(8, 9) end)),
    (id="aca", label="aca", group="Basis & transforms", base=false, parent="disk", schemes=ALL,
     about="aca: the adaptive cross approximation of a smooth kernel on the grid points, from the dense matrix and from the kernel function",
     run=:(let z = vec(Ω.z), K = (x, y) -> 1 / (1 + abs2(x - y))
               aca([K(x, y) for x in z, y in z]); aca(K, z) end)),
    # the Laplace operators
    (id="single", label="𝒮", group="Laplace", base=false, parent="ellipse", schemes=ALL,
     about="𝒮(u, Ω): the single layer of the 3D Laplacian applied to a density",
     run=:(𝒮(real.(Ω.z) .+ 1, Ω))),
    (id="hyper", label="𝒩", group="Laplace", base=false, parent="domain", schemes=["disk", "domain"],
     about="𝒩(u, Ω): the hypersingular operator, applied to a function vanishing on the boundary (the ellipse does not build it)",
     run=:(𝒩(real.(Ω.z) .* Ω.w, Ω))),
    (id="hyperinv", label="𝒩⁻¹", group="Laplace", base=false, parent="domain", schemes=["disk", "domain"],
     about="𝒩⁻¹(f, Ω): the inverse of the hypersingular operator (the ellipse does not build it)",
     run=:(𝒩⁻¹(real.(Ω.z), Ω))),
    (id="sninv", label="𝒮𝒩⁻¹", group="Laplace", base=false, parent="disk", schemes=DISK,
     about="𝒮𝒩⁻¹(u, D): the product 𝒮𝒩⁻¹ in coefficient space, on the disk",
     run=:(𝒮𝒩⁻¹(real.(Ω.z), Ω))),
    # volume and differential operators (on the disk)
    (id="volume", label="𝒱", group="Volume & differential", base=false, parent="disk", schemes=DISK,
     about="𝒱(u, D): the volume potential of the 2D Laplacian, a dense sum over the grid",
     run=:(𝒱(ones(size(Ω.z)), Ω))),
    (id="bilaplace", label="ℬ", group="Volume & differential", base=false, parent="disk", schemes=DISK,
     about="ℬ(u, D; κ): the volume potential of the 2D bilaplacian, a dense sum over the grid",
     run=:(ℬ(ones(size(Ω.z)), Ω; κ=0.5))),
    (id="bilaplace3d", label="𝒯", group="Volume & differential", base=false, parent="disk", schemes=DISK,
     about="𝒯(u, D): the single layer of the 3D bilaplacian, from four single layers 𝒮",
     run=:(𝒯(ones(size(Ω.z)), Ω))),
    (id="poisson", label="Δ⁻¹", group="Volume & differential", base=false, parent="disk", schemes=DISK,
     about="Δ⁻¹(f, g, D): the Poisson problem Δu = f with u = g on the boundary, solved by GMRES",
     run=:(Δ⁻¹(-1, 0, Ω))),
    (id="derivatives", label="Derivatives", group="Volume & differential", base=false, parent="disk", schemes=DISK,
     about="∂ζ, ∂ζ̄, ∂x, ∂y, grad, div, lap and the normal derivative ∂n of a smooth function, in coefficient space",
     run=:(let u = exp.(real.(Ω.z))
               ∂ζ(u, Ω); ∂ζ̄(u, Ω); ∂x(u, Ω); ∂y(u, Ω); div(grad(u, Ω), Ω); lap(u, Ω); ∂n(u, Ω) end)),
    # Stokes
    (id="stokes", label="𝒮_st", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="𝒮_st(f, Ω): the in-plane Stokes single layer of a force density (a tuple), and one component 𝒮_st(i, j, f, Ω)",
     run=:(let f = (real.(Ω.z), ones(size(Ω.z)))
               𝒮_st(f, Ω); 𝒮_st(1, 2, ComplexF64.(f[2]), Ω) end)),
    (id="stokesinv", label="𝒮_st⁻¹", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="𝒮_st⁻¹(u, Ω): the force density of a translating and of a rotating plate",
     run=:(begin 𝒮_st⁻¹((1.0, 0.0), Ω); 𝒮_st⁻¹((-imag.(Ω.z), real.(Ω.z)), Ω) end)),
    (id="mobmatrix", label="Mobility matrix", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="stokes_mobility_matrix(Ω): the 3 × 3 matrix of forces and torque of the rigid motions (here in free space, μ_top = 1)",
     run=:(stokes_mobility_matrix(Ω; μ_top=1.0))),
    (id="mobsolve", label="Mobility solve", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="stokes_mobility_solve: the velocity and angular velocity of one body in a shear flow, with a slip velocity and an applied force and torque",
     run=:(stokes_mobility_solve(shear(Ω.z), (0.1 .* ones(size(Ω.z)), zeros(size(Ω.z))), [1.0, 0.0, 0.1], Ω))),
    (id="normal", label="Normal forces", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="forces out of the plane in free space: 𝒮_st and 𝒮_st⁻¹ of 3-tuples (fx, fy, fz) with μ, stokes_normal_mobility_matrix, and a six-component stokes_mobility_solve",
     run=:(let x = real.(Ω.z), y = imag.(Ω.z)
               f = 𝒮_st⁻¹((1.0, 0.0, y), Ω; μ=1.0); 𝒮_st(f, Ω; μ=1.0)
               stokes_mobility_solve((shear(Ω.z)..., x), nothing, [1.0, 0.0, 1.0, 0.2, 0.0, 0.1], Ω; μ=1.0) end)),
    (id="offbody", label="Off-body velocity", group="Stokes", base=false, parent="disk", schemes=ALL,
     about="𝒮_st(tgt, f, Ω): the Stokes single layer at points outside the body, by direct quadrature",
     run=:(𝒮_st([3.0 + 0.0im, 2.0 + 2.0im], (ones(ComplexF64, size(Ω.z)), zeros(ComplexF64, size(Ω.z))), Ω))),
    # shape derivatives (on the disk)
    (id="dS", label="δ𝒮", group="Shape derivatives", base=false, parent="disk", schemes=DISK,
     about="δ𝒮(u, m, D): the shape derivative of the single layer for a boundary perturbation of mode m (here 2)",
     run=:(δ𝒮(ones(ComplexF64, size(Ω.z)), 2, Ω))),
    (id="dN", label="δ𝒩", group="Shape derivatives", base=false, parent="disk", schemes=DISK,
     about="δ𝒩(u, m, D): the shape derivative of the hypersingular operator",
     run=:(δ𝒩(ComplexF64.(real.(Ω.z) .* Ω.w), 2, Ω))),
    (id="dV", label="δ𝒱", group="Shape derivatives", base=false, parent="disk", schemes=DISK,
     about="δ𝒱(u, m, D): the shape derivative of the 2D Laplace volume potential",
     run=:(δ𝒱(ones(ComplexF64, size(Ω.z)), 2, Ω))),
    (id="dB", label="δℬ", group="Shape derivatives", base=false, parent="disk", schemes=DISK,
     about="δℬ(u, m, D): the shape derivative of the 2D bilaplace volume potential",
     run=:(δℬ(ones(ComplexF64, size(Ω.z)), 2, Ω))),
    # suspensions
    (id="suspension", label="Mobility solve, N bodies", group="Suspensions", base=false, parent="disk", schemes=ALL,
     about="suspension_mobility_solve of two nearby bodies in a shear flow with applied forces and a torque: GMRES preconditioned by single-body solves, cross terms summed by Ewald + NUFFT (StokesEwald), within-body terms removed with suspension_self_matrix, and close evaluation between disks or ellipses",
     run=:(suspension_mobility_solve(xcm, θcm, shear, nothing, FT, Ω; eps=1e-8, gmrestol=1e-8))),
    (id="fmm", label="method = :fmm", group="Suspensions", base=false, parent="disk", schemes=ALL,
     about="method = :fmm: the cross terms summed by FMM3D's Stokes FMM instead of Ewald",
     run=:(suspension_mobility_solve(xcm, θcm, shear, nothing, FT, Ω; eps=1e-8, gmrestol=1e-8, method=:fmm))),
    (id="noclose", label="close_eval = false", group="Suspensions", base=false, parent="disk", schemes=["disk", "ellipse"],
     about="close_eval = false: node quadrature for every cross term (a general conformal domain always uses it)",
     run=:(suspension_mobility_solve(xcm, θcm, shear, nothing, FT, Ω; eps=1e-8, gmrestol=1e-8, close_eval=false))),
    (id="selfmatrix", label="Precomputed S_self, M", group="Suspensions", base=false, parent="disk", schemes=ALL,
     about="suspension_self_matrix(Ω) and stokes_mobility_matrix(Ω) built once and passed to the solve as S_self and M, as a time-stepping simulation does",
     run=:(let S = suspension_self_matrix(Ω), M = stokes_mobility_matrix(Ω)
               suspension_mobility_solve(xcm, θcm, shear, nothing, FT, Ω; eps=1e-8, gmrestol=1e-8, S_self=S, M=M) end)),
    (id="veleval", label="Velocity from forces", group="Suspensions", base=false, parent="disk", schemes=ALL,
     about="suspension_velocity_eval (and 𝒮_st(xcm, θcm, f, Ω)): the velocity on every body of given force densities",
     run=:(let f = [(ones(size(Ω.z)), zeros(size(Ω.z))), (zeros(size(Ω.z)), ones(size(Ω.z)))]
               suspension_velocity_eval(xcm, θcm, f, Ω; eps=1e-8); 𝒮_st(xcm, θcm, f, Ω; eps=1e-8) end)),
    (id="velsolve", label="Forces from velocity", group="Suspensions", base=false, parent="disk", schemes=ALL,
     about="suspension_velocity_solve (and 𝒮_st⁻¹(xcm, θcm, u, Ω)): the force densities that give each body a prescribed velocity, by GMRES",
     run=:(let u = [(ones(size(Ω.z)), zeros(size(Ω.z))), (zeros(size(Ω.z)), ones(size(Ω.z)))]
               suspension_velocity_solve(xcm, θcm, u, Ω; eps=1e-8, gmrestol=1e-8); 𝒮_st⁻¹(xcm, θcm, u, Ω; eps=1e-8, gmrestol=1e-8) end)),
]

# ---- child: run the settings under coverage, one LCOV tracefile each -------------------------------------------
if Base.JLOptions().code_coverage != 0
    outdir = ARGS[1]
    Core.eval(Main, SETUP)
    for X in SETTINGS
        X.parent == "" || Core.eval(Main, :(Ω = DOMAINS[$(X.parent)]))
        sec = @elapsed begin
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), joinpath(outdir, X.id * ".pre.info"))
            Core.eval(Main, X.run)
            ccall(:jl_write_coverage_data, Cvoid, (Cstring,), joinpath(outdir, X.id * ".info"))
        end
        println(rpad(X.id, 14), round(sec; digits=1), " s"); flush(stdout)
    end
    exit()
end

# ---- parent: the top-level functions of src/ and their line ranges --------------------------------------------
# A function is a name in a file (all its methods there); its lines run from its definition to the next top-level
# statement, so a docstring belongs to the function it documents. A macro definition counts as a function named
# @name: the lines of `@wrap_operator` are those of the Disk methods of 𝒮, 𝒮⁻¹, 𝒩 and 𝒩⁻¹ it generates.
_unwrap(ex) = ex isa Expr && ex.head === :macrocall ? _unwrap(ex.args[end]) : ex
_docstring(ex) = ex isa Expr && ex.head === :macrocall && ex.args[1] == GlobalRef(Core, Symbol("@doc")) &&
                 ex.args[3] isa String ? ex.args[3] : nothing
const GENERIC = Set([:size, :eltype, :length, :show, :getindex, :iterate, :copy, :convert, :(==), :(*), :mul!,
                     :getproperty])
const BASETYPES = Set([:AbstractVector, :AbstractMatrix, :AbstractArray, :Vector, :Matrix, :Number, :Real, :Integer,
                       :Int, :Float64, :IO, :MIME, :Function, :Any, :Tuple, :NamedTuple])
_tname(t) = t isa Symbol ? t : t isa Expr && t.head === :curly ? _tname(t.args[1]) :
            t isa Expr && t.head === :. ? t.args[end].value : nothing
function _argtype(sig)
    for a in sig.args[2:end]
        a isa Expr && a.head === :parameters && continue
        a isa Expr && a.head === :kw && (a = a.args[1])
        a isa Expr && a.head === :(::) || continue
        t = _tname(a.args[end])
        t === nothing || t in BASETYPES || return t
    end
    nothing
end
function _fname(ex)
    ex = _unwrap(ex)
    ex isa Expr && ex.head === :macro && return "@" * string(ex.args[1].args[1])     # macro wrap_operator(…)
    ex isa Expr && ex.head in (:function, :(=)) || return nothing
    sig = ex.args[1]
    while sig isa Expr && sig.head in (:where, :(::)) && !(sig.head === :(::) && length(sig.args) == 1)
        sig = sig.args[1]
    end
    sig isa Expr && sig.head === :call || return nothing
    f = sig.args[1]
    f isa Expr && f.head === :curly && (f = f.args[1])
    f isa Expr && f.head === :. && (f = f.args[end].value)                          # LinearAlgebra.ldiv!, Base.getproperty
    f isa Expr && f.head === :(::) && return "(::" * string(_tname(f.args[end])) * ")"   # (op::T)(x)
    f isa Symbol || return nothing
    t = f in GENERIC ? _argtype(sig) : nothing
    t === nothing ? string(f) : string(f, "(::", t, ")")
end
function _summary(doc)
    doc === nothing && return ""
    lines = split(rstrip(doc), '\n')              # not strip: the signature block's indent marks it
    k = findfirst(l -> !startswith(l, "    ") && !isempty(strip(l)), lines)       # skip the signature block
    k === nothing && return ""
    para = String[]
    for l in lines[k:end]
        isempty(strip(l)) && break
        push!(para, strip(l))
    end
    X = join(para, " ")
    m = match(r"^(.*?\.)(\s|$)", X)
    m === nothing ? X : m.captures[1]
end
function functions_of(path)
    text = read(path, String)
    nlines = count(==('\n'), text) + 1
    top = Meta.parseall(text; filename=path)
    stmts = Tuple{Int,Any}[]
    line = 1
    for a in top.args
        if a isa LineNumberNode
            line = a.line
        else
            push!(stmts, (line, a))
        end
    end
    fns = Dict{String,Any}()
    for (k, (l0, ex)) in enumerate(stmts)
        name = _fname(ex)
        name === nothing && continue
        l1 = k < length(stmts) ? stmts[k+1][1] - 1 : nlines
        f = get!(fns, name) do
            Dict("name" => name, "line" => l0, "ranges" => UnitRange{Int}[], "doc" => "")
        end
        push!(f["ranges"], l0:l1)
        isempty(f["doc"]) && (f["doc"] = _summary(_docstring(ex)))
    end
    sort!(collect(values(fns)); by=f -> f["line"]), nlines
end
function counts(info)                                # line counts of an LCOV tracefile: (file, line) => count
    c = Dict{Tuple{String,Int},Int}()
    file = ""
    for l in eachline(info)
        if startswith(l, "SF:")
            file = realpath(l[4:end])
        elseif startswith(l, "DA:")
            ln, n = parse.(Int, split(l[4:end], ',')[1:2])
            c[(file, ln)] = n
        end
    end
    c
end
function executed(pre, post)                         # the lines a setting ran: file => Set(lines)
    before, after = counts(pre), counts(post)
    hits = Dict{String,Set{Int}}()
    for ((file, ln), n) in after
        n > get(before, (file, ln), 0) && push!(get!(hits, file, Set{Int}()), ln)
    end
    hits
end

# the code of src/ except the module file (tuple arithmetic, includes, exports, __init__)
files = sort([relpath(joinpath(d, f), SRC) for (d, _, fs) in walkdir(SRC) for f in fs
              if endswith(f, ".jl") && f != "ProjectedSphericalHarmonics.jl"])
tmp = isempty(ARGS) ? mktempdir() : mkpath(ARGS[1])
if !all(X -> isfile(joinpath(tmp, X.id * ".pre.info")) && isfile(joinpath(tmp, X.id * ".info")), SETTINGS)
    cmd = `$(Base.julia_cmd()) -t 1 --project=$ROOT --code-coverage=@$SRC --code-coverage=$(joinpath(tmp, "exit.info")) $(@__FILE__) $tmp`
    println("running ", length(SETTINGS), " settings under coverage …")
    run(cmd)
end
hits = Dict(X.id => executed(joinpath(tmp, X.id * ".pre.info"), joinpath(tmp, X.id * ".info")) for X in SETTINGS)
fnlist = Any[]; filelist = Any[]
for f in files
    path = realpath(joinpath(SRC, f))
    fns, nlines = functions_of(path)
    push!(filelist, (path=f, dir=dirname(f) == "" ? "src" : dirname(f), lines=nlines))
    for fn in fns
        used = [X.id for X in SETTINGS if any(r -> any(in(get(hits[X.id], path, Set{Int}())), r), fn["ranges"])]
        push!(fnlist, (name=fn["name"], file=f, line=fn["line"], doc=fn["doc"], used=used))
    end
end

# ---- write the data ---------------------------------------------------------------------------------------------
js(X::AbstractString) = "\"" * escape_string(X) * "\""
js(v::AbstractVector) = "[" * join(js.(v), ",") * "]"
js(x::Real) = string(x)
js(b::Bool) = string(b)
js(nt::NamedTuple) = "{" * join((string(k) * ":" * js(v) for (k, v) in pairs(nt)), ",") * "}"
open(OUT, "w") do io
    println(io, "// Generated by docs/codemap.jl: the top-level functions of ProjectedSphericalHarmonics' src/ and the settings that run them.")
    println(io, "window.CODEMAP = {")
    rev = try strip(read(`git -C $ROOT rev-parse --short HEAD`, String)) catch; "?" end
    println(io, "  generated: ", js(string(Base.Libc.strftime("%Y-%m-%d", time()), ", src at commit ", rev, " (with uncommitted changes, if any)")), ",")
    println(io, "  settings: [")
    for X in SETTINGS
        println(io, "    ", js((id=X.id, label=X.label, group=X.group, about=X.about, base=X.base, parent=X.parent, schemes=X.schemes)), ",")
    end
    println(io, "  ],\n  files: [")
    for f in filelist
        println(io, "    ", js(f), ",")
    end
    println(io, "  ],\n  functions: [")
    for f in fnlist
        println(io, "    ", js(f), ",")
    end
    println(io, "  ],\n};")
end
println("wrote ", relpath(OUT, ROOT), ": ", length(fnlist), " functions in ", length(filelist), " files, ",
        count(f -> !isempty(f.used), fnlist), " run by at least one setting")
