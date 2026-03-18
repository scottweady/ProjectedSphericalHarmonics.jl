using Test

@testset "TriangularCoeffArray" begin
    include("struct_test.jl")
    include("array_interface_test.jl")
    include("arithmetic_test.jl")
    include("reorder_test.jl")
    include("transforms.jl")
end
