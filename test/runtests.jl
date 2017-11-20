using PKPDSimulator
using Base.Test

tic()
@time @testset "Parsing Tests" begin include("parsing_tests.jl") end
@time @testset "Single Dosage Tests" begin include("single_dosage_tests.jl") end
@time @testset "Analytical Single Dosage Tests" begin
                include("analytical_single_dosage_tests.jl") end
@time @testset "Multiple Dosage Tests" begin include("multiple_dosage_tests.jl") end
@time @testset "Analytical Multiple Dosage Tests" begin
                include("analytical_multiple_dosage_tests.jl") end
toc()
