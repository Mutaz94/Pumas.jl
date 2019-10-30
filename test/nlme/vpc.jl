using Pumas, Plots, Test

# Make sure that PUMASMODELS dict is loaded
if !isdefined(Main, :PUMASMODELS)
  Base.include(Main, "testmodels/testmodels.jl")
end

import Main: PUMASMODELS

@testset "VPC" begin
    warfarin = PUMASMODELS["1cpt"]["oral"]["normal_additive"]["warfarin"]

    ft = fit(warfarin["analytical"], warfarin["data"], warfarin["param"], Pumas.FOCEI())

    _vpc = vpc(ft)

    @test _vpc isa Pumas.VPC

    p = plot(_vpc)

    @test p isa Plots.Plot
end
