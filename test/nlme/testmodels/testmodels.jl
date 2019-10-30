# Common place for defining models with associated data and parameters to avoid repeating the same model definitions during testing

PUMASMODELS = Dict()
PUMASMODELS["1cpt"] = Dict()
PUMASMODELS["1cpt"]["oral"] = Dict()
PUMASMODELS["1cpt"]["oral"]["normal_additive"] = Dict()

PUMASMODELS["misc"] = Dict()

include("warfarin.jl")
include("hcv.jl")
