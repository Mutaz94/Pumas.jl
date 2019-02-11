module NCA

using Reexport
using GLM
using DataFrames
using RecipesBase
using Pkg, Dates, Printf, LinearAlgebra
import ..PuMaS: Formulation, IV, EV

include("type.jl")
include("data_parsing.jl")
include("utils.jl")
include("interpolate.jl")
include("auc.jl")
include("simple.jl")

export DataFrame

export NCASubject, NCAPopulation, NCADose, showunits
export parse_ncadata
#export auc, aumc, lambdaz, auc_extrap_percent, aumc_extrap_percent,
#       clast, tlast, cmax, tmax, cmin, c0, tmin, thalf, cl, clf, vss, vz,
#       bioav, tlag, mrt, mat, tau, cavg, fluctation, accumulationindex,
#       swing, superposition
export NCAReport
export normalizedose

for f in (:lambdaz, :lambdazr2, :lambdazadjr2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst,
          :cmax, :tmax, :cmin, :c0, :tmin, :clast, :tlast, :thalf, :cl, :clf, :vss, :vz,
          :interpextrapconc, :auc, :aumc, :auc_extrap_percent, :aumc_extrap_percent,
          :bioav, :tlag, :mrt, :mat, :tau, :cavg, :fluctation, :accumulationindex,
          :swing)
  @eval $f(conc, time, args...; kwargs...) = $f(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
  @eval function $f(pop::NCAPopulation, args...; id_occ=true, kwargs...) # NCAPopulation handling
    if ismultidose(pop)
      sol = map(enumerate(pop)) do (i, subj)
        if $f == mat
          _sol = $f(subj, args...; kwargs...)
          sol  = vcat(_sol, fill(missing, length(subj.dose)-1)) # make `mat` as long as the other ones
        else
          sol = $f(subj, args...; kwargs...)
        end
        id_occ ? DataFrame(id=subj.id, occasion=eachindex(sol), $f=sol) : DataFrame($f=sol)
      end
    else
      sol = map(pop) do subj
        id_occ ? DataFrame(id=subj.id, $f=$f(subj, args...; kwargs...)) : DataFrame($f=$f(subj, args...; kwargs...))
      end
    end
    return vcat(sol...) # splat is faster than `reduce(vcat, sol)`
  end
end

# special handling for superposition
superposition(conc, time, args...; kwargs...) = superposition(NCASubject(conc, time; kwargs...), args...; kwargs...) # f(conc, time) interface
function superposition(pop::NCAPopulation, args...; kwargs...) # NCAPopulation handling
  sol = map(pop) do subj
    res = superposition(subj, args...; kwargs...)
    DataFrame(id=subj.id, conc=res[1], time=res[2])
  end
  return vcat(sol...) # splat is faster than `reduce(vcat, sol)`
end

# Multiple dosing handling
for f in (:clast, :tlast, :cmax, :tmax, :cmin, :tmin, :_auc, :tlag, :mrt, :fluctation,
          :cavg, :tau, :accumulationindex, :swing,
          :lambdaz, :lambdazr2, :lambdazadjr2, :lambdazintercept, :lambdaznpoints, :lambdaztimefirst)
  @eval function $f(nca::NCASubject{C,TT,T,tEltype,AUC,AUMC,D,Z,F,N,I,P,ID}, args...; kwargs...) where {C,TT,T,tEltype,AUC,AUMC,D<:AbstractArray,Z,F,N,I,P,ID}
    obj = map(eachindex(nca.dose)) do i
      subj = subject_at_ithdose(nca, i)
      $f(subj, args...; kwargs...)
    end
  end
end

end
