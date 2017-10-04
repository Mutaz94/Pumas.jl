using PKPDSimulator

# Read the data
covariates = [:sex,:wt,:etn]
dvs = [:dv]
data = process_data(joinpath(Pkg.dir("PKPDSimulator"),"examples/data1.csv"),
                 covariates,dvs)

# Define the ODE

function depot_model(t,u,p,du)
 Depot,Central = u
 Ka,CL,V = p
 du[1] = -Ka*Depot
 du[2] =  Ka*Depot - (CL/V)*Central
end
f = ParameterizedFunction(depot_model,[2.0,20.0,100.0])

# Population parameters

θ = [2.268,74.17,468.6,0.5876] # [2.0,3.0,10.0,1.0] # Unfitted
ω = [0.05 0.0
     0.0 0.2]

# User definition of the set_parameters! function

function set_parameters!(p,θ,η,z)
  wt,sex = z[:wt],z[:sex]
  Ka = θ[1]
  CL = θ[2]*((wt/70)^0.75)*(θ[4]^sex)*exp(η[1])
  V  = θ[3]*exp(η[2])
  p[1] = Ka; p[2] = CL; p[3] = V
end

# Call simulate
tspan = (0.0,19.0)
num_dependent = 2
sol = simulate(f,tspan,num_dependent,set_parameters!,θ,ω,data)

#=
using Plots; plotly()
plot(sol,title="Plot of all trajectories",xlabel="time")
summ = MonteCarloSummary(sol,0:0.1:19)
plot(summ,title="Summary plot",xlabel="time")
=#

output_func = function (sol,i)
  sol(data[i].obs_times;idxs=2)./sol.prob.f.params[3],false
end
sol = simulate(f,tspan,num_dependent,set_parameters!,θ,ω,data,output_func)
