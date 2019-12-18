struct VPC
  time
  empirical
  simulated
  probabilities
  confidence_level
  stratify_cvs
  covariate_values
end

function Base.show(io::IO, ::MIME"text/plain", vpc::VPC)
  println(io, summary(vpc))
end

function quantile_t_data(sim,probabilities,dvname,times,stratify_on::Nothing)
  df = DataFrame(t=times, dv=collect(Iterators.flatten([sim[i].observations[dvname] for i in 1:length(sim)])))
  df_quantile = by(df, :t, (low_quantile = :dv => t -> ( quantile(t, probabilities[1])), quantile = :dv => t -> ( quantile(t, probabilities[2])), high_quantile = :dv => t -> ( quantile(t, probabilities[3]))))
  df_quantile, nothing
end

function quantile_t_data(sim,probabilities,dvname,times,stratify_on::Symbol)
  df = DataFrame(t=times, dv=collect(Iterators.flatten([sim[i].observations[dvname] for i in 1:length(sim)])), cvs = collect(Iterators.flatten([[sim[i].covariates[stratify_on] for j in 1:length(sim[i].observations[dvname])] for i in 1:length(sim)])))
  cvs__ = df[!,:cvs]
  if length(unique(df[!,:cvs])) > 4
      cvs_ = quantile(unique(df[!,:cvs]), [0.25,0.5,0.75,1.0])
      cvs__ = map(x-> cvs_[sum([x>cvs_[1],x>cvs_[2],x>cvs_[3]])+1],df[!,:cvs])
      df[!,:cvs] = cvs__
  end
  df_quantile = by(df, [:t,:cvs], (low_quantile = :dv => t -> ( quantile(t, probabilities[1])), quantile = :dv => t -> ( quantile(t, probabilities[2])), high_quantile = :dv => t -> ( quantile(t, probabilities[3]))))
  groupby(df_quantile, :cvs), cvs__
end

function quantile_t_sim(sim,probabilities,dvname,times,cvs__::Nothing)
  df = DataFrame(t=times, dv=collect(Iterators.flatten([sim[i].observed[dvname] for i in 1:length(sim)])))
  df_quantile = by(df, :t, (low_quantile = :dv => t -> ( quantile(t, probabilities[1])), quantile = :dv => t -> ( quantile(t, probabilities[2])), high_quantile = :dv => t -> ( quantile(t, probabilities[3]))))
  df_quantile
end

function quantile_t_sim(sim,probabilities,dvname,times,cvs__::AbstractVector)
  df = DataFrame(t=times, dv=collect(Iterators.flatten([sim[i].observed[dvname] for i in 1:length(sim)])), cvs = cvs__)
  df_quantile = by(df, [:t,:cvs], (low_quantile = :dv => t -> ( quantile(t, probabilities[1])), quantile = :dv => t -> ( quantile(t, probabilities[2])), high_quantile = :dv => t -> ( quantile(t, probabilities[3]))))
  df_quantile
end

function quantile_sub_sim(dfs,ci_probabilities,times,stratify_on::Nothing)
  df_aggregate = DataFrame()
  map(df -> append!(df_aggregate, df), dfs)
  df_sim_quantile =  by(df_aggregate, [:t], (:low_quantile => t -> quantile(t, ci_probabilities), :quantile => t -> quantile(t, ci_probabilities),:high_quantile => t -> quantile(t, ci_probabilities)))
  return df_sim_quantile 
end

function quantile_sub_sim(dfs,ci_probabilities,times,stratify_on::Symbol)
  df_aggregate = DataFrame()
  map(df -> append!(df_aggregate, df), dfs)
  df_sim_quantile =  by(df_aggregate, [:t,:cvs], (:low_quantile => t -> quantile(t, ci_probabilities), :quantile => t -> quantile(t, ci_probabilities),:high_quantile => t -> quantile(t, ci_probabilities)))
  return groupby(df_sim_quantile, :cvs) 
end

function vpc(
  m::PumasModel,
  population::Population,
  param::NamedTuple,
  reps::Integer = 499;
  probabilities::NTuple{3,Float64} = (0.1, 0.5, 0.9),
  ci_level::Float64 = 0.95,
  dvname::Symbol = :dv,
  stratify_on = nothing
  )

  time = collect(Iterators.flatten([subject.time for subject in population]))

  # Compute the quantile of the samples
  empirical, cvs__ = quantile_t_data(population, probabilities, dvname, time, stratify_on)

  # Simulate `reps` new populations
  sims = [simobs(m, population, param) for i in 1:reps]
  # Compute the probabilities for the CI based on the level
  ci_probabilities = ((1 - ci_level)/2, (1 + ci_level)/2)

  # Compute the quantiles of the simulated data for the CIs
  sim_quantile = map(sim -> quantile_t_sim(sim, probabilities, dvname, time, cvs__), sims)
  simulated = quantile_sub_sim(sim_quantile, ci_probabilities, time, stratify_on)

  return VPC(time, empirical, simulated, probabilities, ci_level, stratify_on, cvs__ != nothing ? unique(cvs__) : nothing)
end

"""
vpc(fpm::FittedPumasModel, reps::Integer=499; kwargs...)

Computes the quantiles for VPC for a `FittedPumasModel` with simulated confidence intervals around the empirical quantiles based on `reps` simulated populations. The default is to compute the 10th, 50th and 90th percentiles.

The following keyword arguments are supported:
 - `probabilities::NTuple{3,Float64}`: A three-tuple of the probabilities for which the quantiles will be computed. The default is `(0.1, 0.5, 0.9)`.
 - `ci_level::Float64`: Confidence level to use for the simulated confidence intervals. The default it `0.95`.
 - `dvname::Symbol`: The name of the dependent variable to use for the VPCs. The default is `:dv`.
"""
vpc(fpm::FittedPumasModel, reps::Integer=499; kwargs...) = vpc(fpm.model, fpm.data, coef(fpm), reps; kwargs...)

@recipe function f(vpc::VPC)
  empirical_style = [:dashdot, :solid, :dot]
  cvs = vpc.covariate_values
  plot_title --> "Confidence interval VPC"
  if :cvs in names(vpc.empirical)
    layout --> length(vpc.empirical)
    for cv_ind in 1:length(vpc.empirical)
      legendfontsize --> 4
      ribbon_simulated = hcat([vpc.simulated[cv_ind][!,i] for i in [:low_quantile_function, :quantile_function, :high_quantile_function]])
      for i in 1:3
        @series begin
          subplot --> cv_ind
          label --> "Empirical $(vpc.probabilities[i]*100)% quantile"
          xlabel --> "time"
          linewidth --> 2
          linecolor --> :red
          linestyle --> empirical_style[i]
          legend --> cv_ind ==  length(vpc.empirical) ? :outerbottom : :false
          vpc.empirical[cv_ind][!,1], vpc.empirical[cv_ind][!,i+2]
        end

        @series begin
          subplot --> cv_ind
          title --> "$(vpc.stratify_cvs)"*"<"*string(cvs[cv_ind])
          label --> hcat(i == 3 ? "Simulated $(vpc.confidence_level*100)% confidence intervals" : "", "")
          xlabel --> "time"
          fillrange --> [[ribbon_simulated[i][j][1] for j in 1:length(ribbon_simulated[i])],[[ribbon_simulated[i][j][2] for j in 1:length(ribbon_simulated[i])]]]
          fillcolor --> :blue
          fillalpha --> 0.2
          linewidth --> 0.0
          legend --> cv_ind == length(vpc.empirical) ? :outerbottom : :false
          vpc.simulated[cv_ind][!,1], hcat(vpc.empirical[cv_ind][!,i+2], vpc.empirical[cv_ind][!,i+2])
        end
      end
    end
  else
    ribbon_simulated = hcat([vpc.simulated[!,i] for i in [:low_quantile_function, :quantile_function, :high_quantile_function]])
    for i in 1:3
      @series begin
        label --> "Empirical $(vpc.probabilities[i]*100)% quantile"
        xlabel --> "time"
        linewidth --> 2
        linecolor --> :red
        linestyle --> empirical_style[i]
        vpc.empirical[!,1], vpc.empirical[!,i+1]
      end

      @series begin
        label --> hcat(i == 3 ? "Simulated $(vpc.confidence_level*100)% confidence intervals" : "", "")
        xlabel --> "time"
        fillrange --> [[ribbon_simulated[i][j][1] for j in 1:length(ribbon_simulated[i])],[[ribbon_simulated[i][j][2] for j in 1:length(ribbon_simulated[i])]]]
        fillcolor --> :blue
        fillalpha --> 0.2
        linewidth --> 0.0
        vpc.simulated[!,1], hcat(vpc.empirical[!,i+1], vpc.empirical[!,i+1])
      end
    end
  end
end
