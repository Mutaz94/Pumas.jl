struct VPC
  time
  empirical
  simulated
  probabilities
  confidence_level
end

function Base.show(io::IO, ::MIME"text/plain", vpc::VPC)
  println(io, summary(vpc))
end

function quantile_t_sub(population,probabilities,dvname,times)
  quantiles = []
  for ti in times
      quantile_ti = []
      for subject in population
        if ti in subject.time
          tj = findall(x-> x==ti, subject.time)[1]
          push!(quantile_ti, subject.observations[dvname][tj])
        end
      end
      quantile_ti = [i for i in quantile_ti]
      push!(quantiles,quantile(quantile_ti,probabilities))
  end
  return quantiles
end

function quantile_t_sim(sims,probabilities,dvname,ci_probabilities,times)
  quantiles = []
  for ti in times
    quantile_quantiles = []
    quantile_ti = []
    for sim in sims
      quantile_ti_sub = []
      for subject in sim
        if ti in subject.times
          tj = findall(x-> x==ti, subject.times)[1]
          push!(quantile_ti_sub,subject.observed[dvname][tj])
        end
      end
      push!(quantile_ti,quantile(quantile_ti_sub,probabilities))
    end
    for i in [1, 2, 3]
      push!(quantile_quantiles,quantile(getindex.(quantile_ti, i), ci_probabilities))
    end
    push!(quantiles,quantile_quantiles)
  end
  return quantiles
end

function vpc(
  m::PumasModel,
  population::Population,
  param::NamedTuple,
  reps::Integer = 499;
  probabilities::NTuple{3,Float64} = (0.1, 0.5, 0.9),
  ci_level::Float64 = 0.95,
  dvname::Symbol = :dv
  )

  # FIXME! For now we assume homogenous sampling time across subjects. Eventually this should handle inhomogenuous sample times, e.g. with binning but preferably with some kind of Loess like estimator
  time = unique(sort(collect(Iterators.flatten([subject.time for subject in population]))))

  # Compute the quantile of the samples
  empirical = quantile_t_sub(population,probabilities,dvname,time)

  # Simulate `reps` new populations
  sims = [simobs(m, population, param) for i in 1:reps]

  # Compute the probabilities for the CI based on the level
  ci_probabilities = ((1 - ci_level)/2, (1 + ci_level)/2)

  # Compute the quantiles of the simulated data for the CIs
  simulated = quantile_t_sim(sims,probabilities,dvname,ci_probabilities,time)

  return VPC(time, empirical, simulated, probabilities, ci_level)
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

# Define an upzip function while we are waiting for the one in Base.
function unzip(itr)
    c = collect(itr)
    ntuple(i->map(t->t[i],c), Val(length(c[1])))
end

@recipe function f(vpc::VPC)
  empirical = unzip(vpc.empirical)
  simulated = unzip(vpc.simulated)

  empirical_style = [:dashdot, :solid, :dot]

  title --> "Confidence interval VPC"
  for i in 1:3
    @series begin
      label --> "Empirical $(vpc.probabilities[i]*100)% quantile"
      xlabel --> "time"
      linewidth --> 2
      linecolor --> :red
      linestyle --> empirical_style[i]
      vpc.time, empirical[i]
    end

    @series begin
      label --> hcat(i == 3 ? "Simulated $(vpc.confidence_level*100)% confidence intervals" : "", "")
      xlabel --> "time"
      fillrange --> hcat(reverse(unzip(simulated[i]))...)
      fillcolor --> :blue
      fillalpha --> 0.2
      linewidth --> 0.0
      vpc.time, hcat(empirical[i], empirical[i])
    end
  end
end
