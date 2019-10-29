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
  time = first(population).time

  # Copmute the quantile of the samples
  empirical = [quantile([subject.observations[dvname][ti] for subject in population], probabilities) for ti in eachindex(time)]

  # Simulate `reps` new populations
  sims = [simobs(m, population, param) for i in 1:reps]

  # Compute the probabilities for the CI based on the level
  ci_probabilities = ((1 - ci_level)/2, (1 + ci_level)/2)

  # Compute the quantiles of the simulated data for the CIs
  simulated = map(eachindex(time)) do tj
    simulated_quantiles = map(sims) do sim
      pop_i_at_tj = map(sim) do subject
        return subject.observed[dvname][tj]
      end
      return quantile(pop_i_at_tj, probabilities)
    end
    tuple_of_vectors = map((1, 2, 3)) do i
      quantile(getindex.(simulated_quantiles, i), ci_probabilities)
    end
    return tuple_of_vectors
  end

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
    ntuple(i->map(t->getfield(t,i),c), Val(length(c[1])))
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
