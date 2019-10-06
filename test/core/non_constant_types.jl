using Pumas

data = read_pumas(example_data("data1"),
                      cvs = [:sex,:wt,:etn,:time])

tmp = []
push!(tmp, Subject(obs=data[1].observations, time=data[1].time, cvs=(sex=1, wt=[51.6], etn=1, time=data[1].covariates.time), evs=data[1].events))
push!(tmp, data[2])
new_data = identity.(tmp)

mdsl = @model begin
    @param begin
        θ ∈ VectorDomain(4, lower=zeros(4), init=ones(4))
        Ω ∈ PSDDomain(2)
        Σ ∈ RealDomain(lower=0.0, init=1.0)
        a ∈ ConstDomain(0.2)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates sex wt etn time

    @pre begin
        _wt = @tvcov wt time DataInterpolations.ZeroSpline
        Ka = θ[1]
        CL = t -> θ[2] * ((_wt(t)/70)^0.75) * (θ[4]^sex) * exp(η[1])
        V  = θ[3] * exp(η[2])
    end

    @vars begin
      conc = Central / V
      conc2 = Central^2
    end

    @dynamics begin
        Depot'   := -Ka*Depot # test for `:=` handling
        Central' =  Ka*Depot - CL(t)*conc
    end

    @derived begin
      dv ~ @. Normal(conc, conc*Σ)
      T_max = maximum(t)
    end

    @observed begin
      obs_cmax = maximum(dv)
    end
end

obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleSerial())
obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleThreads())
obs = simobs(mdsl, new_data, init_param(mdsl), ensemblealg = EnsembleDistributed())
