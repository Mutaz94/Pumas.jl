## Types

struct Person{T1,T2,T3,T4,T5}
  id::Int
  obs::T1
  obs_times::T2
  covariates::T3
  event_times::T4
  events::T5
end

struct Population{T} <: AbstractVector{T}
  patients::T
end

Base.start(A::Population) = start(A.patients)
Base.next(A::Population, i) = next(A.patients, i)
Base.done(A::Population, i) = done(A.patients, i)

# size
Base.length(A::Population) = length(A.patients)
Base.size(A::Population) = size(A.patients)

# indexing
@inline function Base.getindex(A::Population, I...)
    @boundscheck checkbounds(A.patients, I...)
    @inbounds return A.patients[I...]
end
@inline function Base.setindex!(A::Population, x, I...)
    @boundscheck checkbounds(A.patients, I...)
    @inbounds A.patients[I...] = x
end
Base.indices(A::Population) = indices(A.patients)
Base.IndexStyle(::Type{<:Population}) = Base.IndexLinear()

## Parsing Functions

function generate_person(id,covariates,dvs,raw_data)
  id_start = findfirst(x -> x==id, raw_data[:id])
  id_end   = findlast(x ->  x==id, raw_data[:id])
  person_data = raw_data[id_start:id_end, :]
  covs = Dict([sym => person_data[1,sym] for sym in covariates])
  obs_idxs = find(x ->  x==0, person_data[:mdv])
  obs = person_data[obs_idxs,dvs]
  obs_times = person_data[obs_idxs,:time]
  event_idxs = find(x ->  x!=0, person_data[:evid])
  event_times = person_data[event_idxs,:time]
  events = person_data[event_idxs,[:amt,:evid]]
  Person(id,obs,obs_times,covs,event_times,events)
end

function process_data(path,covariates,dvs;kwargs...)
  raw_data = readtable(path;kwargs...)
  ids = unique(raw_data[:id])
  pop = Population(map(id->generate_person(id,covariates,dvs,raw_data),ids))
end
