using Pumas, SafeTestsets, Test

if haskey(ENV,"GROUP")
    group = ENV["GROUP"]
else
    group = "All"
end

is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )


@time begin

if group == "All" || group == "Core"
    include("core/runtests.jl")
end

if group == "All" || group == "Parallel"
    # Do not put into a module because processes are spawned
    @time @testset "Parallelism Tests" begin
        include("core/parallel.jl")
    end
end

if group == "All" || group == "NCA"
    include("nca/runtests.jl")
end

if group == "All" || occursin("NLME", group)
    include("nlme/runtests.jl")
end

if group == "All" || group == "Features"
    include("features/runtests.jl")
end

if group == "All" || group == "IVIVC"
    include("ivivc/runtests.jl")
end

end
