using EarthSciData, EarthSciMLBase, GasChem,
    ModelingToolkit, OrdinaryDiffEq, Dates
using Plots, ProgressMeter
using Statistics
using GeoDataFrames
using Unitful
using NCDatasets
using Test
using Profile, PProf

include("utils.jl")
include("callback.jl")
include("output.jl")
include("operators.jl")
include("strang_problem.jl")


ModelingToolkit.check_units(eqs...) = nothing

const start = Dates.datetime2unix(Dates.DateTime(2016, 5, 15))

@parameters t

quick = false # Set to true for a quick simulation with low resolution.
if quick
    lons = collect(-130.0:2:-60.0)
    lats = collect(9.75:2:60.0)
    levs = collect(1.0:72)
    s_per_timestep = 300
else
    lons = collect(-129.975:2:-61.975)
    lats = collect(17.025:2:59.025)
    levs = collect(1.0:72) # About 11 km: https://wiki.seas.harvard.edu/geos-chem/index.php/GEOS-Chem_vertical_grids
    s_per_timestep = 300
end

write_interval = 3600.0 # seconds
nsims = length(lons) * length(lats) * length(levs)
ndays = 15 # 0.07
ntimesteps = Int(round(ndays * 24 * 60 * 60 / s_per_timestep))
finish = start + ntimesteps * s_per_timestep

geos = GEOSFP{Float64}("0.25x0.3125_NA", t; coord_defaults=Dict(:lat => 34.0, :lon => -100.0, :lev => 1))

@parameters lon lat lev
@parameters Δz = 60 [unit = u"m"]
Δz2 = 60.0
emis = NEI2016MonthlyEmis{Float64}("mrggrid_withbeis_withrwc", t, lon, lat, lev, Δz)

# Create a model by combining SuperFast chemistry and FastJX photolysis.
model_ode = SuperFast(t) + FastJX(t) #+ geos

# Set up model
sys = structural_simplify(get_mtk(model_ode))

defaults = ModelingToolkit.get_defaults(sys)
default_ps = [defaults[p] for p in parameters(sys)]
prob = ODEProblem(sys, [], (start, finish), default_ps, jac=true)
u0 = Float64.([defaults[v] for v in states(sys)])

indexof(sym, syms) = findfirst(isequal(sym), syms)

adv = advection_operator(geos, states(sys), lons, lats, levs)
emit = emis_operator(sys, emis, lons, lats, levs, Δz2)


# Run
write_times = start:write_interval:finish
outfile = "/home/xiaoran/ra/aq_demo/demo_output/out_strang.nc"
o = NetCDFOutputter(lons, lats, levs, states(sys), write_times, outfile)

u = zeros(Float64, length(states(sys)), length(lons), length(lats), length(levs));
for i in eachindex(states(sys))
    u[i, :, :, :] .= u0[i] * 0.00001 # Initial conditions
end
p_grid = zeros(length(prob.p), size(u)[2:4]...);
for ii in CartesianIndices(size(p_grid)[2:4])
    p_grid[:, ii] .= prob.p
end

r = StrangProblem(prob, u, p_grid, o, adv, emit);

# Profile.Allocs.clear()
# Profile.Allocs.@profile sample_rate=0.0001 run!(r, start, finish, Float64(s_per_timestep))
# PProf.Allocs.pprof()

#uu = reshape(r.u, :, length(lons), length(lats), length(levs))
#heatmap(uu[indexof(sys.superfast.NO, states(sys)), :, :, 1])

@time run!(r, start, finish, Float64(s_per_timestep))

#@code_warntype step!(r, start, Float64(s_per_timestep))

#datetime2unix(now())
