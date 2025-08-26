include("./src/reimpl_ca.jl")
using OrdinaryDiffEq
using CairoMakie, LaTeXStrings
using ColorSchemes

set_theme!(theme_latexfonts(), colormap=ColorSchemes.thermal)

reltol = 1e-8
abstol = 1e-8

u0 = [0.1, 100.0, 0.08, 0.1]
tspan = (0.0, 1800.0)

mp = reimpl_ca.MagneticField()
p = reimpl_ca.CalciumModel(mp=mp)
p_4mT = remake(p, mp=remake(p.mp, B=4e-3))

prob = ODEProblem(reimpl_ca.dWdt, u0, tspan, p)
prob_4mT = ODEProblem(reimpl_ca.dWdt, u0, tspan, p_4mT)

# Plot 1
frequencies_mHz = range(0.0, 1.7e-3π, length=401)

function prob_func_mHz(prob, i, repeat)
    remake(prob, p=remake(prob.p, mp=remake(prob.p.mp, ω=frequencies_mHz[i])))
end

ensemble_prob_mHz = EnsembleProblem(prob, prob_func=prob_func_mHz)
sim_mHz = solve(
    ensemble_prob_mHz, DP5(), EnsembleThreads(),
    reltol=reltol, abstol=abstol, trajectories=size(frequencies_mHz)[1]
)

# Plot 2
inductions = range(0.0, 100.0e-3, length=401)

function prob_func_induction(prob, i, repeat)
    remake(prob, p=remake(prob.p, mp=remake(prob.p.mp, B=inductions[i])))
end

ensemble_prob_induction = EnsembleProblem(prob, prob_func=prob_func_induction)
sim_induction = solve(
    ensemble_prob_induction, DP5(), EnsembleThreads(),
    reltol=reltol, abstol=abstol, trajectories=size(inductions)[1]
)

# Plot 3
frequencies_Hz = range(0.0, 20.0, length=401)

function prob_func_Hz(prob, i, repeat)
    remake(prob, p=remake(prob.p, mp=remake(prob.p.mp, ω=frequencies_Hz[i])))
end

ensemble_prob_Hz = EnsembleProblem(prob_4mT, prob_func=prob_func_Hz)
sim_Hz = solve(
    ensemble_prob_Hz, Tsit5(), EnsembleThreads(),
    reltol=reltol, abstol=abstol, trajectories=size(frequencies_Hz)[1]
)

# Visualisations
function get_sim_heatmap(sim, xs)
    hcat(map(x -> hcat(sim(x)...)[1, :], xs)...)
end

xs = 0:1:1800
clims = (0.1, 0.3)

fig8 = let
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1], xlabel=L"$t$, s", ylabel=L"$\omega$, π mHz")
    hm = heatmap!(ax,
        xs,
        frequencies_mHz * 1e3 / π,
        transpose(get_sim_heatmap(sim_mHz, xs)),
        colorrange=clims,
    )
    Colorbar(fig[:, end+1], hm)
    fig
end

save("Fig_8.pdf", fig8)

fig9 = let
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1], xlabel=L"$t$, s", ylabel=L"$B$, mT")
    hm = heatmap!(ax,
        xs,
        inductions * 1e3,
        transpose(get_sim_heatmap(sim_induction, xs)),
        colorrange=clims,
    )
    Colorbar(fig[:, end+1], hm)
    fig
end

save("Fig_9.pdf", fig9)

fig10 = let
    fig = Figure(size=(1000, 500))
    ax = Axis(fig[1, 1], xlabel=L"$t$, s", ylabel=L"$ω$, Hz")
    hm = heatmap!(ax,
        xs,
        frequencies_Hz,
        transpose(get_sim_heatmap(sim_Hz, xs)),
        colorrange=clims,
    )
    Colorbar(fig[:, end+1], hm)
    fig
end

save("Fig_10.pdf", fig10)
