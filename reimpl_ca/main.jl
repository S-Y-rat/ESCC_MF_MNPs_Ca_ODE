using DifferentialEquations, Plots, reimpl_ca

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
    ensemble_prob_mHz, Tsit5(), EnsembleThreads(),
    reltol=reltol, abstol=abstol, trajectories=size(frequencies_mHz)[1]
)

# Plot 2
inductions = range(0.0, 100.0e-3, length=401)

function prob_func_induction(prob, i, repeat)
    remake(prob, p=remake(prob.p, mp=remake(prob.p.mp, B=inductions[i])))
end

ensemble_prob_induction = EnsembleProblem(prob, prob_func=prob_func_induction)
sim_induction = solve(
    ensemble_prob_induction, Tsit5(), EnsembleThreads(),
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

plt = heatmap(
    xs, range(0, 1.7, length=size(frequencies_mHz)[1]), get_sim_heatmap(sim_mHz, xs),
    clims=clims,
    yticks=range(0, 1.7, 3),
    xaxis="\$t\$, s", yaxis="\$ω\$, π mHz", dpi=600
)

savefig(plt, "Fig_8.png")

plt = heatmap(
    xs, range(0.0, 100.0, length=size(inductions)[1]), get_sim_heatmap(sim_induction, xs),
    clims=clims,
    xaxis="\$t\$, s", yaxis="\$B\$, mT", dpi=600
)

savefig(plt, "Fig_9.png")

plt = heatmap(
    xs, frequencies_Hz, get_sim_heatmap(sim_Hz, xs),
    clims=clims,
    xaxis="\$t\$, s", yaxis="\$ω\$, π Hz", dpi=600
)

savefig(plt, "Fig_10.png")
