using CairoMakie: activate!
include("./src/reimpl_ca.jl")
using OrdinaryDiffEq
using CairoMakie, LaTeXStrings
using ColorSchemes

set_theme!(theme_latexfonts(), linewidth=2)

const reltol, abstol = 1e-8, 1e-8
const u0 = [0.1, 100.0, 0.08, 0.1]
const tspan = (0.0, 1800.0)

const mp = reimpl_ca.MagneticField()
const model_no_p = reimpl_ca.CalciumModel(mp=nothing)
const model_no_mf = reimpl_ca.CalciumModel(mp=remake(mp, B=0))
const model_mf = reimpl_ca.CalciumModel(mp=mp)
const prob0 = ODEProblem(reimpl_ca.dSdt, u0, tspan, model_no_p)
const prob1 = ODEProblem(reimpl_ca.dWdt, u0, tspan, model_no_mf)
const prob2 = ODEProblem(reimpl_ca.dWdt, u0, tspan, model_mf)

const ts = 0.0:0.1:1800.0
const us0 = solve(prob0, DP5(), reltol=reltol, abstol=abstol)(ts).u
const us1 = solve(prob1, DP5(), reltol=reltol, abstol=abstol)(ts).u
const us2 = solve(prob2, DP5(), reltol=reltol, abstol=abstol)(ts).u
const us2_tsit5 = solve(prob2, Tsit5(), reltol=reltol, abstol=abstol)(ts).u

const fig_cmp = let
    fig = Figure(size=(900, 900))
    Label(fig[0, 1:2],
        L"$Ca^{2+}$ dynamics of Chang model and modified one",
        fontsize=22,
    )
    ax11 = Axis(fig[1, 1],
        title="A",
        ylabel=L"$c(t)$, $\mu M$",
        titlesize=20,
        ylabelsize=18,
    )
    ax12 = Axis(fig[2, 1],
        title="C",
        ylabel=L"$c(t)$, $\mu M$",
        titlesize=20,
        ylabelsize=18,
    )
    ax13 = Axis(fig[3, 1],
        title="E",
        ylabel=L"$\Delta c(t)$, $\mu M$",
        titlesize=20,
        ylabelsize=18,
        limits=((-0.0001, nothing), nothing),
    )
    ax21 = Axis(fig[1, 2],
        title="B",
        yscale=log10,
        ylabel=L"$c_e(t)$, $\mu M$",
        titlesize=20,
        ylabelsize=18,
    )
    ax22 = Axis(fig[2, 2],
        title="D",
        yscale=log10,
        ylabel=L"$c_e(t)$, $\mu M$",
        titlesize=20,
        ylabelsize=18,
    )
    ax23 = Axis(fig[3, 2],
        title="F",
        ylabel=L"$\Delta c_e(t), \mu M$",
        titlesize=20,
        ylabelsize=18,
        limits=((-0.003, nothing), nothing),
    )
    Label(fig[4, 1:2], L"$t$, s", fontsize=18)
    ys10 = getindex.(us0, 1)
    ys11 = getindex.(us1, 1)
    ys20 = getindex.(us0, 2)
    ys21 = getindex.(us1, 2)
    lines!(ax11, ts, ys10)
    lines!(ax12, ts, ys11)
    scatter!(ax13, ts, ys11 - ys10, markersize=3)
    lines!(ax21, ts, ys20)
    lines!(ax22, ts, ys21)
    scatter!(ax23, ts, ys21 - ys20, markersize=3)
    fig
end

const fig_int = let
    fig = Figure(size=(1000, 500))
    Label(fig[0, 1:2],
          L"Differenses in numerical solving of $Ca^{2+}$ dynamics",
          fontsize=20,
    )
    ax1 = Axis(fig[1, 1],
        title="A",
        titlesize=18,
        ylabel=L"$\Delta c(t)$, $\mu M$",
        ylabelsize=16,
    )
    ax2 = Axis(fig[1, 2],
        title="B",
        titlesize=18,
        ylabel=L"$\Delta c_e(t)$, $\mu M$",
        ylabelsize=16,
    )
    scatter!(ax1,
        ts,
        getindex.(us2_tsit5, 1) - getindex.(us2, 1),
        markersize=3,
    )
    scatter!(ax2,
        ts,
        getindex.(us2_tsit5, 2) - getindex.(us2, 2),
        markersize=3,
    )
    Label(fig[2, 1:2], L"$t$, s", fontsize=16)
    fig
end

Δus = [
    [1.5, -15.0, 0.0, 0.0],
    [-0.5, -30.0, 0.0, 0.0],
    [0.75, -45.0, 0.0, 0.0],
    [0.5, -60.0, 0.0, 0.0],
    [-0.5, -75.0, 0.0, 0.0],
    [1.5, -100.0, 0.0, 0.0],
]

function prob_func(prob, i, repeat)
    remake(prob, u0=u0 + Δus[i])
end

function fig_traj(prob; y1=18, y2=27)
    fig = Figure(size=(1000, 500))
    ax1 = Axis(fig[1, 1],
        xlabel=L"$c(t)$, $\mu M$",
        ylabel=L"$c_e(t)$, $\mu M$",
        title="Full view",
    )
    ax2 = Axis(fig[1, 2],
        xlabel=L"$c(t)$, $\mu M$",
        ylabel=L"$c_e(t)$, $\mu M$",
        limits=((0.0, 0.5), (y1, y2)),
        title="Zoomed in",
    )

    ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    sim = solve(
        ensemble_prob, DP5(), EnsembleThreads(),
        reltol=reltol, abstol=abstol, trajectories=size(Δus)[1]
    )
    for (s, Δu, col) in zip(sim(ts), Δus, ColorSchemes.Dark2_6)
        c = getindex.(s.u, 1)
        c_e = getindex.(s.u, 2)
        lines!(ax1, c, c_e, color=col, alpha=0.5)
        lines!(ax2,
            c,
            c_e,
            color=col,
            label=latexstring("\$Δc=$(Δu[1])\$ \$μM\$, \$Δc_e=$(Δu[2])\$ \$μM\$"),
            alpha=0.5,
        )
        scatter!(ax1, c[1], c_e[1], color=col)
    end
    axislegend(ax2, position=:rt, framecolor=(:grey, 0.5))
    fig
end

save("Fig_cmp.pdf", fig_cmp)
save("Fig_cmp.svg", fig_cmp)

save("Fig_int.pdf", fig_int)
save("Fig_int.svg", fig_int)

begin
    fig = fig_traj(prob1; y1=15, y2=24)
    save("Fig_traj_no_mf.pdf", fig)
    save("Fig_traj_no_mf.svg", fig)
end

begin
    fig = fig_traj(prob2)
    save("Fig_traj_mf.pdf", fig)
    save("Fig_traj_mf.svg", fig)
end
