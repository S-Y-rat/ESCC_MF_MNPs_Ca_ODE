using CairoMakie
include("./src/reimpl_ca.jl")
using OrdinaryDiffEq
using CairoMakie, LaTeXStrings
using Random, Distributions, Statistics
using ColorSchemes

set_theme!(theme_latexfonts(), linewidth=2)
const color = ColorSchemes.seaborn_colorblind

const reltol, abstol = 1e-8, 1e-8
const u0 = [0.1, 100.0, 0.08, 0.1]
const tspan = (0.0, 1800.0)

const mp = reimpl_ca.MagneticField()
const model_no_mf = reimpl_ca.CalciumModel(mp=remake(mp, B=0))
const model_mf = reimpl_ca.CalciumModel(mp=mp)
const prob1 = ODEProblem(reimpl_ca.dWdt, u0, tspan, model_no_mf)
const prob2 = ODEProblem(reimpl_ca.dWdt, u0, tspan, model_mf)

const ts = 0.0:0.1:1800.0
const us1 = solve(prob1, DP5(), reltol=reltol, abstol=abstol)(ts).u
const us2 = solve(prob2, DP5(), reltol=reltol, abstol=abstol)(ts).u

const label_no_mf = L"$B=0.0$ mT, $\omega=1.7\cdot\pi$ mHz"
const label_mf = L"$B=25.0$ mT, $\omega=1.7\cdot\pi$ mHz"

function fig3()
    fig = Figure(size=(1000, 500))
    Label(fig[0, 1:2],
        L"$c(t)$ is $Ca^{2+}$ concentration in the cytosol",
        fontsize=24,
    )
    ax1 = Axis(fig[1, 1],
        title="A",
        titlesize=22,
        limits=((-3, 25), nothing)
    )
    ax2 = Axis(fig[1, 2],
        title="B",
        titlesize=22,
        limits=((100, 1810), (0.0, 0.58))
    )
    Label(fig[2, 1:2], L"$t$, s", fontsize=20)
    Label(fig[1:2, 0], L"$c(t)$, $\mu M$", fontsize=20, rotation=π/2)
    ys1 = getindex.(us1, 1)
    ys2 = getindex.(us2, 1)

    lines!(ax1, ts, ys1, color=color[1])
    lines!(ax1, ts, ys2, color=color[2], linestyle=:dash)
    lines!(ax2, ts, ys1, color=color[1], label=label_no_mf)
    lines!(ax2, ts, ys2, color=color[2], linestyle=:dash, label=label_mf)

    axislegend(ax2, position=:rt, framecolor=(:grey, 0.5))
    colsize!(fig.layout, 1, Relative(0.25))
    colsize!(fig.layout, 2, Relative(0.75))
    fig
end

function fig4()
    fig = Figure(size=(1000, 500))
    Label(fig[0, 1:2], L"$c_e(t)$ is $Ca^{2+}$ concentration in ER", fontsize=24)
    ax1 = Axis(fig[1, 1],
        title="A",
        yscale=log10,
        titlesize=22,
        limits=((-3, 400), nothing),
    )
    ax2 = Axis(fig[1, 2],
        title="B",
        titlesize=22,
        limits=((400, 1810), (15, 22))
    )
    Label(fig[2, 1:2], L"$t$, s", fontsize=20)
    Label(fig[1:2, 0], L"$c_e(t)$, $\mu M$", fontsize=20, rotation=π/2)
    ys1 = getindex.(us1, 2)
    ys2 = getindex.(us2, 2)

    lines!(ax1, ts, ys1, color=color[1])
    lines!(ax1, ts, ys2, color=color[2], linestyle=:dash)
    lines!(ax2, ts, ys1, color=color[1], label=label_no_mf)
    lines!(ax2, ts, ys2, color=color[2], linestyle=:dash, label=label_mf)

    axislegend(ax2, position=:rt, framecolor=(:grey, 0.5))
    colsize!(fig.layout, 1, Relative(0.33))
    colsize!(fig.layout, 2, Relative(0.67))
    fig
end

function full_in(cp::reimpl_ca.CalciumModel, c_e, t)
    cp.δ * map(x -> reimpl_ca.J_in(cp, x),
        c_e
    ) + map(x -> reimpl_ca.J_magn(cp, x), t)
end

function fig5()
    fig = Figure(size=(1000, 500))
    Label(fig[0, 1],
        L"$Ca^{2+}$ influx from outside of the cell",
        fontsize=22,
    )
    ax = Axis(fig[1, 1],
        xlabel=L"$t$, s",
        xlabelsize=18,
        ylabelsize=18,
        ylabel=L"$v$, $\mu M$",
        limits=((-10, 1810), nothing)
    )
    ys1 = full_in(model_no_mf, getindex.(us1, 2), ts)
    ys2 = full_in(model_mf, getindex.(us2, 2), ts)

    lines!(ax, ts, ys1, label=label_no_mf, color=color[1])
    lines!(ax, ts, ys2, linestyle=:dash, color=color[2], label=label_mf)

    axislegend(ax, position=:rb, framecolor=(:grey, 0.5))
    colsize!(fig.layout, 1, Relative(1.0))
    fig
end

function local_max(arr)
    left_slice = arr[1:end-2]
    center_slice = arr[2:end-1]
    right_slice = arr[3:end]
    mask = falses(size(arr))
    mask[2:end-1] = (left_slice .< center_slice) .& (center_slice .> right_slice)
    mask[1] = arr[1] > arr[2]
    mask[end] = arr[end] > arr[end-1]
    arr[mask]
end

function local_min(arr)
    left_slice = arr[1:end-2]
    center_slice = arr[2:end-1]
    right_slice = arr[3:end]
    mask = falses(size(arr))
    mask[2:end-1] = (left_slice .> center_slice) .& (center_slice .< right_slice)
    mask[1] = arr[1] < arr[2]
    mask[end] = arr[end] < arr[end-1]
    arr[mask]
end

function fig6()
    alpha = 0.2
    fig = Figure(size=(1000, 600))
    ax11 = Axis(fig[1, 1], title="A", limits=(nothing, (1.22, 1.6)))
    ax12 = Axis(fig[2:3, 1], xlabel="Local\nMaxima", limits=(nothing, (0.0, 0.52)))
    ax2 = Axis(fig[1:3, 2],
        title="B",
        xlabel="Local\nMinima",
        limits=(nothing, (0.0, nothing)),
    )
    ax31 = Axis(fig[1, 3], title="C", limits=(nothing, (1.22, 1.6)))
    ax32 = Axis(fig[2:3, 3], xlabel="MF", limits=(nothing, (0.0, 0.52)))
    Label(fig[0, 1:3], "Local extrema and median regressions", fontsize=20)
    Label(fig[1:3, 0], L"$C$, $\mu M$", rotation=π/2)

    max_no_mf = local_max(getindex.(us1, 1))
    min_no_mf = local_min(getindex.(us1, 1))
    max_mf = local_max(getindex.(us2, 1))
    min_mf = local_min(getindex.(us2, 1))
    rng = () -> MersenneTwister(56353)
    d = Uniform(-0.25, 0.25)

    function linear_coeffs((x1, y1), (x2, y2))
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        "$(round(slope, digits=4))x+$(round(intercept, digits=4))"
    end

    function column1!(ax::Axis)
        scatter!(ax,
            zeros(size(max_no_mf)) + rand(rng(), d, size(max_no_mf)),
            max_no_mf,
            alpha=alpha,
            color=ColorSchemes.Set1_5[1],
        )
        scatter!(ax,
            ones(size(max_mf)) + rand(rng(), d, size(max_mf)),
            max_mf,
            alpha=alpha,
            color=ColorSchemes.Set1_5[2],
        )
    end

    function regression!(ax::Axis, y1::Number, y2::Number)
        scatterlines!(ax,
            0:1,
            [y1, y2],
            label=linear_coeffs((0.0, y1), (1.0, y2)),
            markersize=15,
            color=ColorSchemes.Set1_5[5],
        )
    end

    column1!(ax11)
    linkxaxes!(ax11, ax12)
    hidexdecorations!(ax11, grid = false)
    column1!(ax12)
    regression!(ax12, median(max_no_mf), median(max_mf))
    ax12.xticks = (0:1, ["No MF", "MF"])
    axislegend(ax12, position=:lb, framecolor=(:grey, 0.5))

    scatter!(ax2,
        zeros(size(min_no_mf)) + rand(rng(), d, size(min_no_mf)),
        min_no_mf,
        alpha=alpha,
        color=ColorSchemes.Set1_5[3],
    )
    scatter!(ax2,
        ones(size(min_mf)) + rand(rng(), d, size(min_mf)),
        min_mf,
        alpha=alpha,
        color=ColorSchemes.Set1_5[4],
    )
    regression!(ax2, median(min_no_mf), median(min_mf))
    ax2.xticks = (0:1, ["No MF", "MF"])
    axislegend(ax2, position=:lb, framecolor=(:grey, 0.5))

    function column3!(ax::Axis)
        scatter!(ax,
            zeros(size(max_mf)) + rand(rng(), d, size(max_mf)),
            max_mf,
            alpha=alpha,
            color=ColorSchemes.Set1_5[2],
        )
        scatter!(ax,
            ones(size(min_mf)) + rand(rng(), d, size(min_mf)),
            min_mf,
            alpha=alpha,
            color=ColorSchemes.Set1_5[4],
        )
    end

    column3!(ax31)
    linkxaxes!(ax31, ax32)
    hidexdecorations!(ax31, grid=false)
    column3!(ax32)
    regression!(ax32, median(max_mf), median(min_mf))
    ax32.xticks = (0:1, ["Local\nMaxima", "Local\nMinima"])
    axislegend(ax32, position=:lb, framecolor=(:grey, 0.5))

    fig
end

begin
    fig = fig3()
    save("Fig_3.pdf", fig)
    save("Fig_3.svg", fig)
end

begin
    fig = fig4()
    save("Fig_4.pdf", fig)
    save("Fig_4.svg", fig)
end

begin
    fig = fig5()
    save("Fig_5.pdf", fig)
    save("Fig_5.svg", fig)
end

begin
    fig = fig6()
    save("Fig_6.pdf", fig)
    save("Fig_6.svg", fig)
end
