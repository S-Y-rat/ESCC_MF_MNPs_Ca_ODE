module reimpl_ca

struct MagneticField
    Ms::Float64
    B::Float64
    ω::Float64
    N_mn::Float64
    MagneticField(; Ms=510e3, B=25e-3, ω=1.7e-3π, N_mn=10) = new(Ms, B, ω, N_mn)
end

τ_magn(mp::MagneticField, t) = π / (6mp.N_mn) * mp.Ms * mp.B * cos(mp.ω * t)

struct CalciumModel
    mp::MagneticField
    k_f::Float64
    V_p::Float64
    K::Float64
    k_p::Float64
    δ::Float64
    α_0::Float64
    α_1::Float64
    K_e::Float64
    V_pm::Float64
    K_pm::Float64
    γ::Float64
    K_h::Float64
    K_τ::Float64
    τ_max::Float64
    β_p::Float64
    p_s::Float64
    k_β::Float64
    K_c::Float64
    q_max::Float64
    CalciumModel(
        ; mp, k_f=3.9, V_p=0.9, K=1.9e-5, k_p=0.2, δ=1.5, α_0=2.7e-3, α_1=0.385, K_e=8, V_pm=0.11, K_pm=0.3,
        γ=5.5, K_h=0.08, K_τ=0.1, τ_max=1420.0, β_p=27e-3, p_s=0.1, k_β=0.4, K_c=0.2, q_max=0.04
    ) = new(mp, k_f, V_p, K, k_p, δ, α_0, α_1, K_e, V_pm, K_pm, γ, K_h, K_τ, τ_max, β_p, p_s, k_β, K_c, q_max)
end

function dSdt(u, cp::CalciumModel, t)
    c, c_e, h, p = u

    h_∞ = cp.K_h^4 / (cp.K_h^4 + c^4)
    h̄_α = h_∞
    τ_h = cp.τ_max * cp.K_τ^4 / (cp.K_τ^4 + c^4)
    m̄_α = c^4 / (cp.K_c^4 + c^4)
    m̄_β = m̄_α
    B = p^2 / (cp.k_p^2 + p^2)
    A = 1 - B
    α = A * (1 - m̄_α * h̄_α)
    β = B * m̄_β * h
    P_o = β / (β + cp.k_β * (β + α))

    J_serca = cp.V_p * (c^2 - cp.K * c_e^2) / (c^2 + cp.k_p^2)
    J_in = cp.α_0 + cp.α_1 * cp.K_e^4 / (cp.K_e^4 + c_e^4)
    J_pm = cp.V_pm * c^2 / (cp.K_pm^2 + c^2)
    J_IP3R = cp.k_f * P_o * (c_e - c)

    dcdt = J_IP3R - J_serca + cp.δ * (J_in - J_pm)
    dc_edt = cp.γ * (J_serca - J_IP3R)
    dhdt = (h_∞ - h) / τ_h
    dpdt = cp.β_p * (cp.p_s - p)

    [dcdt, dc_edt, dhdt, dpdt]
end

function dWdt(u, cp::CalciumModel, t)
    α_P = 1e6
    f_e_P = 0.0134
    T_e_P = 310.0
    k_P = 1.3807e-23
    N_ch = 1e12
    l_P = 3.5e-5
    ε_P = 0.1
    δ_P = 1e-5

    τ = τ_magn(cp.mp, t)
    W_denom = ε_P * τ * l_P + √(16δ_P^2 + ε_P^2 * τ^2 * l_P^2)
    W = (W_denom - 4δ_P)^2 / (8W_denom)

    J_magn = cp.q_max / (1 + α_P * exp(-f_e_P * W / (k_P * T_e_P * N_ch)))
    dcdt, dc_edt, dhdt, dpdt = dSdt(u, cp, t)

    [dcdt + J_magn, dc_edt, dhdt, dpdt]
end

end # module reimpl_ca
