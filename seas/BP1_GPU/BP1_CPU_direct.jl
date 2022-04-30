include("odefun.jl")
include("ops_stripped.jl")
include("friction_knls.jl")
using Plots
using SparseArrays
using LinearAlgebra
using CUDA
using DifferentialEquations
using IterativeSolvers

    # number of grid points in each dimension
    Nr = 2^10
    Ns = 2^10
    stride_space = 1 # write-out every stride_space grid points along fault
    stride_time = 5
    Lx = 80
    Ly = 80

    sim_years = 1e-6

    Vp = 1e-9 # plate rate
    ρ = 2.670
    cs = 3.464 / Ly
    μ = cs^2 * ρ 
    σn = 50 / Ly
    RSamin = 0.01
    RSamax = 0.025
    RSb = 0.015
    RSDc = 0.032
    RSf0 = 0.6
    RSV0 = 1e-6
    RSVinit = 1e-9
    RSH1 = 15/Ly
    RSH2 = 18/Ly
    RSWf = 40/Ly

    μshear = cs^2 * ρ
    η = μshear / (2 * cs)

    # SBP interior order
    SBPp   = 2

    r = Array(LinRange(0, 1, Nr+1))  
    s = Array(LinRange(0, 1, Ns+1))

    # create operators
    (M̃, F, τ, H̃, HfI_FT) = get_operators(SBPp, Nr, Ns, μ)
    # factor with Cholesky
    M = cholesky(Symmetric(M̃))
    # initialize vector g that stores boundary data
    t = 0
    g = zeros((Nr+1) * (Ns+1))
    δ = zeros(Ns+1)
  

    bc_Dirichlet = (lf, x, y) -> (2-lf) .* (0 * x .+ 0 .* y) + (lf-1) .* (0 .* x .+ 0 .* y)
    bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
   
    bdry_vec_mod!(g, F, τ, r, s, bc_Dirichlet, bc_Neumann)
    
    u = M \ g
    
    yf = s # just s-variable here
    (mm, δNp) = findmin(abs.(RSWf .- yf))
    @assert yf[δNp] ≈ RSWf
  
    Δτ = zeros(Ns+1)

    # Assemble fault variables/data
    RSa = zeros(δNp)
    for n = 1:δNp
        RSa[n] = RSamin - (RSamin - RSamax) *
          min(1, max(0, (RSH1 - yf[n])/(RSH1 - RSH2)))
    end

    τz0 = σn * RSamax * asinh(RSVinit / (2 * RSV0) *
                                 exp.((RSf0 + RSb * log.(RSV0 / RSVinit)) /
                                      RSamax)) + η * RSVinit

    θ = (RSDc ./ RSV0) .* exp.((RSa ./ RSb) .* log.((2 .* RSV0 ./ RSVinit) .*
      sinh.((τz0 .- η .* RSVinit) ./ (RSa .* σn))) .- RSf0 ./ RSb)

    ψ0 = RSf0 .+ RSb .* log.(RSV0 .* θ ./ RSDc)


    ψδ = zeros(δNp + Ns + 1)  #because length(ψ) = δNp,  length(δ) = N+1
    ψδ[1:δNp] .= ψ0


    stations = yf[1:stride_space:δNp] .* Ly

    u_old = copy(u)

    #write out initial data
    no_stored_time_steps = Integer(1e6);
    vv = Array{Float64, 2}(undef, no_stored_time_steps, 2 + length(stations))
    vv[1, 1] = t
    vv[1, 2] = RSVinit
    vv[1, 3:end] = δ[1:stride_space:δNp]

  # set up parameters sent to right hand side
  odeparam = (
              Vp=Vp,
              M = M,
              u=u,
              u_old=u_old,
              Δτ = Δτ,
              g = g,
              μshear=μshear,
              RSa=RSa,
              RSb=RSb,
              σn=σn,
              η=η,
              RSV0=RSV0,
              τz0=τz0,
              RSDc=RSDc,
              RSf0=RSf0,
              δNp = δNp,
              N = Ns,
              F = F,
              τ = τ,
              r = r ,
              s = s,
              HfI_FT = HfI_FT,
              vv = vv,
              save_stride_fields = stride_time # save every save_stride_fields time steps
             )

  
  tspan = (0, sim_years * year_seconds)
  prob = ODEProblem(odefun_CPU_direct, ψδ, tspan, odeparam)
  
 
  cb_mod = SavingCallback((ψδ, t, i) -> write_text_slip_CPU(ψδ, t, i, yf, stations, odeparam, "BP1_", 10 * year_seconds), SavedValues(Float64, Float64))

  
 

 @time sol = solve(prob, Tsit5(); dt=0.01,
              atol = 1e-14, rtol = 1e-14, save_everystep=true, gamma = 0.2,
              internalnorm=(x, _)->norm(x, Inf), callback=cb_mod);

5

 

