include("odefun.jl")
include("ops_GPU.jl")
include("friction_knls.jl")
using Plots
using SparseArrays
using LinearAlgebra
using CUDA
using DifferentialEquations
using IterativeSolvers
 

    # number of grid points in each dimension
    Nr = 2^6
    Ns = 2^6 
    stride_space = 1 # write-out every stride_space grid points along fault
    stride_time = 1 #only currently saves correctly if equal to 1
    Lx = 80
    Ly = 80
    sim_years = 300.

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

    r_h = Array(LinRange(0, 1, Nr+1))  
    s_h = Array(LinRange(0, 1, Ns+1))

    # create operators
    (M_h, F_h, τ_h, H̃, HfI_FT_h) = get_operators(SBPp, Nr, Ns, μ)
    M_d = CUSPARSE.CuSparseMatrixCSC(M_h)
    HfI_FT_d  = (CUSPARSE.CuSparseMatrixCSC(HfI_FT_h[1]), CUSPARSE.CuSparseMatrixCSC(HfI_FT_h[2]),CUSPARSE.CuSparseMatrixCSC(HfI_FT_h[3]), CUSPARSE.CuSparseMatrixCSC(HfI_FT_h[4]))
    F_d  = (CUSPARSE.CuSparseMatrixCSC(F_h[1]), CUSPARSE.CuSparseMatrixCSC(F_h[2]),CUSPARSE.CuSparseMatrixCSC(F_h[3]), CUSPARSE.CuSparseMatrixCSC(F_h[4]))

 
    t = 0
    g_h = zeros((Nr+1) * (Ns+1))
    g_d = CuArray(convert(Array{Float64, 1}, g_h))
    u_h = zeros(size(g_h))
    cuarry_ones = CuArray(convert(Array{Float64, 1}, ones(Ns+1)))

    δ_h = zeros(Ns+1)
    δ_d = CuArray(convert(Array{Float64, 1}, δ_h))

    bc_Dirichlet = (lf, x, y) -> (2-lf) * (0 * x + 0 * y) + (lf-1) * (0 * x + 0 * y)
    bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
   
    #bdry_vec_mod!(g_h, F_h, τ_h, r_h, s_h, bc_Dirichlet, bc_Neumann)
    bdry_vec_mod_GPU!(g_d, F_d, δ_d, Vp, t, cuarry_ones)

    τ_d  = (CUSPARSE.CuSparseMatrixCSC(τ_h[1]), CUSPARSE.CuSparseMatrixCSC(τ_h[2]),CUSPARSE.CuSparseMatrixCSC(τ_h[3]), CUSPARSE.CuSparseMatrixCSC(τ_h[4]))

    r_d = CuArray(convert(Array{Float64, 1}, r_h))
    s_d = CuArray(convert(Array{Float64, 1}, s_h))

    u_d = CuArray(convert(Array{Float64, 1}, u_h))

    #cg!(u_h, M_h, g_h)
    cg!(u_d, M_d, g_d)



    # initialize 
    t = 0
  
    yf = s_h # just s-variable here
    (mm, δNp) = findmin(abs.(RSWf .- yf))
    @assert yf[δNp] ≈ RSWf

 
   Δτ_h = zeros(Ns+1)
   Δτ_d = CuArray(convert(Array{Float64, 1},  Δτ_h))

  computetraction_stripped_GPU!(Δτ_d, HfI_FT_d, τ_d, 1, u_d, δ_d)

   # Assemble fault variables/data
   RSa_h = zeros(δNp)
   for n = 1:δNp
       RSa_h[n] = RSamin - (RSamin - RSamax) *
         min(1, max(0, (RSH1 - yf[n])/(RSH1 - RSH2)))
   end

   τz0 = σn * RSamax * asinh(RSVinit / (2 * RSV0) *
                                exp.((RSf0 + RSb * log.(RSV0 / RSVinit)) /
                                     RSamax)) + η * RSVinit

   θ = (RSDc ./ RSV0) .* exp.((RSa_h ./ RSb) .* log.((2 .* RSV0 ./ RSVinit) .*
     sinh.((τz0 .- η .* RSVinit) ./ (RSa_h .* σn))) .- RSf0 ./ RSb)
   ψ0 = RSf0 .+ RSb .* log.(RSV0 .* θ ./ RSDc)

   ψδ_h = zeros(δNp + Ns + 1)  #because length(ψ) = δNp,  length(δ) = N+1
   ψδ_h[1:δNp] .= ψ0

  RSa_d = CuArray(convert(Array{Float64, 1}, RSa_h))

 
#u_old_h = copy(u_h)
u_old_d = copy(u_d)


 no_stored_time_steps = Int(1e6);
 vv  = CuArray(convert(Array{Float64, 2}, zeros(no_stored_time_steps, length(δ_d))))
 vv[1, :] = δ_d


 # set up parameters sent to right hand side
 odeparam = (
             Vp=Vp,
             M = M_d,
             u=u_d,
             u_old=u_old_d,
             Δτ = Δτ_d,
             g = g_d,
             μshear=μshear,
             RSa=RSa_d,
             RSb=RSb,
             σn=σn,
             η=η,
             RSV0=RSV0,
             τz0=τz0,
             RSDc=RSDc,
             RSf0=RSf0,
             δNp = δNp,
             N = Ns,
             F = F_d,
             τ = τ_d,
             r = r_d ,
             s = s_d,
             HfI_FT = HfI_FT_d,
             save_stride_fields = stride_time, # save every save_stride_fields time steps
             vv = vv,
 	     cuarry_ones = cuarry_ones
  	     )

 stations = yf .* Ly

 station_indices = find_station_index(stations, yf .* Ly)
 ODEresults = ODE_results([], [], [], Dict(i => [] for i = 1:length(stations)))
 cb_mod = SavingCallback((ψδ, t, i) -> write_text_slip(ψδ, t, i, yf, stations, station_indices, odeparam, "BP1_", 10 * year_seconds), SavedValues(Float64, Float64))

open("devol.txt", "w") do io
   write(io,"0.0 0.0 ")
   write(io,"\n")
end



  ψδ_d = CuArray(convert(Array{Float64, 1}, ψδ_h))

  tspan = (0, sim_years * year_seconds)
   prob = ODEProblem(odefun, ψδ_d, tspan, odeparam)

 sol = solve(prob, Tsit5(); dt=0.01,
             atol = 1e-14, rtol = 1e-14, save_everystep=true, gamma = 0.2,
             internalnorm=(x, _)->norm(x, Inf), callback=cb_mod)


vv_c = Array(vv)
