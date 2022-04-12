include("odefun.jl")
using Plots
using SparseArrays
using LinearAlgebra
using DelimitedFiles

include("ops_stripped.jl")
 

function main()


     # number of grid points in each dimension
     Nr = 2^9
     Ns = 2^9
     stride_space = 10 # write-out every stride_space grid points along fault
     stride_time = 5
    Lx = 80
    Ly = 80

    sim_years = 1500.

    Vp = 1e-9 # plate rate
    ρ = 2.670
    cs = 3.464 / Ly
    μ = cs^2 * ρ 
    σn = 50 / Ly
    RSamin = 0.01
    RSamax = 0.025
    RSb = 0.015
    RSDc = 0.016
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
  
    #gL = sinh(0) .* cos.(π .* s)
    #gR = sinh(π) .* cos.(π .* s)
    #bc_Dirichlet = (lf, x, y) -> (2-lf) .* sinh(π .* x) .* cos.(π .* y) + (lf-1) .* sinh(π .* x) .* cos.(π .* y)
    bc_Dirichlet = (lf, x, y) -> (2-lf) .* (0 * x .+ 0 .* y) + (lf-1) .* (0 .* x .+ 0 .* y)
    bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
   
    bdry_vec_mod!(g, F, τ, r, s, bc_Dirichlet, bc_Neumann)


     u = M \ g
    
    #=
     uexact = zeros((Nr+1), (Ns+1))
     for i = 1:Nr+1
        for j = 1:Ns+1
          uexact[i, j] = sinh(π * r[i]) * cos(π * s[j])
        end
      end
   
      uex = uexact[:]
      ru = reshape(u, Nr+1, Ns+1)
      #display(surface(r, s, ru))
      err = uex - u
      @show sqrt(err' * H̃ * err)
    
      ddddd
    =#

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


    function find_station_index(stations, grid_points)
      numstations = length(stations)
      @show numstations
      station_ind = zeros(numstations)
      @show station_ind
      for i in range(1, stop=numstations)
        @show argmin(abs.(grid_points .- stations[i]))
        station_ind[i] = argmin(abs.(grid_points .- stations[i]))
        # station_ind[i] = abs.(grid_points .- stations[i])[1]
      end
      return Integer.(station_ind)
    end

    #stations = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35] ./ Ly
    stations = yf[1:stride_space:δNp] .* Ly
    station_indices = find_station_index(stations, yf .* Ly)

    u_old = copy(u)

  # set up parameters sent to right hand side
  odeparam = (reject_step = [false],
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
              save_stride_fields = stride_time # save every save_stride_fields time steps
             )

  dψV = zeros(δNp + Ns + 1)
  tspan = (0, sim_years * year_seconds)
  prob = ODEProblem(odefun, ψδ, tspan, odeparam)
  
  function stepcheck(_, p, _)
    if p.reject_step[1]
      p.reject_step[1] = false
      println("reject")
      return true
    end
    return false
  end

  ODEresults = ODE_results([], [], [], Dict(i => [] for i = 1:length(stations)))

  cb = SavingCallback((ψδ, t, i) -> saveslip(ψδ, t, i, ODEresults, yf, stations, station_indices, odeparam, "BP1_", 10 * year_seconds), SavedValues(Float64, Float64))
  cbstop = SavingCallback((ψδ, t, i) -> savestop(ψδ, t, i, ODEresults, odeparam), SavedValues(Float64, Float64))
  cb_mod = SavingCallback((ψδ, t, i) -> write_text_slip(ψδ, t, i, yf, stations, station_indices, odeparam, "BP1_", 10 * year_seconds), SavedValues(Float64, Float64))

  open("devol.txt", "w") do io
    write(io,"0.0 0.0 ")
        for i in 1:length(stations)
          write(io,"$(stations[i]) ")
        end
        write(io,"\n")
  end

  #write out initial data
  vv = Array{Float64}(undef, 1, 2+length(stations))
      vv[1] = t
      vv[2] = log10(RSVinit)
      vv[3:end] = δ[station_indices]
      open("devol.txt", "a") do io
         writedlm(io, vv)
      end


  sol = solve(prob, Tsit5(); isoutofdomain=stepcheck, dt=year_seconds,
              atol = 1e-14, rtol = 1e-14, save_everystep=true,
              internalnorm=(x, _)->norm(x, Inf), callback=cb_mod)

  #@show ODEresults.t_list
 
  return (sol, yf, δNp)
end

function plot_slip(S, δNp, yf, stride_time)

  m = length(yf)
  no_time_steps = size(S.t)
  slip_final = S.u[end][end]

  for i = 1:stride_time:no_time_steps[1]

    slip_t = S.u[i][δNp+1:end] # slip at time t
    #pyplot()
    display(plot(slip_t, -yf, xtickfont=font(18),
    ytickfont=font(18),
    guidefont=font(18),
    legendfont=font(18), ylabel = "Depth (km)", xlabel = "Slip (m)", xlims = (0, slip_final)))
    sleep(0.1)
  end

  #nothing
end

(S, yf, δNp) = main()
#plot_slip(S, δNp, yf, 10)
