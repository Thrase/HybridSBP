const year_seconds = 31556926

global const ctr = Ref{Int64}(1) 

using OrdinaryDiffEq
using DiffEqCallbacks
using Printf
using Plots
using DelimitedFiles
using CUDA

function odefun_GPU_v1(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT
  cuarry_ones = p.cuarry_ones


  #@show t ./ 31556926


  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]
  δ_d = CuArray(convert(Array{Float64, 1}, δ))

  bdry_vec_mod_GPU!(g, F, δ_d, Vp, t, cuarry_ones)
  
  # solve for displacements everywhere in domain
  cg!(u, M, g)


  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]
  
  dψ .= 0
  V  .= 0


  # Update the fault data
  Δτ .= 0
  lf1 = 1


  computetraction_stripped_GPU!(Δτ, HfI_FT, τ, 1, u, δ_d)
  
  Δτ_h = Array(Δτ)

  for n = 1:δNp
    ψn = ψ[n]
    an = RSa[n]

    τn = Δτ_h[n] + τz0

    VR = abs(τn / η)
    VL = -VR
    Vn = V[n]
    obj_rs(V) = rateandstate(V, ψn, σn, τn, η, an, RSV0)
    (Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-9,
                                 atolx = 1e-9, rtolx = 1e-9)

    #@show iter 
  
    V[n] = Vn

    dψ[n] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(Vn) / RSV0)
   
  end
  
  
  V[δNp+1:N+1] .= Vp


  nothing
end




function odefun_GPU_v2(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT
  cuarry_ones = p.cuarry_ones


  #@show t ./ 31556926


  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]
  

  bdry_vec_mod_GPU!(g, F, δ, Vp, t, cuarry_ones)
  
  # solve for displacements everywhere in domain
  cg!(u, M, g)


  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]
  
  dψ .= 0
  V  .= 0


  # Update the fault data
  Δτ .= 0
  lf1 = 1


  computetraction_stripped_GPU!(Δτ, HfI_FT, τ, 1, u, δ)
  
  Δτ_h = Array(Δτ)
  ψ_h = Array(ψ)
  V_h = Array(V[1:δNp])
  dψ_h = Array(dψ)

  for n = 1:δNp
    ψn = ψ_h[n]
    an = RSa[n]

    τn = Δτ_h[n] + τz0

    VR = abs(τn / η)
    VL = -VR
    Vn = V_h[n]
    obj_rs(V) = rateandstate(V, ψn, σn, τn, η, an, RSV0)
    (Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-9,
                                 atolx = 1e-9, rtolx = 1e-9)

    #@show iter 
  
    V_h[n] = Vn

    dψ_h[n] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(Vn) / RSV0)
   
  end
  V_d = CuArray(convert(Array{Float64, 1}, V_h))
  dψ_d = CuArray(convert(Array{Float64, 1}, dψ_h))
  V[1:δNp] .= V_d
  dψ .= dψ_d
  
  V[δNp+1:N+1] .= Vp

  nothing
end





function odefun_GPU_with_for_loop(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT
  cuarry_ones = p.cuarry_ones


  #@show t ./ 31556926


  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]

  bdry_vec_mod_GPU!(g, F, δ, Vp, t, cuarry_ones)
  
  # solve for displacements everywhere in domain
  cg!(u, M, g)


  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]
  
  dψ .= 0
  V  .= 0


  # Update the fault data
  Δτ .= 0
  lf1 = 1


  computetraction_stripped_GPU!(Δτ, HfI_FT, τ, 1, u, δ)

  num_threads_per_block = 64
  num_blocks = cld(δNp, num_threads_per_block)
 @cuda threads=num_threads_per_block blocks=num_blocks knl_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)

 #@cuda threads=num_threads_per_block blocks=num_blocks knl_do_nothing_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)
synchronize()

# V[:] .= Vp
  V[δNp+1:N+1] .= Vp


  nothing
end


function odefun_CPU_CG(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT


  #@show t ./ 31556926

  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]

  bc_Dirichlet = (lf, x, y) -> (2-lf)*(0.5 * δ) + (lf-1)*fill(t * Vp/2, size(y))
  bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
  bdry_vec_mod!(g, F, τ, r, s, bc_Dirichlet, bc_Neumann)

  # solve for displacements everywhere in domain
  cg!(u, M, g)
  # uout, ch  = cg(M, g, log=true)
  # u[:] = uout
  #u[:] = M\g

 # @show ch.isconverged
  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]

  dψ .= 0
  V  .= 0

  # Update the fault data
  Δτ .= 0
  lf1 = 1

  Δτ .= -computetraction_stripped(HfI_FT, τ, lf1, u, δ)


  for n = 1:δNp
    ψn = ψ[n]
    an = RSa[n]

    τn = Δτ[n] + τz0

    #VR = abs(τn / η)
    #VL = -VR
    #Vn = V[n]
    #obj_rs(V) = rateandstate(V, ψn, σn, τn, η, an, RSV0)
    #(Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-9,
     #                            atolx = 1e-9, rtolx = 1e-9)

    #@show iter 
  
    V[n] = 2*RSV0*sinh(τn/(σn * an)) * exp(-ψn / an)
    Vn = V[n]
    #V[n] = Vn

    dψ[n] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(Vn) / RSV0)
   
  end
  
  


  V[δNp+1:N+1] .= Vp

  nothing
end




function odefun_CPU_direct(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT


 #@show t ./ 31556926

  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]

  bc_Dirichlet = (lf, x, y) -> (2-lf)*(0.5 * δ) + (lf-1)*fill(t * Vp/2, size(y))
  bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
  
  bdry_vec_mod!(g, F, τ, r, s, bc_Dirichlet, bc_Neumann)

  # solve for displacements everywhere in domain
  u[:] = M \ g


  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]

  dψ .= 0
  V  .= 0


  # Update the fault data
  Δτ .= 0
  lf1 = 1

  Δτ .= -computetraction_stripped(HfI_FT, τ, lf1, u, δ)
 
  for n = 1:δNp
    ψn = ψ[n]
    an = RSa[n]

    τn = Δτ[n] + τz0
  
    #VR = abs(τn / η)
    #VL = -VR
    #Vn = V[n]
    #obj_rs(V) = rateandstate(V, ψn, σn, τn, η, an, RSV0)
    #(Vn, _, iter) = newtbndv(obj_rs, VL, VR, Vn; ftol = 1e-9,
    #                             atolx = 1e-9, rtolx = 1e-9)

#@show iter

  V[n] = 2*RSV0*sinh(τn/(σn * an)) * exp(-ψn / an)
  Vn = V[n]
  #V[n] = Vn

    #V[n] = Vn

    dψ[n] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(Vn) / RSV0)
   
  end

  V[δNp+1:N+1] .= Vp

  nothing
end



function write_text_slip_GPU(ψδ,t,i,yf,stations,p,base_name="",tdump=100)
  Vmax = 0.0

  if isdefined(i,:fsallast) 
    δNp = p.δNp
    N = p.N
    dψV = i.fsallast
    dψ = @view dψV[1:δNp]
    V = @view dψV[δNp .+ (1:N+1)]
    Vmax = CUDA.maximum(V)
    δ = @view ψδ[δNp .+ (1:δNp)]
    ψ = @view ψδ[1:δNp]
    vv = p.vv
    tv = p.tv
        
    if mod(ctr[], p.save_stride_fields) == 0 || t == (sim_years ./ 31556926)
    
      tv[ctr[], 1] = t
      tv[ctr[], 2] = Vmax
      vv[ctr[], :] = δ
      
    end
    global ctr[] += 1
    #@show ctr[]
  end
    

     Vmax
 end


function odefun_CPU_CG_frictionloop(dψV, ψδ, p, t)
  
  Vp = p.Vp
  M = p.M
  u = p.u
  Δτ = p.Δτ
  g = p.g
  μshear = p.μshear
  RSa = p.RSa
  RSb = p.RSb
  σn = p.σn
  η = p.η
  RSV0 = p.RSV0
  τz0 = p.τz0
  RSDc = p.RSDc
  RSf0 = p.RSf0
  δNp = p.δNp
  N = p.N
  F = p.F
  τ = p.τ
  r = p.r 
  s = p.s
  HfI_FT = p.HfI_FT


 # @show t ./ 31556926

  ψ  = @view ψδ[        (1:δNp) ]
  δ  = ψδ[ δNp .+ (1:N+1) ]

  bc_Dirichlet = (lf, x, y) -> (2-lf)*(0.5 * δ) + (lf-1)*fill(t * Vp/2, size(y))
  bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
  bdry_vec_mod!(g, F, τ, r, s, bc_Dirichlet, bc_Neumann)

  # solve for displacements everywhere in domain
  cg!(u, M, g)


  # set up rates of change for  state and slip
  dψ = @view dψV[       (1:δNp) ]
  V  = @view dψV[ δNp .+ (1:N+1)]

  dψ .= 0
  V  .= 0

  # Update the fault data
  Δτ .= 0
  lf1 = 1

  Δτ .= -computetraction_stripped(HfI_FT, τ, lf1, u, δ)
  fake_knl_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)



  V[δNp+1:N+1] .= Vp

  nothing
end



 function write_text_slip_CPU(ψδ,t,i,yf,stations,p,base_name="",tdump=100)
  Vmax = 0.0

  if isdefined(i,:fsallast) 
    δNp = p.δNp
    N = p.N
    dψV = i.fsallast
    dψ = @view dψV[1:δNp]
    V = @view dψV[δNp .+ (1:N+1)]
    Vmax = maximum(abs.(extrema(V)))
    δ = @view ψδ[δNp .+ (1:δNp)]
    ψ = @view ψδ[1:δNp]
    vv = p.vv
    
    if mod(ctr[], p.save_stride_fields) == 0 || t == (sim_years ./ 31556926)
    
      vv[ctr[], 1] = t
      vv[ctr[], 2] = Vmax
      vv[ctr[], 3:end] = δ
      
    end
    global ctr[] += 1
   # @show ctr[]
  end

     Vmax
 end
