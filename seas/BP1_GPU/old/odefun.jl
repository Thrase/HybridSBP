const year_seconds = 31556926

global const ctr = Ref{Int64}(1) 

using OrdinaryDiffEq
using DiffEqCallbacks
using Printf
using Plots
using DelimitedFiles
using CUDA

function odefun(dψV, ψδ, p, t)
  
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
  vv = p.vv
 cuarry_ones = p.cuarry_ones

  @show t ./ 31556926

  
  ψ  = @view ψδ[1:δNp]
  δ  = ψδ[δNp+1:δNp+N+1]


  #bc_Dirichlet = (lf, x, y) -> (2-lf)*(0.5 * δ) + (lf-1)*fill(t * Vp/2, size(y))
  #bc_Neumann   = (lf, x, y, nx, ny) -> zeros(size(x))
  

  bdry_vec_mod_GPU!(g, F, δ, Vp, t, cuarry_ones)

  # solve for displacements everywhere in domain
  cg!(u, M, g)


  # set up rates of change for  state and slip
  dψ = @view dψV[1:δNp]
  V  = @view dψV[δNp+1:δNp+N+1]

  dψ[:] = 0 * cuarry_ones
  V[:]= 0 * cuarry_ones


  # Update the fault data
  Δτ[:] = 0 * cuarry_ones
  lf1 = 1


  #Δτ .= -computetraction_stripped(HfI_FT, τ, lf1, u, δ)
  computetraction_stripped_GPU!(Δτ, HfI_FT, τ, 1, u, δ)

  num_threads_per_block = 64
  num_blocks = cld(δNp, num_threads_per_block)
  @cuda threads=num_threads_per_block blocks=num_blocks knl_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)
  synchronize()
  
   V[:] = Vp * cuarry_ones
  #V[δNp+1:N+1] .= Vp
  

  nothing
end


function setupfaultstations(locations)
  T = eltype(locations)
  @assert size(locations,2) == 2
end


struct ODE_results
  t_list::Array{Any,1}
  V_list::Array{Any,1}
  δ_list::Array{Any,1}
 #  station_1::Array{Any,1}
 #  station_2::Array{Any,1}
 #  station_3::Array{Any,1}
 #  station_4::Array{Any,1}
 #  station_5::Array{Any,1}
 #  station_6::Array{Any,1}
 #  station_7::Array{Any,1}
 #  station_8::Array{Any,1}
 #  station_9::Array{Any,1}
 #  station_10::Array{Any,1}
 #  station_11::Array{Any,1}
 stations::Dict{Int64,Array{Any,1}}
end

function stopping_criteria(ψδ,t,i)
 Vmax = 0.0
 if isdefined(i,:fsallast)
   δNp = div(length(ψδ),2)
   dψV = i.fsallast
   dψ = @view dψV[1:δNp]
   V = @view dψV[δNp .+ (1:δNp)]
   Vmax = maximum(abs.(extrema(V)))
   δ = @view ψδ[δNp .+ (1:δNp)]
   ψ = @view ψδ[1:δNp]
   return Vmax >= 1e-3
 end
end

affect!(integrator)= terminate!(integrator)

function savestop(ψδ,t,i,ODEresults,p)
 Vmax = 0.0
 # count = 0
 if isdefined(i,:fsallast)
   # δNp = div(length(ψδ),2)
   δNp = p.δNp
   N = p.N
   dψV = i.fsallast
   dψ = @view dψV[1:δNp]
   push!(ODEresults.t_list,t)

   
   V = @view dψV[δNp .+ (1:N+1)]
   Vmax = maximum(abs.(extrema(V)))
   δ = @view ψδ[δNp .+ (1:δNp)]
   ψ = @view ψδ[1:δNp]
   # p.u_old .= p.u
   if Vmax >= 1e-3
     # terminate!(i)
     # p.count[1] += 1
     # p.count[1] .= 1
     # @show p.count[1]
     # open("data_for_toby.txt","a") do io
     #   writedlm(io,p.u)
     # end
     # utemp = p.u
     push!(p.counter,1)
     if length(p.counter) >= 2
       terminate!(i)
       open("data_for_toby.txt","w") do io
         write(io,"$(ODEresults.t_list[end]) $(ODEresults.t_list[end-1])\n")
         write(io,"*"^40,"\n")
         writedlm(io,ψ,' ')
         write(io,"*"^40,"\n")
         writedlm(io,[p.u p.u_old],' ')
       end
     end
     p.u_old .= p.u
   end

   # if p.count[1] >= 2
   #   terminate!(i)
   # end 
   # if count >= 2
   #   terminate!(i)
   #   utemp = hcat(utemp,p.u)
   #   open("data_for_toby.txt","w") do io
   #     writedlm(io,utemp)
   #   end
   # end
 end
 # open("data_for_toby.txt","w") do io
 #   writedlm(io,[p.u p.u_old], ' ')
 # end
 Vmax
end

function saveslip(ψδ,t,i,ODEresults,yf,stations,station_indices,p,base_name="",tdump=100)
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
    
     push!(ODEresults.t_list,t)
     push!(ODEresults.V_list,Vmax)
     push!(ODEresults.δ_list,copy(δ))
    
     for i =1:12
      push!(ODEresults.stations[i],[δ[station_indices[i]], V[station_indices[i]], p.RSDc .* exp.((ψ[station_indices[i]] .- p.RSf0) ./ p.RSb) ./ p.RSV0])
       #push!(ODEresults.stations[i],[V[station_indices[i]],p.τ[station_indices[i]] .- p.η * V[station_indices[i]], p.RSDc .* exp.((ψ[station_indices[i]] .- p.RSf0) ./ p.RSb) ./
       #p.RSV0])
     end
   
     
     if (t == sim_years * year_seconds)
       @show station_indices
       for i in range(1,stop=length(station_indices))
         station_name = Symbol("station_",i)
         # @show station_name
         station_id = station_indices[i]
         open("$(base_name)$(stations[i]).dat","w") do f
           write(f,"t slip slip_rate shear_stress state\n")
           for n = 1:length(ODEresults.t_list)
             # write(f,"$(ODEresults.t_list[n]) $(ODEresults.δ_list[n][station_id]) $(log10(abs(ODEresults.(eval(station_name[i])))))\n")
             write(f,"$(ODEresults.t_list[n]) $(ODEresults.δ_list[n][station_id]) $(log10(abs(ODEresults.stations[i][n][1]))) $(ODEresults.stations[i][n][2]) $(log10(ODEresults.stations[i][n][3]))\n")
           end
         
         end
       end
     end
     if (t == sim_years * year_seconds)
       open("$(base_name)V.dat","w") do f
         write(f,"z \n")
         write(f,"t Slip_rate Slip")
         write(f,"\n") 
         write(f,"0.0 0.0 ")
         # for i in stations
         for i in 1:δNp
           # write(f,"$(stations[i])")
           # write(f,"$(Integer((i-1)*40000/(δNp-1))) ")
           write(f,"$(yf[i]) ")
         end
         write(f,"\n")
       for n = 1:length(ODEresults.t_list)
       # for n = 1:3
         # write(f, "$(ODEresults.t_list[n]) $(ODEresults.V_list[n]) \n")
         # write(f, "$(ODEresults.t_list[n]) $(log10(ODEresults.V_list[n])) $(ODEresults.δ_list[n])")
         write(f, "$(ODEresults.t_list[n]) $(log10(ODEresults.V_list[n]))")
         # for i in 1:div(length(V)+1,2)
         #   write(f,"$(ODEresults.δ_list[n][2*i-1])")
         # end

         # for i in Integer.(range(1,81,length=81))
         for i in 1:δNp
             # write(f," $(ODEresults.δ_list[n][i])")
             write(f," $(ODEresults.δ_list[n][i])")
         end
         write(f,"\n")
       end
       # write(f,ODEresults.δ_list)
       # @show δ
       # @show ODEresults.δ_list
     end
     # @show ODEresults.δ_list
   end
   end
   Vmax
  end



function write_text_slip(ψδ,t,i,yf,stations,station_indices,p,base_name="",tdump=100)
  Vmax = 0.0

  if isdefined(i,:fsallast) 
    δNp = p.δNp
    N = p.N
    dψV = i.fsallast
    dψ = @view dψV[1:δNp]
    V = @view dψV[δNp .+ (1:N+1)]
    Vmax = CUDA.maximum(V)
    δ = @view ψδ[δNp .+ (1:N+1)]
    ψ = @view ψδ[1:δNp]
    vv = p.vv

    if mod(ctr[], p.save_stride_fields) == 0 || t == (sim_years ./ 31556926)

       vv[ctr[], :] = ψδ[δNp+1:length(ψδ)]	
       
       dd = Array{Float64}(undef, 1, 2)
       dd[1] = t
       dd[2] = log10(Vmax)
    
      open("devol.txt", "a") do io
        
        writedlm(io, dd)
      end
      
    end
    global ctr[] += 1
    @show ctr[]
  end
    

     Vmax
 end

