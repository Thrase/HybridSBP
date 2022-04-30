
using CUDA 

function knl_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)
    N = length(ψ)
    bid = blockIdx().x  # get the thread's block ID
    tid = threadIdx().x # get my thread ID
    dim = blockDim().x  # how many threads in each block

    i = dim * (bid - 1) + tid  # unique global thread ID
    if i <= N
        ψn = ψ[i]
        an = RSa[i]

        τn = Δτ[i] + τz0
        xR = abs(τn / η)
        xL = -xR
        x = V[i]
        ftol = 1e-9
        maxiter = 500
        minchange=0
        atolx = 1e-9
        rtolx = 1e-9

       
        fL = σn * an * asinh(xL * (1 / (2 * RSV0)) * exp(ψn / an))    + η * xL - τn
        fR = σn * an * asinh(xR * (1 / (2 * RSV0)) * exp(ψn / an))    + η * xR - τn
        
        if fL * fR > 0
            #no_iters[i] = -maxiter
            #print("No root")
            return nothing
        end

        f =  σn * an * asinh(x * (1 / (2 * RSV0)) * exp(ψn / an))  + η * x - τn
        df = σn * an * (1 / sqrt(1 + (x * (1 / (2 * RSV0)) * exp(ψn / an))^2)) * (1 / (2 * RSV0)) * exp(ψn / an) + η
        dxlr = xR - xL
      

        for iter = 1:maxiter
            dx = -f / df
            x  = x + dx
        
            if x < xL || x > xR || abs(dx) / dxlr < minchange
              x = (xR + xL) / 2
              dx = (xR - xL) / 2
            end
        
            f = σn * an * asinh(x * (1 / (2 * RSV0)) * exp(ψn / an))    + η * x - τn
            df = σn * an * (1 / sqrt(1 + (x * (1 / (2 * RSV0)) * exp(ψn / an))^2)) * (1 / (2 * RSV0)) * exp(ψn / an) + η
        
        
            if f * fL > 0
              (fL, xL) = (f, x)
            else
              (fR, xR) = (f, x)
            end
            dxlr = xR - xL
        
            if abs(f) < ftol && abs(dx) < atolx + rtolx * (abs(dx) + abs(x))
                V[i] = x
                dψ[i] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(x) / RSV0)
                #no_iters[i] = iter

              continue
            end
          end
    end

    return nothing
end


function fake_knl_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)
  N = length(ψ)
 
 for i = 1:N 
      ψn = ψ[i]
      an = RSa[i]

      τn = Δτ[i] + τz0
      xR = abs(τn / η)
      xL = -xR
      x = V[i]
      ftol = 1e-9
      maxiter = 500
      minchange=0
      atolx = 1e-9
      rtolx = 1e-9

     
      fL = σn * an * asinh(xL * (1 / (2 * RSV0)) * exp(ψn / an))    + η * xL - τn
      fR = σn * an * asinh(xR * (1 / (2 * RSV0)) * exp(ψn / an))    + η * xR - τn
      
      if fL * fR > 0
          #no_iters[i] = -maxiter
          print("No root")
          return nothing
      end

      f =  σn * an * asinh(x * (1 / (2 * RSV0)) * exp(ψn / an))  + η * x - τn
      df = σn * an * (1 / sqrt(1 + (x * (1 / (2 * RSV0)) * exp(ψn / an))^2)) * (1 / (2 * RSV0)) * exp(ψn / an) + η
      dxlr = xR - xL
    

      for iter = 1:maxiter
          dx = -f / df
          x  = x + dx
      
          if x < xL || x > xR || abs(dx) / dxlr < minchange
            x = (xR + xL) / 2
            dx = (xR - xL) / 2
          end
      
          f = σn * an * asinh(x * (1 / (2 * RSV0)) * exp(ψn / an))    + η * x - τn
          df = σn * an * (1 / sqrt(1 + (x * (1 / (2 * RSV0)) * exp(ψn / an))^2)) * (1 / (2 * RSV0)) * exp(ψn / an) + η
      
      
          if f * fL > 0
            (fL, xL) = (f, x)
          else
            (fR, xR) = (f, x)
          end
          dxlr = xR - xL
      
          if abs(f) < ftol && abs(dx) < atolx + rtolx * (abs(dx) + abs(x))
              V[i] = x
              dψ[i] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - abs(x) / RSV0)
              #no_iters[i] = iter
	      
            continue
          end
        
        end
  end
	
  return nothing
end




function knl_do_nothing_solveV!(Δτ, τz0, V, ψ, RSa, RSb, RSDc, RSV0, RSf0, σn, η, dψ)
  N = length(ψ)
  bid = blockIdx().x  # get the thread's block ID
  tid = threadIdx().x # get my thread ID
  dim = blockDim().x  # how many threads in each block

  i = dim * (bid - 1) + tid  # unique global thread ID
  if i <= N


 
     ψn = ψ[i]
     an = RSa[i]

        τn = Δτ[i] + τz0
        xR = abs(τn / η)
        xL = -xR
        x = V[i]
        ftol = 1e-9
        maxiter = 500
        minchange=0
        atolx = 1e-9
        rtolx = 1e-9
  
        @inbounds V[i] = 2*RSV0*sinh(τn/(σn * an)) * exp(-ψn / an)
        @inbounds dψ[i] = (RSb * RSV0 / RSDc) * (exp((RSf0 - ψn) / RSb) - CUDA.abs(x) / RSV0)
             
 
  end

  return nothing
end




