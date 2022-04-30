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
    Nr = 2^8
    Ns = Nr
    μ = 1
    # SBP interior order
    SBPp   = 2

    r = Array(LinRange(0, 1, Nr+1))  
    s = Array(LinRange(0, 1, Ns+1))

    # create operators
    (M̃, F, τ, H̃, HfI_FT) = get_operators(SBPp, Nr, Ns, μ)
    # factor with Cholesky
    M_f = cholesky(Symmetric(M̃))

    # create operators
    M_h = M̃
    M_d = CUSPARSE.CuSparseMatrixCSC(M_h)
    
    bb = 0.001 .* sin.(r) * sin.(s')
    g_h = zeros((Nr+1) * (Ns+1))
    g_d = CuArray(convert(Array{Float64, 1}, g_h))
    u_h = 0.001 * ones(size(g_h))
    u_h_c = copy(u_h)
    u_d = CuArray(convert(Array{Float64, 1}, u_h_c))

    @time M_f\g_h;
    #@show size(M_f)
    @time cg!(u_h, M_h, g_h);
    @time cg!(u_d, M_d, g_d);

    @time u_dd = M_d\g_d
    
    5