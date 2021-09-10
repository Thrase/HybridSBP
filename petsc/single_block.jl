# Force the random number generator to be the same for all runs of the script
using Random
Random.seed!(777)

include(joinpath("..", "global_curved.jl"))

using MPI
using PETSc

include("sbp_operators.jl")

#XXX: I should remove this
function sbp_width(p)
    if p == 2
        return (3, 3)
    elseif p == 4
        return (6, 5)
    elseif p == 6
        return (9, 7)
    else
        error("unknown sbp order: $p")
    end
end

# Have the ksp version call the DMDA version
sbp_matrix(p, A, J, ksp::PETSc.AbstractKSP, crr, crs, css; kwargs...) =
    sbp_matrix(p, A, J, PETSc.getDMDA(ksp), crr, crs, css; kwargs...)

function sbp_matrix(p, A, J, da::PETSc.AbstractDMDA, crr, crs, css; τscale = 1)
    csr = crs

    # Get the floating point type
    Float = eltype(A)

    # Get the corners of the box that this processor is responsible for
    corners = PETSc.getcorners(da)

    # get the global grid dimension
    Nq = PETSc.getinfo(da).global_size[1:2]

    # Get the grid spacing

    # Get operator coefficients
    sbp = (SBP(Float, p, Nq[1]), SBP(Float, p, Nq[2]))

    for i in ((corners.lower):(corners.upper))
        #
        # ∂_ξ1 ∂_ξ1
        #

        A_ξ1 = sbp[1].A(view(crr, :, i[2]))
        for j1 in nzcols(A_ξ1, i[1])
            val = sbp[2].H[i[2], i[2]] * A_ξ1[i[1], j1]
            PETSc.addindex!(A, val, i, CartesianIndex(j1, i[2], 1))
        end

        #
        # ∂_ξ2 ∂_ξ2
        #

        A_ξ2 = sbp[2].A(view(css, i[1], :))
        for j2 in nzcols(A_ξ2, i[2])
            val = sbp[1].H[i[1], i[1]] * A_ξ2[i[2], j2]
            PETSc.addindex!(A, val, i, CartesianIndex(i[1], j2, 1))
        end

        #
        # ∂_ξ1 ∂_ξ2
        # ∂_ξ2 ∂_ξ1
        #

        for j2 in nzcols(sbp[2].Q, i[2])
            for j1 in nzcols(sbp[1].Q, i[1])
                val = sbp[2].Q[j2, i[2]] * csr[i[1], j2] * sbp[1].Q[i[1], j1]
                val += sbp[1].Q[j1, i[1]] * crs[j1, i[2]] * sbp[2].Q[i[2], j2]
                PETSc.addindex!(A, val, i, CartesianIndex(j1, j2, 1))
            end
        end

        # Face ξ2 = -1
        if i[1] == 1
            # crr[1] * (S0 + S0')
            for j1 in nzcols(sbp[1].S0, 1)
                val = crr[i] * sbp[1].S0[1, j1] * sbp[2].H[i[2], i[2]]
                PETSc.addindex!(A, val, i, CartesianIndex(j1, i[2], 1))
                PETSc.addindex!(A, val, CartesianIndex(j1, i[2], 1), i)
            end
            # crs * Q + Q' * crs
            for j2 in nzcols(sbp[2].Q, i[2])
                val = crs[i] * sbp[2].Q[i[2], j2]
                # crs[1] * Q
                PETSc.addindex!(A, val, i, CartesianIndex(i[1], j2, 1))
                # crs[1] * Q'
                PETSc.addindex!(A, val, CartesianIndex(i[1], j2, 1), i)
            end

            # Compute the near boundary minimum value
            ψ = mapreduce(min, CartesianIndex.(1:(sbp[1].l_min), i[2], 1)) do j
                (crr[j] + css[j] - sqrt((crr[j] - css[j])^2 + 4 * crs[j]^2)) / 2
            end

            # Compute the penalty parameter
            τ =
                2τscale / (ψ * sbp[1].h) *
                (crr[i]^2 / sbp[1].β + crs[i]^2 / sbp[1].α)

            # Scale by H matrix
            val = sbp[2].H[i[2], i[2]] * τ
            PETSc.addindex!(A, val, i, i)
        end

        # Face ξ2 = 1
        if i[1] == Nq[1]
            # crr[end] * (SN + SN')
            for j1 in nzcols(sbp[1].SN, Nq[1])
                val = -crr[i] * sbp[1].SN[Nq[1], j1] * sbp[2].H[i[2], i[2]]
                PETSc.addindex!(A, val, i, CartesianIndex(j1, i[2], 1))
                PETSc.addindex!(A, val, CartesianIndex(j1, i[2], 1), i)
            end
            # crs * Q + Q' * crs
            for j2 in nzcols(sbp[2].Q, i[2])
                val = -crs[i] * sbp[2].Q[i[2], j2]
                PETSc.addindex!(A, val, i, CartesianIndex(i[1], j2, 1))
                PETSc.addindex!(A, val, CartesianIndex(i[1], j2, 1), i)
            end

            # Compute the near boundary minimum value
            ψ = mapreduce(
                min,
                CartesianIndex.(Nq[1] + 1 .- (1:(sbp[1].l_min)), i[2], 1),
            ) do j
                (crr[j] + css[j] - sqrt((crr[j] - css[j])^2 + 4 * crs[j]^2)) / 2
            end

            # Compute the penalty parameter
            τ =
                2τscale / (ψ * sbp[1].h) *
                (crr[i]^2 / sbp[1].β + crs[i]^2 / sbp[1].α)

            # Scale by H matrix
            val = sbp[2].H[i[2], i[2]] * τ
            PETSc.addindex!(A, val, i, i)
        end

        # Face ξ1 = -1
        if i[2] == 1
            # css[1] * (S0 + S0')
            for j2 in nzcols(sbp[2].S0, 1)
                val = css[i] * sbp[2].S0[1, j2] * sbp[1].H[i[1], i[1]]
                PETSc.addindex!(A, val, i, CartesianIndex(i[1], j2, 1))
                PETSc.addindex!(A, val, CartesianIndex(i[1], j2, 1), i)
            end
            # csr * Q + Q' * csr
            for j1 in nzcols(sbp[1].Q, i[1])
                val = csr[i] * sbp[1].Q[i[1], j1]
                PETSc.addindex!(A, val, i, CartesianIndex(j1, i[2], 1))
                PETSc.addindex!(A, val, CartesianIndex(j1, i[2], 1), i)
            end

            # Compute the near boundary minimum value
            ψ = mapreduce(min, CartesianIndex.(i[1], 1:(sbp[2].l_min), 1)) do j
                (crr[j] + css[j] - sqrt((crr[j] - css[j])^2 + 4 * csr[j]^2)) / 2
            end

            # Compute the penalty parameter
            τ =
                2τscale / (ψ * sbp[2].h) *
                (css[i]^2 / sbp[1].β + csr[i]^2 / sbp[1].α)

            # Scale by H matrix
            val = sbp[1].H[i[1], i[1]] * τ
            PETSc.addindex!(A, val, i, i)
        end

        # Face ξ1 = 1
        if i[2] == Nq[2]
            # css[end] * (SN + SN')
            for j2 in nzcols(sbp[2].SN, Nq[2])
                val = -css[i] * sbp[2].SN[Nq[2], j2] * sbp[1].H[i[1], i[1]]
                PETSc.addindex!(A, val, i, CartesianIndex(i[1], j2, 1))
                PETSc.addindex!(A, val, CartesianIndex(i[1], j2, 1), i)
            end
            # crs * Q + Q' * crs
            for j1 in nzcols(sbp[1].Q, i[1])
                val = -csr[i] * sbp[1].Q[i[1], j1]
                PETSc.addindex!(A, val, i, CartesianIndex(j1, i[2], 1))
                PETSc.addindex!(A, val, CartesianIndex(j1, i[2], 1), i)
            end

            # Compute the near boundary minimum value
            ψ = mapreduce(
                min,
                CartesianIndex.(i[1], Nq[2] + 1 .- (1:(sbp[2].l_min)), 1),
            ) do j
                (crr[j] + css[j] - sqrt((crr[j] - css[j])^2 + 4 * crs[j]^2)) / 2
            end

            # Compute the penalty parameter
            τ =
                2τscale / (ψ * sbp[2].h) *
                (css[i]^2 / sbp[2].β + csr[i]^2 / sbp[2].α)

            # Scale by H matrix
            val = sbp[1].H[i[1], i[1]] * τ
            PETSc.addindex!(A, val, i, i)
        end
    end

    PETSc.assemble!(A)
end

function single_block(
    sbp_order,
    Nq,
    Float = Float64,
    comm = MPI.COMM_WORLD,
    opts = (ksp_monitor = true, ksp_view = true),
)
    λ1 = rand(Float, Nq...)
    λ2 = rand(Float, Nq...)
    θ = π * rand(Float, Nq...)

    # Create some random positive definite coefficients
    crr = @. λ1 * cos(θ)^2 + λ2 * sin(θ)^2
    css = @. λ1 * sin(θ)^2 + λ2 * cos(θ)^2
    crs = @. (λ2 - λ1) * cos(θ) * sin(θ)

    # get petsc library and initialize
    petsclib = PETSc.getlib(; PetscScalar = Float)
    PETSc.initialize(petsclib)

    # get the stencil width (which is the boundary stencil size minus 1)
    stencil_width = sbp_width(sbp_order)[1] - 1

    # Create the PETSC dmda object
    da = PETSc.DMDA(
        petsclib,
        comm,
        (PETSc.DM_BOUNDARY_NONE, PETSc.DM_BOUNDARY_NONE), # Use no ghost nodes
        Nq,                                               # Global grid size
        1,                                                # Number of DOF per node
        stencil_width,                                    # Stencil width
        PETSc.DMDA_STENCIL_BOX;                           # Stencil type
        opts...,
    )

    # Setup the Krylov solver for the distributed array
    ksp = PETSc.KSP(da; opts...)

    PETSc.setcomputeoperators!(
        (x...) -> sbp_matrix(sbp_order, x..., crr, crs, css),
        ksp,
    )

    A = PETSc.MatAIJ(da)
    sbp_matrix(sbp_order, A, nothing, ksp, crr, crs, css)

    # println()
    # println("A")
    # Libc.flush_cstdio()
    # flush(stdout)
    # PETSc.view(A)
    # Libc.flush_cstdio()
    # flush(stdout)

    metrics = (
        crr = crr,
        crs = crs,
        css = css,
        J = ones(Float, Nq...),
        coord = nothing,
        facecoord = nothing,
        sJ = nothing,
        nx = nothing,
        ny = nothing,
    )
    lop = locoperator(sbp_order, Nq[1] - 1, Nq[2] - 1, metrics; τscale = 1)
    # @show lop[313, :]
    B = PETSc.MatSeqAIJ(petsclib, -lop.M̃)

    # println()
    # println("-B")
    # Libc.flush_cstdio()
    # flush(stdout)
    # PETSc.view(B)
    # Libc.flush_cstdio()
    # flush(stdout)

    PETSc.LibPETSc.MatAXPY(
        petsclib,
        B,
        Float(1),
        A,
        PETSc.LibPETSc.DIFFERENT_NONZERO_PATTERN,
    )

    println("norm(A) = $(norm(A))")
    println("norm(A - B) = $(norm(B))")

    PETSc.destroy(B)

    PETSc.destroy(ksp)
    PETSc.destroy(da)
    PETSc.finalize(petsclib)
    nothing
end

single_block(4, (30, 25))
single_block(4, (25, 30))
