abstract type AbstractSBPMatrix{T} <: AbstractMatrix{T} end
Base.size(op::AbstractSBPMatrix) = (op.Nq, op.Nq)

struct SBP{T, HBL, D1IW, D1BL, D1BW, D2IW, D2ID, D2BL, D2BW, D2BD, D2SW}
    Nq::Int
    # H Operators
    bh::Vector{T}
    bhinv::Vector{T}
    # D1 Operators
    D1_int::Vector{T}
    D1_bnd::Matrix{T}
    # D2 Operators
    A_int::Matrix{T}
    A_bnd::Array{T, 3}
    S_bnd::Vector{T}
    grid::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}}
    # Borrowing paramters
    α::T       # same as bh[1]
    β::T       # borrowing lemma parameter
    l_min::Int # near boundary min rng
end
SBP(T, p::Integer, Nq) = SBP(T, Val(p), Nq)

function Base.getproperty(
    sbp::S,
    sym::Symbol,
) where {
    S <:
    SBP{
        T,
        HBL,
        D1IW,
        D1BL,
        D1BW,
        D2IW,
        D2ID,
        D2BL,
        D2BW,
        D2BD,
        D2SW,
    },
} where {T, HBL, D1IW, D1BL, D1BW, D2IW, D2ID, D2BL, D2BW, D2BD, D2SW}
    if sym === :max_width
        max(D1IW, D1BW, D2IW, D2BW, D2SW)
    elseif sym === :HBL      # H boundary Width
        return HBL
    elseif sym === :H
        return SBPH{T, S}(sbp)
    elseif sym === :h
        return T(sbp.grid.step)
    elseif sym === :D1IW # D1 Interior Width
        return D1IW
    elseif sym === :D1BL # D1 boundary length size(D1_bnd, 1)
        return D1BL
    elseif sym === :D1BW # D1 boundary width size(D1_bnd, 2)
        return D1BW
    elseif sym === :D1
        return SBPD1{T, S}(sbp)
    elseif sym === :Q
        return SBPQ{T, S}(sbp)
    elseif sym === :D2IW # D2 Interior Width size(A_int, 2)
        return D2IW
    elseif sym === :D2ID # D2 Interior depth size(A_int, 1)
        return D2ID
    elseif sym === :D2BL # D2 boundary length size(A_bnd, 3)
        return D2BL
    elseif sym === :D2BW # D2 boundary width size(A_bnd, 2)
        return D2BW
    elseif sym === :D2BD # D2 boundary width size(A_bnd, 1)
        return D2BD
    elseif sym === :D2SW # S boundary width length(S_bnd)
        return D2SW
    elseif sym === :A
        return (x) -> SBPA(sbp, x)
    elseif sym === :S0
        return SBPS0{T, S}(sbp)
    elseif sym === :SN
        return SBPSN{T, S}(sbp)
    else
        return getfield(sbp, sym)
    end
end

function A_remove_interior!(A_bnd, A_int, HBL)
    istart = HBL + 1
    blen = size(A_bnd, 2)
    radius = div(size(A_int, 1), 2)
    rng = (-radius):radius
    # We subtract the interior contribution from the boundary stencil weights
    # since these will be added separately in the matrix construction
    # j: vector weight index
    # k: stencil column index
    # i: stencil row index
    for i in istart:blen,
        (kn, k) in enumerate(i .+ rng),
        (jn, j) in enumerate(i .+ rng)

        if all((j, i, k) .<= size(A_bnd))
            @inbounds A_bnd[j, k, i] -= A_int[jn, kn]
        end
    end
end

#
# Define SBP H matrix
#
struct SBPH{T, S <: SBP{T}} <: AbstractSBPMatrix{T}
    sbp::S
end

function Base.getproperty(op::SBPH, sym::Symbol)
    if sym === :sbp
        return getfield(op, sym)
    elseif sym === :BL
        op.sbp.HBL
    else
        return getproperty(op.sbp, sym)
    end
end

function Base.getindex(op::SBPH, i::Int, j::Int)
    T = eltype(op)
    @boundscheck checkbounds(op, i, j)

    BL = op.BL
    Nq = op.Nq

    if i ≠ j
        return -zero(T)
    elseif i <= BL # left boundary
        @inbounds return op.bh[i] * op.h
    elseif Nq + 1 - BL <= i # right boundary
        i = Nq + 1 - i
        @inbounds return op.bh[i] * op.h
    else # interior
        return op.h
    end
end

#
# Define SBP D1 and Q matrix
#
struct SBPD1{T, S <: SBP{T}} <: AbstractSBPMatrix{T}
    sbp::S
end
struct SBPQ{T, S <: SBP{T}} <: AbstractSBPMatrix{T}
    sbp::S
end

function Base.getproperty(op::Union{SBPD1, SBPQ}, sym::Symbol)
    if sym === :sbp
        return getfield(op, sym)
    elseif sym === :IW
        return op.sbp.D1IW
    elseif sym === :BW
        return op.sbp.D1BW
    elseif sym === :BL
        return op.sbp.D1BL
    else
        return getproperty(op.sbp, sym)
    end
end

function Base.getindex(op::SBPD1, i::Int, j::Int)
    T = eltype(op)
    @boundscheck checkbounds(op, i, j)

    Nq = op.Nq
    BL = op.BL
    BW = op.BW
    IR = div(op.IW, 2)

    if i <= BL # left boundary
        @inbounds return j <= BW ? T(op.D1_bnd[i, j] / op.h) : -zero(T)
    elseif Nq + 1 - BL <= i # right boundary
        (i, j) = Nq + 1 .- (i, j)
        @inbounds return j <= BW ? -T(op.D1_bnd[i, j] / op.h) : -zero(T)
    else # interior
        k = j - i
        @inbounds return k ∈ (-IR):IR ? T(op.D1_int[IR + 1 + k] / op.h) :
                         -zero(T)
    end
end

function Base.getindex(op::SBPQ, i::Int, j::Int)
    T = eltype(op)
    @boundscheck checkbounds(op, i, j)

    Nq = op.Nq
    BL = op.BL
    BW = op.BW
    IR = div(op.IW, 2)

    if i <= BL # left boundary
        @inbounds return j <= BW ? op.bh[i] * op.D1_bnd[i, j] : -zero(T)
    elseif Nq + 1 - BL <= i # right boundary
        (i, j) = Nq + 1 .- (i, j)
        @inbounds return j <= BW ? -op.bh[i] * op.D1_bnd[i, j] : -zero(T)
    else # interior
        k = j - i
        @inbounds return k ∈ (-IR):IR ? op.D1_int[IR + 1 + k] : -zero(T)
    end
end

# Get the possibly non-zero columns for this row
function nzcols(op::Union{SBPD1, SBPQ}, row::Int)
    @boundscheck checkbounds(op, row, :)
    Nq = op.Nq
    BL = op.BL
    BW = op.BW
    IR = div(op.IW, 2)
    if row <= BL # left boundary
        return 1:BW
    elseif Nq + 1 - BL <= row # right boundary
        return (Nq + 1 - BW):Nq
    else # interior
        return (row - IR):(row + IR)
    end
end

#
# Define SBP A matrix
#

# NumPacker always returns c for any getindex
struct NumPacker{T}
    c::T
end
Base.getindex(c::NumPacker, _...) = return c.c

struct SBPA{T, S <: SBP{T}, C <: Union{NumPacker{T}, AbstractVector{T}}} <:
       AbstractSBPMatrix{T}
    sbp::S
    c::C
    function SBPA(
        sbp::S,
        c::C,
    ) where {T, S <: SBP{T}, C <: Union{T, AbstractVector{T}}}
        sbp.Nq >= 2sbp.D2BL || error("Nq = $(sbp.Nq) should be ≥ $(2sbp.D2BL)")
        if c isa AbstractVector
            length(c) == sbp.Nq || error("`c` must be of length $(sbp.Nq)")
        else
            c = NumPacker(c)
        end
        return new{T, S, typeof(c)}(sbp, c)
    end
end

function Base.getproperty(op::Union{SBPA}, sym::Symbol)
    if sym === :sbp
        return getfield(op, sym)
    elseif sym === :c
        return getfield(op, sym)
    elseif sym === :IW
        return op.sbp.D2IW
    elseif sym === :ID
        return op.sbp.D2ID
    elseif sym === :BW
        return op.sbp.D2BW
    elseif sym === :BL
        return op.sbp.D2BL
    elseif sym === :BD
        return op.sbp.D2BD
    else
        return getproperty(op.sbp, sym)
    end
end

function Base.getindex(op::SBPA, i::Int, j::Int)
    T = eltype(op)

    @boundscheck checkbounds(op, i, j)

    Nq = op.Nq
    BL = op.BL
    BW = op.BW
    BD = op.BD
    IR = div(op.IW, 2)
    ID = div(op.ID, 2)

    v = -zero(T)

    if op.HBL < i < op.Nq + 1 - op.HBL
        k = j - i
        if k ∈ (-IR):IR
            for l in (-ID):ID
                @inbounds v += op.A_int[ID + 1 + l, ID + 1 + k] * op.c[i + l]
            end
        end
    end

    if i <= BL && j <= BW # left boundary 
        for l in 1:BD
            @inbounds v += op.A_bnd[l, j, i] * op.c[l]
        end
    elseif Nq + 1 - BL <= i && Nq + 1 - BW <= j
        (i, j) = Nq + 1 .- (i, j)
        for l in 1:BD
            @inbounds v += op.A_bnd[l, j, i] * op.c[Nq + 1 - l]
        end
    end

    return v / T(op.h)
end

# Get the possibly non-zero columns for this row
function nzcols(op::Union{SBPA}, row::Int)
    @boundscheck checkbounds(op, row, :)
    Nq = op.Nq
    BL = op.BL
    BW = op.BW
    HBL = op.HBL
    IR = div(op.IW, 2)
    if row <= BW # left boundary
        return 1:max(BW, row + IR)
    elseif Nq + 1 - BW <= row # right boundary
        return min(Nq + 1 - BW, row - IR):Nq
    else # interior
        return (row - IR):(row + IR)
    end
end

# Define SBP S0 and SN matrix
#
struct SBPS0{T, S <: SBP{T}} <: AbstractSBPMatrix{T}
    sbp::S
end
struct SBPSN{T, S <: SBP{T}} <: AbstractSBPMatrix{T}
    sbp::S
end

function Base.getproperty(op::Union{SBPS0, SBPSN}, sym::Symbol)
    if sym === :sbp
        return getfield(op, sym)
    elseif sym === :SW
        return op.sbp.D2SW
    elseif sym === :S_bnd
        return op.sbp.S_bnd
    else
        return getproperty(op.sbp, sym)
    end
end

function Base.getindex(op::SBPS0, i::Int, j::Int)
    @boundscheck checkbounds(op, i, j)

    T = eltype(op)

    if i == 1 && j <= op.SW
        @inbounds return -op.S_bnd[j] / op.h
    else
        return -zero(T)
    end
end

function nzcols(op::SBPS0, row::Int)
    @boundscheck checkbounds(op, row, :)
    if row == 1
        return 1:(op.SW)
    else
        return 1:0
    end
end

function Base.getindex(op::SBPSN, i::Int, j::Int)
    @boundscheck checkbounds(op, i, j)

    T = eltype(op)

    j = op.Nq + 1 - j
    if i == op.Nq && j <= op.SW
        @inbounds return op.S_bnd[j] / op.h
    else
        return -zero(T)
    end
end

function nzcols(op::SBPSN, row::Int)
    @boundscheck checkbounds(op, row, :)
    if row == op.Nq
        return (op.Nq + 1 - op.SW):(op.Nq)
    else
        return 1:0
    end
end

#
# Coeffficients for 4th order
#

function SBP(T, ::Val{4}, Nq, xc = (-1, 1))

    # Boundary H-matrix
    HBL = 4
    bhinv = T.([48 // 17, 48 // 59, 48 // 43, 48 // 49])
    bh = 1 ./ bhinv

    # D1 operators
    D1IW, D1BL, D1BW = 5, 4, 6
    D1_int = T.([1 // 12, -2 // 3, 0, 2 // 3, -1 // 12])
    D1_bnd =
        T.(
            [
                -24//17 59//34 -4//17 -3//34 0 0
                -1//2 0 1//2 0 0 0
                4//43 -59//86 0 59//86 -4//43 0
                3//98 0 -59//98 0 32//49 -4//49
            ],
        )

    @assert size(D1_bnd) == (D1BL, D1BW)
    @assert length(D1_int) == D1IW
    @assert length(bh) == HBL

    # D2 operators
    α, β, l = T(17 // 48), 0.2505765857, 4
    D2IW, D2ID = 5, 5
    D2BL, D2BW, D2BD = 6, 6, 6
    D2SW = 4

    S_bnd = T.([11 // 6, -3, 3 // 2, -1 // 3])
    @assert length(S_bnd) == D2SW

    @inbounds begin
        A_int = zeros(T, D2ID, D2IW)
        A_int[1:3, 1] = [1 // 8, -1 // 6, 1 // 8]
        A_int[1:4, 2] = [-1 // 6, -1 // 2, -1 // 2, -1 // 6]
        A_int[1:5, 3] = [1 // 24, 5 // 6, 3 // 4, 5 // 6, 1 // 24]
        A_int[2:5, 4] = [-1 // 6, -1 // 2, -1 // 2, -1 // 6]
        A_int[3:5, 5] = [1 // 8, -1 // 6, 1 // 8]

        # Boundary coefficient interpolation weights
        A_bnd = zeros(T, D2BD, D2BW, D2BL)
        A_bnd[1, 1, 1] = 12 // 17
        A_bnd[2, 1, 1] = +59 // 192
        A_bnd[3, 1, 1] = +27010400129 // 345067064608
        A_bnd[4, 1, 1] = +69462376031 // 2070402387648

        A_bnd[1, 1, 2] = A_bnd[1, 2, 1] = -59 // 68
        A_bnd[3, 1, 2] = A_bnd[3, 2, 1] = -6025413881 // 21126554976
        A_bnd[4, 1, 2] = A_bnd[4, 2, 1] = -537416663 // 7042184992

        A_bnd[1, 1, 3] = A_bnd[1, 3, 1] = 2 // 17
        A_bnd[2, 1, 3] = A_bnd[2, 3, 1] = -59 // 192
        A_bnd[3, 1, 3] = A_bnd[3, 3, 1] = +2083938599 // 8024815456
        A_bnd[4, 1, 3] = A_bnd[4, 3, 1] = +213318005 // 16049630912

        A_bnd[1, 1, 4] = A_bnd[1, 4, 1] = 3 // 68
        A_bnd[3, 1, 4] = A_bnd[3, 4, 1] = -1244724001 // 21126554976
        A_bnd[4, 1, 4] = A_bnd[4, 4, 1] = +752806667 // 21126554976

        A_bnd[3, 1, 5] = A_bnd[3, 5, 1] = 49579087 // 10149031312
        A_bnd[4, 1, 5] = A_bnd[4, 5, 1] = -49579087 // 10149031312

        A_bnd[3, 1, 6] = A_bnd[3, 6, 1] = +1 // 784
        A_bnd[4, 1, 6] = A_bnd[4, 6, 1] = -1 // 784

        A_bnd[1, 2, 2] = 3481 // 3264
        A_bnd[3, 2, 2] = +9258282831623875 // 7669235228057664
        A_bnd[4, 2, 2] = +236024329996203 // 1278205871342944

        A_bnd[1, 2, 3] = A_bnd[1, 3, 2] = -59 // 408
        A_bnd[3, 2, 3] = A_bnd[3, 3, 2] = -29294615794607 // 29725717938208
        A_bnd[4, 2, 3] = A_bnd[4, 3, 2] = -2944673881023 // 29725717938208

        A_bnd[1, 2, 4] = A_bnd[1, 4, 2] = -59 // 1088
        A_bnd[3, 2, 4] = A_bnd[3, 4, 2] = +260297319232891 // 2556411742685888
        A_bnd[4, 2, 4] = A_bnd[4, 4, 2] = -60834186813841 // 1278205871342944

        A_bnd[3, 2, 5] = A_bnd[3, 5, 2] = -1328188692663 // 37594290333616
        A_bnd[4, 2, 5] = A_bnd[4, 5, 2] = +1328188692663 // 37594290333616

        A_bnd[3, 2, 6] = A_bnd[3, 6, 2] = -8673 // 2904112
        A_bnd[4, 2, 6] = A_bnd[4, 6, 2] = +8673 // 2904112

        A_bnd[1, 3, 3] = 1 // 51
        A_bnd[2, 3, 3] = +59 // 192
        A_bnd[3, 3, 3] = +378288882302546512209 // 270764341349677687456
        A_bnd[4, 3, 3] = +13777050223300597 // 26218083221499456
        A_bnd[5, 3, 3] = +564461 // 13384296

        A_bnd[1, 3, 4] = A_bnd[1, 4, 3] = 1 // 136
        A_bnd[3, 3, 4] =
            A_bnd[3, 4, 3] = -4836340090442187227 // 5525802884687299744
        A_bnd[4, 3, 4] = A_bnd[4, 4, 3] = -17220493277981 // 89177153814624
        A_bnd[5, 3, 4] = A_bnd[5, 4, 3] = -125059 // 743572

        A_bnd[4, 3, 5] = A_bnd[4, 5, 3] = -10532412077335 // 42840005263888
        A_bnd[3, 3, 5] =
            A_bnd[3, 5, 3] = +1613976761032884305 // 7963657098519931984
        A_bnd[5, 3, 5] = A_bnd[5, 5, 3] = +564461 // 4461432

        A_bnd[3, 3, 6] = A_bnd[3, 6, 3] = +33235054191 // 26452850508784
        A_bnd[4, 3, 6] = A_bnd[4, 6, 3] = -960119 // 1280713392
        A_bnd[5, 3, 6] = A_bnd[5, 6, 3] = -3391 // 6692148

        A_bnd[1, 4, 4] = 3 // 1088
        A_bnd[3, 4, 4] = +507284006600757858213 // 475219048083107777984
        A_bnd[4, 4, 4] = +1950062198436997 // 3834617614028832
        A_bnd[5, 4, 4] = +1869103 // 2230716
        A_bnd[6, 4, 4] = +1 // 24

        A_bnd[3, 4, 5] =
            A_bnd[3, 5, 4] = -4959271814984644613 // 20965546238960637264
        A_bnd[4, 4, 5] = A_bnd[4, 5, 4] = -15998714909649 // 37594290333616
        A_bnd[5, 4, 5] = A_bnd[5, 5, 4] = -375177 // 743572
        A_bnd[6, 4, 5] = A_bnd[6, 5, 4] = -1 // 6

        A_bnd[3, 4, 6] = A_bnd[3, 6, 4] = +752806667 // 539854092016
        A_bnd[4, 4, 6] = A_bnd[4, 6, 4] = +1063649 // 8712336
        A_bnd[5, 4, 6] = A_bnd[5, 6, 4] = -368395 // 2230716
        A_bnd[6, 4, 6] = A_bnd[6, 6, 4] = +1 // 8

        A_bnd[3, 5, 5] = 8386761355510099813 // 128413970713633903242
        A_bnd[4, 5, 5] = +2224717261773437 // 2763180339520776
        A_bnd[5, 5, 5] = +280535 // 371786
        A_bnd[6, 5, 5] = +5 // 6

        A_bnd[3, 5, 6] = A_bnd[3, 6, 5] = -13091810925 // 13226425254392
        A_bnd[4, 5, 6] = A_bnd[4, 6, 5] = -35039615 // 213452232
        A_bnd[5, 5, 6] = A_bnd[5, 6, 5] = -1118749 // 2230716
        A_bnd[6, 5, 6] = A_bnd[6, 6, 5] = -1 // 2

        A_bnd[3, 6, 6] = +660204843 // 13226425254392
        A_bnd[4, 6, 6] = 3290636 // 80044587
        A_bnd[5, 6, 6] = +5580181 // 6692148
        A_bnd[6, 6, 6] = +3 // 4
    end

    A_remove_interior!(A_bnd, A_int, HBL)

    return SBP{T, HBL, D1IW, D1BL, D1BW, D2IW, D2ID, D2BL, D2BW, D2BD, D2SW}(
        Nq,
        bh,
        bhinv,
        D1_int,
        D1_bnd,
        A_int,
        A_bnd,
        S_bnd,
        range(T(xc[1]), xc[2]; length = Nq),
        α,
        β,
        l,
    )
end
