import Compat: range, undef
using Compat.SparseArrays
using Compat.LinearAlgebra

# DIAGONAL_SBP_D1 creates a diagonal norm SBP operator
# (D, Hinv, H, r) = diagonal_sbp_D1(p, N; xc = (-1,1))
#
# inputs:
#   p: sbp interior accuracy
#   N: finite difference grid size is N+1
#   xc: (keyword) grid span [default: (-1, 1)]
#
# outputs:
#   D:   difference operator Hinv*(-M+BS)
#   HI:  inverse of the SBP norm
#   H:   the SBP norm
#   r:   grid from xc[1] to xc[2]
#
#   References:
#   Operators for order 2, 4, 6 are from
#   @book{gustafsson2008high,
#     title={High order difference methods for time dependent PDE},
#     author={Gustafsson, Bertil},
#     year={2008},
#     publisher={Springer},
#     series={Springer Series in Computational Mathematics},
#     volume={38}
#   }
#
#   Order 8 operator is from
#   @article{MattssonSvardShoeybe2008JCP,
#     title={Stable and accurate schemes for the compressible Navier--Stokes equations},
#     author={Mattsson, Ken and Sv{\"a}rd, Magnus and Shoeybi, Mohammad},
#     journal={Journal of Computational Physics},
#     volume={227},
#     number={4},
#     pages={2293--2316},
#     year={2008},
#     doi={10.1016/j.jcp.2007.10.018}
#   }
#   with
#      x1 =  663/950
#      x2 = -229/1900
#      x3 =  415/547
#
#   Order 10 operator is from
#   @Article{MattssonAlmquist2013JCP,
#     author = {K. Mattsson and M. Almquist},
#     title = {A solution to the stability issues with block norm summation by parts operators},
#     journal = {Journal of Computational Physics},
#     volume = {253},
#     pages = {418--442},
#     year = {2013},
#     doi = {10.1016/j.jcp.2013.07.013}
#   }
#{{{
function diagonal_sbp_D1(p, N; xc = (-1, 1))
  M = N+1

  if p == 2
    bhinv = [2]
    d  = [-1/2 0 1/2]
    bd  = [-1 1]
  elseif p == 4
    bhinv = [48/17 48/59 48/43 48/49]
    d  = [1/12 -2/3 0 2/3 -1/12]
    bd = [-24/17  59/34  -4/17  -3/34  0     0;
           -1/2    0      1/2    0     0     0;
            4/43 -59/86   0     59/86 -4/43  0;
            3/98   0    -59/98   0    32/49 -4/49]
  elseif p == 6
    bhinv = [43200/13649 8640/12013 4320/2711 4320/5359 8640/7877 43200/43801];

    d = [-1/60 3/20 -3/4 0 3/4 -3/20 1/60];

    bd = [ -21600/13649   104009/54596   30443/81894  -33311/27298    16863/27298  -15025/163788        0           0         0;
          -104009/240260       0          -311/72078   20229/24026   -24337/48052   36661/360390        0           0         0;
           -30443/162660     311/32532       0        -11155/16266    41287/32532  -21999/54220         0           0         0;
            33311/107180  -20229/21436     485/1398        0           4147/21436    25427/321540      72/5359      0         0;
           -16863/78770    24337/31508  -41287/47262   -4147/15754        0         342523/472620   -1296/7877    144/7877    0;
            15025/525612  -36661/262806  21999/87602  -25427/262806 -342523/525612       0          32400/43801 -6480/43801 720/43801];
  elseif p == 8
    bhinv = [5080320/1498139 725760/1107307 80640/20761 725760/1304999 725760/299527 80640/103097 725760/670091 5080320/5127739];
    d = [1/280 -4/105 1/5 -4/5 0 4/5 -1/5 4/105 -1/280];

    bd = [    -2540160/1498139        699846290793/311403172540  -10531586157/311403172540  -145951651079/186841903524    398124597/15570158627    39152001/113858564  -80631892229/934209517620  -6230212503/311403172540        0                 0               0               0;
          -24132630717/55557028660               0                 2113176981/23016483302      5686186719/11508241651   -3408473341/138098899812  -39291999/210388330     607046586/11508241651    3460467023/483346149342        0                 0               0               0;
            3510528719/90623010660      -704392327/1294614438               0                   503511235/1294614438      354781619/2589228876       407439/3944590        -2986842/16597621        169381493/3020767022          0                 0               0               0;
          145951651079/1139279786988   -5686186719/13562854607    -1510533705/27125709214               0                6763379967/54251418428    13948923/49589962    -1603900430/40688563821   -3742312557/189879964498        0                 0               0               0;
            -398124597/21790888777      3408473341/37355809332    -1064344857/12451936444     -6763379967/12451936444             0                  763665/1198108     -1282435899/12451936444    7822226819/261490665324    -2592/299527          0               0               0;
              -1864381/23506116           13097333/58765290           -407439/19588430           -4649641/11753058          -254555/1237164               0                 5346432/9794215           -923328/9794215          3072/103097       -288/103097        0               0;
           11518841747/417855345780     -607046586/6964255763         1192698/23768791         1603900430/20892767289    1282435899/27857023052   -48117888/63658645              0                 301190400/366539777     -145152/670091      27648/670091    -2592/670091        0;
            6230212503/1065851828540   -3460467023/319755548562   -1524433437/106585182854     3742312557/106585182854  -7822226819/639511097124   58169664/487135205   -2108332800/2804873233              0               4064256/5127739  -1016064/5127739  193536/5127739  -18144/5127739];
  elseif p == 10
    bhinv = [18289152000/5261271563 1828915200/2881040311 406425600/52175551 6096384/11662993 87091200/50124587 72576000/50124587 87091200/148333439 152409600/63867949 16257024/20608675 1828915200/1704508063 18289152000/18425967263];

    d = [-1/1260,5/504,-5/84,5/21,-5/6,0,5/6,-5/21,5/84,-5/504,1/1260];

    bd = [  -1.7380923775745425e+00   2.3557601935237220e+00  -1.5328406598563976e-01  -5.7266565770416333e-01  -1.8308103515008173e-01   1.8186748267946842e-01   2.0034232582598244e-01   2.2678007363666621e-02  -1.1782459320459637e-01  -3.0591175636402144e-02   3.4890895862586133e-02   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
            -4.3020203737210871e-01   0.0000000000000000e+00   1.1837297346927406e-01   3.3928601158526644e-01   1.3241927733034406e-01  -8.7495003780608913e-02  -1.1750484124279399e-01  -1.6401912273575153e-02   6.2537843443041474e-02   1.7143274696828435e-02  -1.8155585855667674e-02   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
             3.4348531361887280e-01  -1.4525207124434036e+00   0.0000000000000000e+00   2.9011513992277767e+00  -2.2419288742360557e+00  -5.4662873578741478e-01   1.2908050607446131e+00   6.1514504292452719e-02  -4.2442625460011202e-01   1.5579158905288801e-02   5.2969140277981920e-02   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
             8.6111387816878188e-02  -2.7937273515056432e-01  -1.9467880944770807e-01   0.0000000000000000e+00   2.0170150914578375e-01   2.4269917331475005e-01  -7.7261988327590472e-02   5.0649247607525059e-02  -7.4775049946661561e-03  -4.0978487203372188e-02   1.8608207238964152e-02   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
             9.1509035082611684e-02  -3.6243526359648576e-01   5.0007055839856984e-01  -6.7045605191055857e-01   0.0000000000000000e+00  -1.7807807859119628e-02   7.5000761407401195e-01  -2.2979723229714316e-01  -1.2521154324370892e-01   6.8278284106004450e-02  -4.1575927541817690e-03   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
            -7.5752056274147259e-02   1.9956355926115746e-01   1.0160630736447970e-01  -6.7227694623145351e-01   1.4839839882599690e-02   0.0000000000000000e+00   5.4091068834671807e-01  -1.2712520372174399e-01  -8.9292453564020990e-02   1.6181541970619609e-01  -5.4289154769785249e-02   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
            -3.3838029883391296e-02   1.0867927550524317e-01  -9.7293058702223670e-02   8.6783825404790446e-02  -2.5344131542932297e-01  -2.1934035945002228e-01   0.0000000000000000e+00   2.7184438867288430e-01   1.9102691945078512e-01  -4.8646826827046824e-02  -6.2407959378425991e-03   4.6597719614658163e-04   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
            -1.5567948806367624e-02   6.1656604470023607e-02  -1.8844858059892756e-02  -2.3122780265804038e-01   3.1560994521078772e-01   2.0951677187991255e-01  -1.1048784865195491e+00   0.0000000000000000e+00   1.1823059621092409e+00  -5.3610400867086083e-01   1.5931375952374752e-01  -2.3673846172827626e-02   1.8939076938262100e-03   0.0000000000000000e+00   0.0000000000000000e+00   0.0000000000000000e+00;
             2.6737701764454301e-02  -7.7712278574126673e-02   4.2981266272823705e-02   1.1284579710276557e-02   5.6847566375570611e-02   4.8647834370398067e-02  -2.5665536068472994e-01  -3.9083324869946684e-01   0.0000000000000000e+00   6.5716944195909766e-01  -1.5822272208022428e-01   4.6954983762905661e-02  -7.8258306271509429e-03   6.2606645017207550e-04   0.0000000000000000e+00   0.0000000000000000e+00;
             9.4425181052687698e-03  -2.8976375375532045e-02  -2.1459742428921558e-03   8.4117843695442701e-02  -4.2165149106440383e-02  -1.1991463562335723e-01   8.8902467992349743e-02   2.4105392677971343e-01  -8.9388344421253152e-01   0.0000000000000000e+00   8.6496680152924643e-01  -2.5547312415382800e-01   6.3868281038457000e-02  -1.0644713506409501e-02   8.5157708051276015e-04   0.0000000000000000e+00;
            -9.9625965676187218e-03   2.8387641187789508e-02  -6.7495090936003027e-03  -3.5335033597892078e-02   2.3750992019053968e-03   3.7216380474824604e-02   1.0550378667904333e-02  -6.6265458456725809e-02   1.9908619649258188e-01  -8.0014409359906680e-01   0.0000000000000000e+00   8.2714572225493910e-01  -2.3632734921569687e-01   5.9081837303924217e-02  -9.8469728839873684e-03   7.8775783071898962e-04];
  else
    error(string("Operators for order ", p, " are not implemented"))
  end

  (bm, bn) = size(bd);

  if(M < 2*bm || M < bn)
    error("Grid not big enough to support the operator. Grid must have N >= ", max(bn,2*bm))
  end

  h = (xc[2] - xc[1]) / N
  @assert h > 0
  H_I = 1:M
  H_V = ones(M)
  H_V[1:bm] = 1 ./ bhinv[:]
  H_V[M-bm+1:M] = 1 ./ bhinv[end:-1:1]
  H = sparse(H_I, H_I, h * H_V)
  HI = sparse(H_I, H_I, 1 ./ (h * H_V))

  n = floor(Int64, length(d)/2);
  B_I1 = (bm+1:M-bm) * ones(1,p+1)
  B_J1 = ones(M-2bm,1) * (-div(p,2):div(p,2))' + B_I1
  B_V1 = ones(M-2bm)
  B_V1 = kron(B_V1, d)

  B_I2 = (1:bm) * ones(1, bn)
  B_J2 = ones(bm) * (1:bn)'
  B_V2 = bd
  B_I3 = (M+1) .- B_I2
  B_J3 = (M+1) .- B_J2
  B_V3 = -B_V2
  D = sparse([B_I1[:];B_I2[:];B_I3[:]],
             [B_J1[:];B_J2[:];B_J3[:]],
             [B_V1[:];B_V2[:];B_V3[:]]/h, N+1, N+1)

  r = Compat.range(xc[1], stop=xc[2], length=N+1)

  (D, HI, H, r)
end
#}}}

#DIAGONAL_SBP_D2 creates a diagonal norm SBP operator for the 2nd drerivative
# (D, S0, SN, Hinv, H, r) = diagonal_sbp_D2(p, N; xc = (-1,1))
#
# inputs:
#   p: sbp interior accuracy
#   N: finite difference grid size is N+1
#   xc: (keyword) grid span [default: (-1, 1)]
#
# outputs:
#   D:   difference operator Hinv*(-M-S0+SN)
#   S0:  boundary derivative operator at first grid point
#   SN:  boundary derivative operator at last grid point
#   HI:  inverse of the SBP norm
#   H:   the SBP norm
#   r:   grid from xc[1] to xc[2]
#
#   References:
#   Operators for order 2, 4, 6, 8 are from
#   @book{gustafsson2008high,
#     title={High order difference methods for time dependent PDE},
#     author={Gustafsson, Bertil},
#     year={2008},
#     publisher={Springer},
#     series={Springer Series in Computational Mathematics},
#     volume={38}
#   }
#
#   Order 10 operator is from
#   @Article{MattssonAlmquist2013JCP,
#     author = {K. Mattsson and M. Almquist},
#     title = {A solution to the stability issues with block norm summation by
#     parts operators},
#     journal = {Journal of Computational Physics},
#     volume = {253},
#     pages = {418--442},
#     year = {2013},
#     doi = {10.1016/j.jcp.2013.07.013}
#   }
#{{{
function diagonal_sbp_D2(p, N; xc = (-1, 1))

  if p == 2
    bhinv = [2];

    d  = [1 -2 1];
    bd  = d;

    BS = [3/2 -2 1/2];
  elseif p == 4
    bhinv = [48/17 48/59 48/43 48/49];

    d  = [-1/12 4/3 -5/2 4/3 -1/12];

    bd = [ 2    -5       4     -1       0      0;
           1    -2       1      0       0      0;
          -4/43 59/43 -110/43  59/43   -4/43   0;
          -1/49  0      59/49 -118/49  64/49  -4/49];

    BS = [11/6 -3 3/2 -1/3];

  elseif p == 6
    bhinv = [43200/13649 8640/12013 4320/2711 4320/5359 8640/7877 43200/43801];

    d = [1/90 -3/20 3/2 -49/18 3/2 -3/20 1/90];

    bd = [  114170/40947   -438107/54596   336409/40947  -276997/81894     3747/13649     21035/163788      0           0         0
              6173/5860      -2066/879       3283/1758      -303/293       2111/3516       -601/4395        0           0         0
            -52391/81330    134603/32532   -21982/2711    112915/16266   -46969/16266     30409/54220       0           0         0
             68603/321540   -12423/10718   112915/32154   -75934/16077    53369/21436    -54899/160770     48/5359      0         0
             -7053/39385     86551/94524   -46969/23631    53369/15754   -87904/23631    820271/472620  -1296/7877     96/7877    0
             21035/525612   -24641/131403   30409/87602   -54899/131403  820271/525612  -117600/43801   64800/43801 -6480/43801 480/43801];

    BS = [25/12 -4 3 -4/3 1/4];

  elseif p == 8
    bhinv = [5080320/1498139 725760/1107307 80640/20761 725760/1304999 725760/299527 80640/103097 725760/670091 5080320/5127739];
    d = [-1/560 8/315 -1/5 8/5 -205/72 8/5 -1/5 8/315 -1/560]

    bd = zeros(8,12);
    bd[1, 1] =  4870382994799/1358976868290;
    bd[1, 2] =  -893640087518/75498714905  ;
    bd[1, 3] =   926594825119/60398971924  ;
    bd[1, 4] = -1315109406200/135897686829 ;
    bd[1, 5] =    39126983272/15099742981  ;
    bd[1, 6] =    12344491342/75498714905  ;
    bd[1, 7] =  -451560522577/2717953736580;

    bd[2, 1] =  333806012194/390619153855;
    bd[2, 2] = -154646272029/111605472530;
    bd[2, 3] =    1168338040/33481641759 ;
    bd[2, 4] =   82699112501/133926567036;
    bd[2, 5] =    -171562838/11160547253 ;
    bd[2, 6] =  -28244698346/167408208795;
    bd[2, 7] =   11904122576/167408208795;
    bd[2, 8] =   -2598164715/312495323084;

    bd[3, 1] =   7838984095/52731029988;
    bd[3, 2] =   1168338040/5649753213 ;
    bd[3, 3] =    -88747895/144865467  ;
    bd[3, 4] =    423587231/627750357  ;
    bd[3, 5] = -43205598281/22599012852;
    bd[3, 6] =   4876378562/1883251071 ;
    bd[3, 7] =  -5124426509/3766502142 ;
    bd[3, 8] =  10496900965/39548272491;

    bd[4, 1] =  -94978241528/828644350023;
    bd[4, 2] =   82699112501/157837019052;
    bd[4, 3] =    1270761693/13153084921 ;
    bd[4, 4] = -167389605005/118377764289;
    bd[4, 5] =   48242560214/39459254763 ;
    bd[4, 6] =  -31673996013/52612339684 ;
    bd[4, 7] =   43556319241/118377764289;
    bd[4, 8] =  -44430275135/552429566682;

    bd[5, 1] =   1455067816/21132528431;
    bd[5, 2] =   -171562838/3018932633 ;
    bd[5, 3] = -43205598281/36227191596;
    bd[5, 4] =  48242560214/9056797899 ;
    bd[5, 5] = -52276055645/6037865266 ;
    bd[5, 6] =  57521587238/9056797899 ;
    bd[5, 7] = -80321706377/36227191596;
    bd[5, 8] =   8078087158/21132528431;
    bd[5, 9] =        -1296/299527     ;

    bd[6, 1] =   10881504334/327321118845;
    bd[6, 2] =  -28244698346/140280479505;
    bd[6, 3] =    4876378562/9352031967  ;
    bd[6, 4] =  -10557998671/12469375956 ;
    bd[6, 5] =   57521587238/28056095901 ;
    bd[6, 6] = -278531401019/93520319670 ;
    bd[6, 7] =   73790130002/46760159835 ;
    bd[6, 8] = -137529995233/785570685228;
    bd[6, 9] =          2048/103097      ;
    bd[6,10] =          -144/103097      ;

    bd[7, 1] = -135555328849/8509847458140;
    bd[7, 2] =   11904122576/101307707835 ;
    bd[7, 3] =   -5124426509/13507694378  ;
    bd[7, 4] =   43556319241/60784624701  ;
    bd[7, 5] =  -80321706377/81046166268  ;
    bd[7, 6] =   73790130002/33769235945  ;
    bd[7, 7] = -950494905688/303923123505 ;
    bd[7, 8] =  239073018673/141830790969 ;
    bd[7, 9] =       -145152/670091       ;
    bd[7,10] =         18432/670091       ;
    bd[7,11] =        -1296/670091        ;


    bd[8, 1] =             0             ;
    bd[8, 2] =   -2598164715/206729925524;
    bd[8, 3] =   10496900965/155047444143;
    bd[8, 4] =  -44430275135/310094888286;
    bd[8, 5] =     425162482/2720130599  ;
    bd[8, 6] = -137529995233/620189776572;
    bd[8, 7] =  239073018673/155047444143;
    bd[8, 8] = -144648000000/51682481381 ;
    bd[8, 9] =       8128512/5127739     ;
    bd[8,10] =      -1016064/5127739     ;
    bd[8,11] =        129024/5127739     ;
    bd[8,12] =         -9072/5127739     ;

    BS = [4723/2100 -839/175 157/35 -278/105 103/140 1/175 -6/175];

  elseif p == 10
    bhinv = [18289152000/5261271563 1828915200/2881040311 406425600/52175551 6096384/11662993 87091200/50124587 72576000/50124587 87091200/148333439 152409600/63867949 16257024/20608675 1828915200/1704508063 18289152000/18425967263];

    M = zeros(11,16);

    M[ 1, 1] =  1.2056593789671863908;
    M[ 1, 2] = -1.3378814169347239658;
    M[ 1, 3] =  0.0036847309286546532061;
    M[ 1, 4] =  0.15698288365600946515;
    M[ 1, 5] = -0.0037472461482539197952;
    M[ 1, 6] = -0.0062491712449361657064;
    M[ 1, 7] = -0.029164045872729581661;
    M[ 1, 8] =  0.00054848184117832929161;
    M[ 1, 9] =  0.013613461413384884448;
    M[ 1,10] = -0.0025059220258337808220;
    M[ 1,11] = -0.00094113457993630916498;

    M[ 2, 2] =  2.1749807117105597139;
    M[ 2, 3] = -0.12369059547124894597;
    M[ 2, 4] = -0.83712574037924152603;
    M[ 2, 5] =  0.050065127254670973258;
    M[ 2, 6] =  0.0081045853127317536361;
    M[ 2, 7] =  0.097405846039248226536;
    M[ 2, 8] = -0.00068942461520402214720;
    M[ 2, 9] = -0.041326971493379188475;
    M[ 2,10] =  0.0075778529605774119402;
    M[ 2,11] =  0.0025800256160095691057;

    M[ 3, 3] =  0.18361596652499065332;
    M[ 3, 4] =  0.048289690013342693109;
    M[ 3, 5] = -0.19719621435164680412;
    M[ 3, 6] =  0.11406859029505842791;
    M[ 3, 7] = -0.029646295985488126964;
    M[ 3, 8] = -0.0016038463172861201306;
    M[ 3, 9] =  0.0032879841528337653050;
    M[ 3,10] = -0.00093242311589807387463;
    M[ 3,11] =  0.00012241332668787820533;

    M[ 4, 4] =  1.2886524606662484673;
    M[ 4, 5] = -0.14403037739488789185;
    M[ 4, 6] = -0.44846291607489015475;
    M[ 4, 7] = -0.10598334599408054277;
    M[ 4, 8] = -0.015873275740355918053;
    M[ 4, 9] =  0.073988493386459608166;
    M[ 4,10] = -0.012508848749152899785;
    M[ 4,11] = -0.0039290233894513005339;

    M[ 5, 5] =  0.51482665719685186210;
    M[ 5, 6] =  0.051199577887125103015;
    M[ 5, 7] = -0.36233561810883077365;
    M[ 5, 8] =  0.091356850268746392169;
    M[ 5, 9] =  0.0024195916108052419451;
    M[ 5,10] = -0.0018564214413731389338;
    M[ 5,11] = -0.00070192677320704413827;

    M[ 6, 6] =  0.68636003380365860083;
    M[ 6, 7] = -0.28358848290867614908;
    M[ 6, 8] = -0.13836006478253396528;
    M[ 6, 9] =  0.0076158070663111995297;
    M[ 6,10] =  0.011447010307180005164;
    M[ 6,11] = -0.0021349696610286552676;

    M[ 7, 7] =  1.5216081480839085990;
    M[ 7, 8] = -0.42653865162216293237;
    M[ 7, 9] = -0.42047484981879143123;
    M[ 7,10] =  0.019813359263872926304;
    M[ 7,11] =  0.019221397241190103344;

    M[ 8, 8] =  1.0656733504627815335;
    M[ 8, 9] = -0.66921872668484232217;
    M[ 8,10] =  0.12022033144141336599;
    M[ 8,11] = -0.030157881394591483631;

    M[ 9, 9] =  2.4064247712949611684;
    M[ 9,10] = -1.5150200315922263367;
    M[ 9,11] =  0.17373015320416595052;

    M[10,10] =  2.7682502485427255096;
    M[10,11] = -1.5975407111468405444;

    M[11,11] =  2.9033627686681129471;
    d     = [1/3150  -5/1008  5/126  -5/21  5/3 -5269/1800 5/3 -5/21 5/126 -5/1008 1/3150]
    for k = 1:5
      M[11-5+k,11 .+ (1:k)] = -d[k:-1:1];
    end

    M[1:11,1:11] = M[1:11,1:11]'+M[1:11,1:11]-diagm(0 => diag(M[1:11,1:11]));

    BS = zeros(1,16);
    BS[1:7] = -[-49/20 6 -15/2 20/3 -15/4 6/5 -1/6];
    e0 = zeros(11,1);
    e0[1] = 1;

    bd = diagm(0 => bhinv[:]) * (-M+e0*BS);

  else
    error(string("Operators for order ", p, " are not implemented"))
  end

  (bm, bn) = size(bd);

  M = N+1;
  if(M < 2*bm || M < bn)
    error("Grid not big enough to support the operator. Grid must have N >= ", max(bn,2*bm))
  end

  h = (xc[2] - xc[1]) / N
  @assert h > 0
  H_I = 1:M
  H_V = ones(M)
  H_V[1:bm] = bhinv[:]
  H_V[M-bm+1:M] = bhinv[end:-1:1]
  HI = sparse(H_I, H_I, H_V / h)
  H  = sparse(H_I, H_I, h ./ H_V)

  n = floor(Int64, length(d)/2);
  B_I1 = (bm+1:M-bm) * ones(1,p+1)
  B_J1 = ones(M-2bm,1) * (-div(p,2):div(p,2))' + B_I1
  B_V1 = ones(M-2bm)
  B_V1 = kron(B_V1, d)

  B_I2 = (1:bm) * ones(1, bn)
  B_J2 = ones(bm) * (1:bn)'
  B_V2 = bd
  B_I3 = (M+1) .- B_I2
  B_J3 = (M+1) .- B_J2
  B_V3 = B_V2
  D = sparse([B_I1[:];B_I2[:];B_I3[:]],
             [B_J1[:];B_J2[:];B_J3[:]],
             [B_V1[:];B_V2[:];B_V3[:]]/h^2, N+1, N+1)

  S0 = sparse(ones(length(BS)), 1:length(BS), -BS[:]/h, N+1, N+1);
  SN = sparse((N+1) * ones(length(BS)), (N+1):-1:(N+2-length(BS)),
              BS[:]/h, N+1, N+1);

  r = Compat.range(xc[1], stop=xc[2], length=N+1)

  (D, S0, SN, HI, H, r)

end
#}}}

end
