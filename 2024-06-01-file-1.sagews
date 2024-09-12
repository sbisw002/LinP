︠3ed69487-91af-4d28-9a5c-ad536073226es︠
#This is Stamford Bridge
# Works on the same error repeatedly to generate  statistics on it
# Read data generated with Explorer_read
from numpy import array, eye, hstack, ones, vstack, zeros, random, append, size
from cvxopt import solvers
from cvxopt.base import matrix as m
from cvxopt.modeling import op, dot, variable
import numpy as np
from sage.matrix.matrix_mod2_dense import pluq



Lx = 4;Ly = 4;NXY = Lx*Ly;n = 1
elements = [0, 1];  probabilities = [0.98, 0.02];
NoSam = 4
rgRange_LPRW = 1
rgRange_RW = 100
H1 = np.matrix(zeros((Ly,Ly)))

for count in range(0,Ly):
    H1[count,count]=1

for count in range(0,Ly-1):
    H1[count,count+1]=1

H1[Ly-1,0] = 1
H2=H1

K1 = H1.transpose(); K2 = H2.transpose()
HzL = np.kron(K2,eye(Lx));HzR = np.kron(eye(Lx),K1)
HxL = np.kron(eye(Lx),H1);HxR = np.kron(H2,eye(Lx))

Hz = hstack([HzL, HzR]);Hx = hstack([HxL, HxR])


def form_Ab(Hz, snd):
    RnK = len(Hz)
    CnK = len(Hz.T)
    Lz = 0
    Cum_L1s = np.zeros((1,RnK+1))

    Cum_L1s = [ 0 for j in range(RnK+1)]
    Var_E = [ 0 for j in range(RnK)]
    Add_M = [[i for i in range(3)] for j in range(3)]
    BigA = zeros((RnK,CnK))


    for df in range(0,RnK):
        Cols = np.count_nonzero(Hz[df,:])
        Lz = Lz + np.power(2, Cols-1)
        Cum_L1s[1+df] = Lz
        Var_E[df] = Cols
        Add_M = [[0 for i in range(np.power(2, Cols-1))] for j in range(RnK )]
        Add_M[df][:] = [1 for j in range(np.power(2, Cols-1))]
        Add_M = np.asmatrix(Add_M)
        BigA = append(BigA, Add_M, axis=1)
    Cn = len(BigA.T)

    for df in range(0,RnK):
        NRo = np.transpose(np.nonzero(Hz[df,:]))
        rowIs = zeros((Var_E[df],1))
        for ef in range(0,Var_E[df]):
            ro, co = NRo[ef]
            rowIs[ef] = int(co)
        S_Mat = form_Ss(rowIs,CnK)
        snI = int(snd[df])
        C_Mat = form_Cs(Var_E[df], snI)
        OneR = zeros((Var_E[df],Cn))
        OneR[:,0:CnK] = S_Mat
        OneR[:,CnK+Cum_L1s[df]:CnK+Cum_L1s[df+1]] = -C_Mat
        BigA = append(BigA, OneR, axis=0)

    varE = np.asmatrix(Var_E)
    sumVarE = varE.sum()
    Colb = np.zeros((RnK+sumVarE,1))
    Colb[0:RnK,0]=np.ones(RnK)

    CNA = len(BigA.T)
    css = np.zeros((CNA,1))
    css[0:2*NXY,0]=np.ones(2*NXY)
    return BigA,Colb,css



def form_Ss(rowIs,CnK):
    nS = len(rowIs)
    S_Mat = np.zeros((nS,CnK))
    for df in range(0,nS):
        S_Mat[df,int(rowIs[df]) ] = 1
    return S_Mat


def form_Cs(bitN, snI):
    C_Mat = np.matrix(np.zeros((bitN,0)) )

    for x in range(pow(2,bitN)):
         v=bin(x)[2:].zfill(bitN)
         b = np.matrix(map(int, v))
         no1 = np.count_nonzero(b)

         if ( np.logical_xor(np.logical_not(snI),no1%2) ):
              C_Mat = np.hstack([C_Mat, b.T])
    return C_Mat


def ratize(inc):
    lenT = len(inc)
    renA = np.zeros((lenT,1))
    for ef in range(lenT):
        Nofill = 1
        for df in range(1,10):
            on_v = 1/df
            diff = abs(inc[ef]-(on_v))
            fra_d = diff*df
            if (fra_d < 0.00001):
                renA[ef] = round(on_v,5)
                Nofill = 0

        if Nofill:
            renA[ef] = round(inc[ef],5)

    return renA



# inA is the array of indices with a 1
# ids is the array of indices with fractional numbers
def NumFr(erDC,NXY):
    Nofr = 0;No1s=0; ids = []; inA = []
    for df in range(0,2*NXY):
        NS = erDC[df,0]
        fl = np.floor(NS); ci = np.ceil(NS)
        dfl = abs(NS-fl); dci = abs(NS-ci)
        if np.logical_or(dfl > 0.0001, dci > 0.0001):
            Nofr = Nofr+1; ids.append(df)

        if (abs(NS-1) < 0.0001):
            No1s = No1s+1;  inA.append(df)

    Weit = erDC.sum()
    return Nofr,No1s,Weit,ids,inA     #ids is the fractional entries,    inA is the integral entries




def Recond(Hz,snd,inA,No1s):
    Msnd = snd
    DedE = np.matrix(np.zeros((2*NXY,1)) )

    MHz=np.delete(Hz,inA,1)  # np.delete(*,*,1) to delete columns
    for df in range(0,No1s):
        CoN = inA[df]
        OnE = np.matrix(np.zeros((2*NXY,1)) )
        OnE[CoN,0]=1   # A column with only one 1; all the rest are zeros
        IRw = Hz*OnE
        Msnd = (Msnd+IRw)%2
        DedE[inA[df],0] = 1
    return MHz,Msnd,DedE

def check1(e,Hz,snd):
    Nsnd = Hz*e
#    print(Nsnd);    print(Nsnd==snd);    print((Nsnd==snd).all())
    if (Nsnd==snd).all():
        return 1
    else:
        return 0

def check2(ePe,Hx):
#    print("ePe is")
#    print(ePe)
    Hx = matrix(GF(2),Hx)
    ePe = matrix(GF(2),ePe)
    Hxe = vstack([Hx, ePe ])

    Hxe2 = matrix(GF(2),Hxe)
    Hx2 = matrix(GF(2),Hx)


    if (np.shape(Hxe)[0]-np.shape(Hx)[0] != 1 ):
        print('Problem in check2: AA   ')
        print('Problem in check2: AA  ')
        print('Problem in check2: AA  ')
        print('Problem in check2: AA  ')

    if (  (np.shape(Hxe)[1] != 2*NXY)       or       (np.shape(Hx)[1] != 2*NXY)  ):
        print('Problem in check2: BB  ')
        print('Problem in check2: BB  ')
        print('Problem in check2: BB  ')
        print('Problem in check2: BB  ')



    nke = rank(Hxe2)
    nk = rank(Hx2)

    if (  ((nke-nk) != 0)       and       ((nke-nk) != 1)  ):
        print('Problem in check2: CC  ')
        print('Problem in check2: CC  ')
        print('Problem in check2: CC  ')
        print('Problem in check2: CC  ')
        print('nke-nk:  ',nke-nk)
    return (1-(nke-nk))




def SetPiv(CapE,Pivs,rnk,LeGT):                    # Sets the pivotal variables to the values according to CapE
    erV = np.matrix(zeros((LeGT,1)))               # erV is the decoded error with all redundant variables set to 0
    for df in range(0,rnk):
        erV[Pivs[df],0]=CapE[df,0]
    return erV

def Err1s(CapE,Pivs,Reds,rnk,df,LeGT):             # select the df entry in Reds[] i.e. Reds[df] and create the error Err1s
    ea = Reds[df]                                  # that comes from setting that redundant variable(Reds[df]) to 1
    eGR = EHz[:,ea]
    PivE = (CapE+eGR)%2
    erD1 = SetPiv(PivE,Pivs,rnk,LeGT)
    erD1[ea,0] = 1
    return erD1





TheErr = [[0 for i in xrange(2*NXY)]]
IntE1s = [[0 for i in xrange(2*NXY)]]
IntE2s = [[0 for i in xrange(2*NXY)]]
UDE_n = 0

Ar_save = np.matrix(zeros((NoSam,12)))
Br_save = np.matrix(zeros((NoSam,12)))
Fr_save = np.matrix(zeros((NoSam,4)))









for tryn in range(0,NoSam):
    #produce erOG form p_bond errors ans save the weight of the error in a new column in A/Br_Save  later
    erOG = np.random.choice(elements, 2*NXY, p=probabilities)


    FixWt = np.count_nonzero(erOG)
    erOG = np.matrix(erOG)
    erOG = erOG.T
    #TheErr = erOG

    #print(erOG)

    Bsnd = Hz*erOG;snd = Bsnd%2
    Rn = len(Hz);Cn = len(Hz.T)


    BigA, Colb, css = form_Ab(Hz,snd)

    CNA = len(BigA.T)    # len() must be used for the numpy matrix. It has a different definition of cvxopt matrix

    BigA_p = m(BigA)
    Colb_p = m(Colb)

    #print(size(BigA))
    #print(size(Colb))
    #print(CNA)
    x = variable(int(CNA))

    Idn = m(np.identity(CNA))
    h1 = m(np.zeros(CNA))
    h2 = m(np.ones(CNA))




#############################################################################################################
#############################################################################################################
#############################################################################################################

    ## For the original toric code only
    P_Nofr=0;P_No1s=0;P_Weit=0;P_ids=[];P_inA=[]       #P_ids is the fractional entries,    P_inA is the integral entries
    #DedE   # never actually used for the toric code decoding part; so no worries!

    BFst_16 = -1
    BW_Fst16 = -1
    BW_RC_Fst16 = -1
    BRgt_16 = -1
    BW_Rgt_16 = -1

    BLEver = 2*NXY
    BLEver_At = -1
    BW_LEver_RC = -1
    BMinW_ever = 2*NXY
    BfoundA = -1
    #MnWT_Xt = 2*NXY

    for rg in range(rgRange_RW):


        if (rg==0):
            PerM = np.matrix( np.identity(2*NXY-P_No1s) )
        else:
            L2=np.random.permutation([ i for i in range(0,2*NXY-P_No1s) ]);
            PerM = np.matrix( np.zeros((2*NXY-P_No1s,2*NXY-P_No1s)) );
            for df in range(0,2*NXY-P_No1s):
                PerM[df,L2[df]]=1
        PHz = Hz*PerM                                          # PHz is the column-permutized Hz
        Hz2 = matrix(GF(2),PHz);snd2 = matrix(GF(2),snd)
        #print(Hz2)
        Hz_s = Hz2.augment(snd2)
        EHz = Hz_s; EHz.echelonize()                           # EHz is the reduced row echelonized PHz
        #print(EHz)
        LUf, Pf, Qf = pluq(Hz_s)                               # LUf is not necessary
                                                               # The 1st rnk(rank) entries of Qf are the pivots(variables
                                                               # with leading 1s) in the reduced row echelonized EHz

        #P, L, U = Hz_s.LU(pivot='nonzero')                    # never used
        #print(U) prints the same matrix as EHz
        #LUf == U  #to check if they are the same or not
        #print(Hz_s == P*L*U)


        #print('EHz:   ');print(EHz);



        rnk = rank(Hz2);    #print(rnk)
        NRed = 2*NXY-rnk-P_No1s

        Hzt = Hz2.T
        #k_Hzt = Hzt.kernel()
        #Gz = k_Hzt.basis_matrix()
        #print('Gz'); print(Gz); print(Hz2*Gz.T)   # check
        #print(Qf[0:rnk]);  print(Qf[rnk:2*NXY+1]);  print(EHz[:,1])
        Pivs = Qf[0:rnk]
        Reds = []
        for df in range(0,2*NXY-P_No1s):
            if (df in Pivs):
                continue
            else:
                Reds.append(df)
        CapE = EHz[:,2*NXY-P_No1s]                            # CapE is the rightmost column in EHz i.e.
                                                              # the reduced row echelon form of (PHz|s)
                                                              # --> the


#        print('Reds:   ');print(Reds);print('For rg= ',rg)
        Lar = np.matrix(np.zeros(2*NXY))
        for df in range(NXY+1):
            Lar[0,Reds[df]] = 1

        RedO = np.where(PerM*(Lar.T))[0]
#        print('Red0 here:   ');print('For rg= ',rg)
#        print(RedO)

#        print('  ');
#        print('  ');
#        print('  ');


        erD00 = (SetPiv(CapE,Pivs,rnk,2*NXY-P_No1s))%2        # the error answer with all redundant variables set to 0s
        oan = [1 for i in range(P_No1s)]

        P_inN = [0 for i in range(P_No1s)]         # initializing P_inN

        for fg in range(0,P_No1s):
            P_inN[fg]= P_inA[fg]-fg
                 # because we have to insert back the integers into a column that has size (2*NXY-P_No1s)
                 # instead of the original size 2*NXY, we have to use P_inN instead of P_inA

        erD0 = (np.insert(PerM*erD00, P_inN, oan) ).T            # the number of columns grow back from
                                                        # (2*NXY-P_No1s) to 2*NXY

        #print(snd.T); print((Hz*erD0%2).T)
        W_erAR = np.matrix(np.zeros((1,NRed+1 ))  )
        erAR = np.matrix(zeros((2*NXY,NRed+1)))
        erAR[:,0] = erD0.T
        W_erAR[0,0] = np.count_nonzero(erAR[:,0])


        for df in range(0,NRed):
            erD01 = PerM*Err1s(CapE,Pivs,Reds,rnk,df,2*NXY-P_No1s)
            erD1 = (np.insert(erD01, P_inN, oan) ).T        # the number of columns grow back from
                                                        # (2*NXY-P_No1s) to 2*NXY
    #        print((Hz*erD1%2).T); #print(snd) #to compare with snd
            W_erAR[0,df+1] = np.count_nonzero(erD1)
            erAR[:,df+1] = erD1.T
    #    print((Hz*erAR) %2)    # check: All the columns shound be the syndrome vector





#        MI2 = np.argmin(W_erAR);        MI = 0        # A little cheating for now
        MI = np.argmin(W_erAR);        MI2 = 0        # A little cheating for now


        MinW = W_erAR[0,MI]
        MinW2 = W_erAR[0,MI2]

#        print('The Error: ',(erAR[:,MI]).T)
#        print(' Alternate Error: ',(erAR[:,MI2]).T)

        erAR_M = erAR[:,MI]
        erAR_M2 = erAR[:,MI2]

        if (np.shape(erAR_M)[0] > 1 ):
            erAR_M = erAR_M.T
        if (np.shape(erAR_M2)[0] > 1 ):
            erAR_M2 = erAR_M2.T



        ePe = erAR_M + erOG.T
        ePe2 = erAR_M2 + erOG.T




        if (MinW < BLEver):
            BLEver = MinW
            BLEver_At = rg
            if ( check2(ePe,Hx) ):
                BW_LEver_RC = 1
            else:
                BW_LEver_RC = 0

        if (MinW == FixWt and BW_Fst16 < 1):
            BW_Fst16 = 1
            BFst_16 = rg

            if ( check2(ePe,Hx) ):
                BW_RC_Fst16 = 1
            else:
                BW_RC_Fst16 = 0

        if (MinW == FixWt and BW_Rgt_16 < 1):
            if ( check2(ePe,Hx) ):
                BW_Rgt_16 = 1
                BRgt_16 = rg
            else:
                BW_Rgt_16 = 0
                BRgt_16 = 0


        if ( MinW < BMinW_ever ):

            if ( check2(ePe,Hx) ):
    #            print("1-(nke-nk): ",check2(ePe,Hx))
                MinER = erAR_M
                BMinW_ever = MinW
                BfoundA = rg
#                print('YaaaYyyyyy!!!! New Min Weight',BMinW_ever,'At rg=',rg)
#            else:
#                print('Not in the right class!',"The MinW was:",MinW)
#        else:
#            print('Dammit! No luck this time! at rg=',rg,"The MinW was:",MinW)


#    print('The decoded weight: ',BMinW_ever,' found At rg= ',BfoundA, 'for tryn= ',tryn)
#    print('The End')

    if (BfoundA < 0):
        UDE_n = UDE_n + 1
#        Trow = Einds
#        Trow = np.matrix(ttems[0:n])
        IntE2s = append(IntE2s, erOG.T , axis=0)

    Br_save[tryn,0] = tryn
    Br_save[tryn,1] = BFst_16
    Br_save[tryn,2] = BW_Fst16
    Br_save[tryn,3] = BW_RC_Fst16
    Br_save[tryn,4] = BRgt_16
    Br_save[tryn,5] = BW_Rgt_16
    Br_save[tryn,6] = BfoundA
    Br_save[tryn,7] = BMinW_ever
    Br_save[tryn,8] = BLEver
    Br_save[tryn,9] = BLEver_At
    Br_save[tryn,10] = BW_LEver_RC
    Br_save[tryn,11] = FixWt





#############################################################################################################
#############################################################################################################
#############################################################################################################
#print('IntE1s',IntE1s)
#print('IntE2s',IntE2s)


save(Fr_save,"./SunF2.sobj")
save(Ar_save,"./SunF1.sobj")
save(Br_save,"./SunF0.sobj")




#print('Summary:')
#print('Fst_16',Fst_16)
#print('W_Fst16',W_Fst16)
#print('W_RC_Fst16',W_RC_Fst16)
#print('W_Rgt_16',W_Rgt_16)
#print('Rgt_16',Rgt_16)

#print('MinW_ever',MinW_ever)
#print('foundA',foundA)
#print('LEver',LEver)
#print('LEver_At',LEver_At)
#print('W_LEver_RC',W_LEver_RC)


#This is Stamford Bridge

print('completa!')
︡e7be54e4-4a65-41b7-a9ab-6d21ab11caa4︡{"stderr":"Error in lines 160-160\nTraceback (most recent call last):\n  File \"/cocalc/lib/python3.11/site-packages/smc_sagews/sage_server.py\", line 1244, in execute\n    exec(\n  File \"\", line 1, in <module>\nNameError: name 'xrange' is not defined\n"}︡{"done":true}









