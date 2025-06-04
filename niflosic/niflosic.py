from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from pyscf.gto import Mole
from pyscf import dft, scf, lo
from scipy.linalg import qr
import numpy as np
import pyscf
import scipy
import sys
import os

#python3 -m pip install rdkit-pypi
#https://greglandrum.github.io/rdkit-blog/posts/2023-02-04-working-with-conformers.html


def printM(a):
    for row in a:
        for col in row:
            print("{:9.6f}".format(col), end=" ")
        print("")

def printL(a):
    for i,r in enumerate(a):
          print(f"{r:9.6f}", end=" ")
          if ((i+1) % 8) == 0: print("")
    print("")

def printLi(s,a):
    print(s)
    for i,r in enumerate(a):
          print(f"{r:3d}", end=" ")
          if ((i+1) % 20) == 0: print("")
    print("")


def generate_molecule(fragment, fragment_name):
    number_of_atoms = fragment.GetNumAtoms()
    charge = Chem.GetFormalCharge(fragment)
    symbols = [a.GetSymbol() for a in fragment.GetAtoms()] 

    fragment= Chem.Mol(fragment,confId=0)  # just pick one conformer
    for i,conf in enumerate(fragment.GetConformers()):
         file = ""
         for atom,symbol in enumerate(symbols):
             p = conf.GetAtomPosition(atom)
             line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
             file+=line
         charge = charge
#    print(file)
#    quit()
    return file, charge

def get_fragments(mol,name):
    fragment_names = []
    fragments = Chem.GetMolFrags(mol,asMols=True)
    labels = ["A","B","C"]
    for label,fragment in zip(labels,fragments):
        fragment_names.append(name+label)
    
    return fragments, fragment_names


def generate_conformations(fragments, max_confs=20):
    for fragment in fragments:
        rot_bond = rdMolDescriptors.CalcNumRotatableBonds(fragment)
        confs = min(3 + 3*rot_bond,max_confs)
        AllChem.EmbedMultipleConfs(fragment,numConfs=confs)
    
    return fragments


def get_coords(molecule_smiles):
   molecule = Chem.MolFromSmiles(molecule_smiles)
   molecule=Chem.rdmolops.AddHs(molecule)
   fragments, fragment_names = get_fragments(molecule,molecule_smiles)
   fragments = generate_conformations(fragments)
   for fragment, fragment_name in zip(fragments, fragment_names):
       input_pyscf,charge = generate_molecule(fragment, molecule_smiles)
       atomic_count = 0
       for atom in fragment.GetAtoms():
          atomic_count += atom.GetAtomicNum()
#       print(atomic_count)

   spin = 0 # do a better calculation here
   return  str(input_pyscf), charge, spin


def printAt(s,a):
    for i,row in enumerate(a):
        print(f"{s[i]:s} ", end="")
        for col in row:
            print(f" {col:9.6f}", end=" ")
        print("")

def printC(a):
    for i,row in enumerate(a):
        for col in row:
            print(f" {col:9.6f}", end=" ")
        print("")



def printM(a):
    for row in a:
        for col in row:
            print("{:9.6f}".format(col), end=" ")
        print("")

def printL(a):
    for i,r in enumerate(a):
          print(f"{r:9.6f}", end=" ")
          if ((i+1) % 8) == 0: print("")
    print("")

def printLi(s,a):
    print(s)
    for i,r in enumerate(a):
          print(f"{r:3d}", end=" ")
          if ((i+1) % 20) == 0: print("")
    print("")


def indmax(l):
    sorted_list = sorted(l, reverse=True)
    max_element = sorted_list[0]
    return l.tolist().index(max_element)

from functools import reduce


def docsic_g(mf,mol,P,orbs='flosic'):
    '''
    initiallize FODs using QR grid decomposition
    
    '''
    grid = pyscf.dft.gen_grid.Grids(mol)
#   check https://pyscf.org/_modules/pyscf/dft/LebedevGrid.html#MakeAngularGrid
    grid.atom_grid = (30, 194)  
#    grid.atom_grid = {'C': (6, 110)} 
    S = mol.intor('int1e_ovlp')
#    print('Trace of P = ', np.trace(S@P))
#    quit()
    grid.build()


    coords = grid.coords
    weights = grid.weights

#LEBEDEV_ORDER = {
#    0  : 1   ,
#    3  : 6   ,
#    5  : 14  ,
#    7  : 26  ,
#    9  : 38  ,
#    11 : 50  ,
#    13 : 74  ,
#    15 : 86  ,
#    17 : 110 ,
#    19 : 146 ,
#    21 : 170 ,
#    23 : 194 ,
#    25 : 230 ,
#    27 : 266 ,
#    29 : 302 ,
#    31 : 350 ,
#    35 : 434 ,
#    41 : 590 ,
#    47 : 770 ,
#    53 : 974 ,
#    59 : 1202,
#    65 : 1454,
#    71 : 1730,
#    77 : 2030,
#    83 : 2354,
#    89 : 2702,
#    95 : 3074,
#    101: 3470,
#    107: 3890,
#    113: 4334,
#    119: 4802,
#    125: 5294,
#    131: 5810


# Uniform grid
#    coords = []
#    for ix in np.arange(-10, 10, .2):
#        for iy in np.arange(-10, 10, .2):
#            for iz in np.arange(-10, 10, .2):
#                coords.append((ix,iy,iz))
#    coords = np.array(coords)
#

    ao = mol.eval_gto('GTOval', coords)
    ni =  dft.numint

# decide RKS or UKS
    if P.ndim == 2: is_r = True
    else: is_r = False
    if is_r:
        orbitals = mf.mo_coeff
        mo = ao @ orbitals
        moe = mf.mo_energy 
        nocc =    mol.nelec[0] 
        energies = mf.mo_energy
        fo = mo[:,:nocc]
        gao = mol.eval_gto('GTOval_ip', coords)
        aogao = np.array((ao,gao[0],gao[1],gao[2]))
        rho,gx,gy,gz,tau =  0.5*ni.eval_rho(mol, aogao, P ,xctype ='MGGA',with_lapl=False)
        sqrt_rho = np.sqrt(rho)   
        grad2 = gx**2+gy**2+gz**2
        Dh     =  3.0/10.0*(3.*np.pi**2)**(2./3.)*rho**(5./3.)
        lol =  1./(1.+2.*tau/Dh)
        lol = np.nan_to_num(lol,nan=0.0, posinf=0.0, neginf=0.0)
#        sqrt_rho = np.nan_to_num(sqrt_rho,nan=0.0, posinf=0.0, neginf=0.0)
        vwke  = grad2/8.0/rho
        D = tau - vwke
        elf = 1.0/( 1.0 + (D/Dh)**2 )
        elf = np.nan_to_num(elf,nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(nocc): 
             maxf =  np.sqrt(max(fo[:,i]**2 / rho   ))
             fo[:,i] =  fo[:,i]/sqrt_rho /maxf * elf    #/  *elf #  * elf   # *lol    /maxf *elf#  *lol


    else:   
        mo_a = ao @ mf.mo_coeff[0]
        mo_b = ao @ mf.mo_coeff[1]
        nocc_a = mol.nelec[0]
        nocc_b = mol.nelec[1]
        fo_a = mo_a[:,:nocc_a] 
        fo_b = mo_b[:,:nocc_b] 
        gao = mol.eval_gto('GTOval_ip', coords)
        aogao = np.array((ao,gao[0],gao[1],gao[2]))
        rho_a,gx_a,gy_a,gz_a,tau_a =  ni.eval_rho(mol, aogao, P[0] ,xctype ='MGGA',with_lapl=False)
        rho_b,gx_b,gy_b,gz_b,tau_b =  ni.eval_rho(mol, aogao, P[1] ,xctype ='MGGA',with_lapl=False)
        sqrt_rho_a = np.sqrt(rho_a)
        sqrt_rho_b = np.sqrt(rho_b)
        grad2_a = gx_a**2+gy_a**2+gz_a**2
        grad2_b = gx_b**2+gy_b**2+gz_b**2
        Dh_a     = 3.0/10.0*(3*np.pi**2)**(2./3.)*rho_a**(5./3.)
        Dh_b     = 3.0/10.0*(3*np.pi**2)**(2./3.)*rho_b**(5./3.)
        lol_a = 1./(1.+2.*tau_a/Dh_a)
        lol_b = 1./(1.+2.*tau_b/Dh_b)
        lol_a =  np.nan_to_num(lol_a,nan=0.0, posinf=0.0, neginf=0.0)
        lol_b =  np.nan_to_num(lol_b,nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(nocc_a): 
            maxf = np.sqrt(max(fo_a[:,i]**2))
            fo_a[:,i] = fo_a[:,i]*lol_a/maxf
        for i in range(nocc_b): 
            maxf = np.sqrt(max(fo_b[:,i]**2))
            fo_b[:,i] = fo_b[:,i]*lol_b/maxf

    if is_r:
        Q,  R, piv = scipy.linalg.qr(fo.T , pivoting=True);
        PIV = piv[0:nocc]
    else:
        Q,  R, piv = scipy.linalg.qr(fo_a.T , pivoting=True);
        PIV_a = piv[0:nocc_a]
        Q,  R, piv = scipy.linalg.qr(fo_b.T , pivoting=True);
        PIV_b = piv[0:nocc_b]


#    print('***** sqrt_rho = ', sqrt_rho[PIV])
#    print('***** elf      = ', elf[PIV])
#    print('***** lol      = ', lol[PIV])
     
# Add coefficients to ks object
###############################
    if is_r:
        if orbs=='scdm-g' or orbs=='flosic' : 
           if orbs=='scdm-g' : Y =  ao[PIV].T 
           if orbs=='flosic' : Y =  ao[PIV].T / sqrt_rho[PIV]
           G =   P @ Y
           O = Y.T @ P @ Y
           O12 = scipy.linalg.sqrtm(O).real
           Om12= scipy.linalg.pinv(O12).real 
           X = G @ Om12 
           P_loc = np.einsum('ia,ja->aji',X,X)
           SPloc = np.sum(P_loc, axis=0) 
           assert np.linalg.norm(SPloc-P) <= 1E-9, 'Localized Densities'
           mf.orbs = X.T
           mf.ploc = P_loc
           mf
        elif method=='lol': pass
        else: raise ValueError('Wrong orbital construction (use scdm-g,flosic)')

#




    if is_r:
        return coords[PIV]
    else:
        return coords[PIV_a], coords[PIV_b]







def print_xyz(mol, FODs,fname):
    '''
    '''
    sys.stdout = open( fname+'.xyz'  , 'w')
    atom_coords = mol.atom_coords() / 1.8897259886  # convert Bohr to A

    # figure out if FODs has alpha and beta
    if FODs.__class__==tuple:
       FODs_a = FODs[0]  
       FODs_b = FODs[1]  
       is_r  = False
    else: 
       is_r  = True

    symbols =[]
    for i in range(len(atom_coords)):
        symbols.append(mol.atom_symbol(i))


    if len(symbols) == 1:   # if there is only one atom don't add it to the xyz file for visualization
        atom = True 
    else:
        atom = False


    if is_r:
        sFOD=[]
        for i in range(len(FODs)):
            sFOD.append('X')
        if not atom: print(len(atom_coords) + len(FODs))
        if     atom: print( len(FODs))
    else:
        sFOD_a=[]
        for i in range(len(FODs_a)):
            sFOD_a.append('X')
        sFOD_b=[]
        for i in range(len(FODs_b)):
            sFOD_b.append('Z')
        if not atom: print(len(atom_coords) + len(FODs_a) + len(FODs_b))
        if     atom: print( len(FODs_a) + len(FODs_b))

    print("")
    if not atom: printAt(symbols,atom_coords)
    if is_r:
        printAt(sFOD,FODs/1.8897259886)   # Bohr to A
    else:
        printAt(sFOD_a,FODs_a/1.8897259886)   # Bohr to A
        printAt(sFOD_b,FODs_b/1.8897259886)   # Bohr to A

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ fname+'.xyz file' )


# check the units of all NRLMOL files

def write_frmidt(mol,FODs):
    sys.stdout = open( 'FRMIDT'  , 'w')
    # figure out if FODs has alpha and beta
    if FODs.__class__==tuple:
       FODs_a = FODs[0]  
       FODs_b = FODs[1]  
       is_r  = False
    else: 
       is_r  = True

    nu = mol.nelec[0]
    nd = mol.nelec[1]
    if is_r:
        print(f'{nu:5d}  {0:5d}')
        printC(FODs)
    else:
        print(f'{nu:5d}  {nd:5d}')
        printC(FODs_a)
        printC(FODs_b)


    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ 'FRMIDT' )


P1 ='''GGA-PBEIBP*GGA-PBEIBP
NONE
''' 
def write_cluster(mol):
    '''
    '''
    sys.stdout = open( 'CLUSTER'  , 'w')
    # CLUSTER is in Bohr 
    ac = mol.atom_coords()

    # A to Bohr: 1.8897259886
    #print(ac)

    symbols =[]
    nat = len(ac)
    for i in range(nat):
        symbols.append(mol.atom_symbol(i))
    # LDA-PW91*LDA-PW91  GGA-PBE*GGA-PBE

    Z = []
    for ia in range(mol.natm):
      Z.append(mol.atom_charge(ia))


    nu = mol.nelec[0]
    nd = mol.nelec[1]
    spin = mol.spin
    charge = mol.charge
    p1 = P1 + str(nat) + "\n"
    for i, s in enumerate(symbols):
      p1 += f'{ac[i][0]:17g} {ac[i][1]:17g} {ac[i][2]:17g} {Z[i]:3g} ALL \n'
    p1 += f'{charge:5d} {spin:5d} \n'

    print(p1)   # CLUSTER is in p1
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ 'CLUSTER' )


p2 = '''# Put Y,N or number next to the equal sign to determine execution
# Don't forget the quotation marks for the letters
# All variables in this list end with v

&input_data
ATOMSPHV      = 'N'
BASISV        = 'DEFAULT' ! Specify basis for calculation(basis.txt)
DFTD3V        = 'N' ! Set to Y to do include Grimmes DFT-D3 dispersion
DIAG1V        =  1  ! diagonalization to use on regular arrays (diagge.f90)
DIAG2V        =  1  ! diagonalization to use on packed arrays (diag_dspgv.f90)
DIAG3V        =  0  ! diagonalization to use on parallel (sdiagge_n.f90)
DMATV         = 'N' ! Create/use/mix density matrix
DOSOCCUV      = 'N' ! Controls wether to calculate density of states
FIXMV         = 'Y' ! Fix spin moment
FOD_LOOPV     = 'N' ! Inernal FOD loop for frozen density optimization
FOD_OPT1V     = 'LBFGS' ! FOD_OPT: algorithm (LBFGS/CG)
FOD_OPT2V     = 'N' ! FOD_OPT: scaling of r and F
FOD_OPT3V     =  0  ! FOD_OPT (0)no constraint (1)fix1s (2)fullForce (3)shellOPT (4)freeFOD
FRAGMENTV     = 'N' ! Process CLUSTER in fragments
JNTDOSV       = 'N' ! This calculates joint density of states (DFA only)
MAXSCFV       = 200 ! Maximum SCF iterations
MIXINGV       = 'H' ! (H)amiltonian (P)otential (D)ensity matrix mixing
MOLDENV       = 'N' ! Use molden and wfx driver
NONSCFV       = 'N' ! Set to Y to do a non SCF calculation
NONSCFFORCESV = 'N' ! Set to Y to calculate forces in a non SCF calculation
NWFOUTV       = 10  ! Write WFOUT file for every N-th iteration
POPULATIONV   = 'N' ! Population analysis
RHOGRIDV      = 'N' ! Set to Y to execute RHOGRID
SCFTOLV       = 1.0D-7 ! SCF tolerance
SPNPOLV       = 'N' ! Run spin polarized calculation from CLUSTER
MESHSETTINGV  =  1  ! Mesh recommended for (0)LDA/PBE, (1)SCAN, (2)rSCAN
SYMMETRYV     = 'N' ! Set to Y to detect symmetry
SYMMMODULEV   = 'N' ! (Y) Use symmtery and approx Ham. (N) use Jacobi rotation (SIC only)
UNIHAMV       = 'N' ! Set to Y to use unified Hamiltonian formalism in SCF-SIC (SIC only)
WFGRIDV       = 'N' ! set to Y to write orbitals in cube format (DFA only)
WFFRMV        = 'N' ! set to Y to write Fermi orbitals in cube format (SIC only)
&end
'''
def write_nrlmol_input(mol):
    sys.stdout = open( 'NRLMOL_INPUT.DAT'  , 'w')
    print(p2)
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ 'NRLMOL_INPUT.DAT' )




p3='''   1
1.0  0.0  0.0
0.0  1.0  0.0
0.0  0.0  1.0
'''
def write_frmgrp(mol):
    sys.stdout = open( 'FRMGRP'  , 'w')
    print(p3)   
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ 'FRMGRP' )



def dispersion(mf,mol):
    P = mf.make_rdm1()
    if P.ndim == 2: 
      is_r = True
      P_loc =  mf.ploc
    else: 
      is_r = False
      P_loc_a =  mf.ploca
      P_loc_b =  mf.plocb


    grid = pyscf.dft.gen_grid.Grids(mol)
    grid.atom_grid =  (40, 350)
    grid.build() 
    coords = grid.coords
    weights = grid.weights



    xctype = dft.xcfun.xc_type(mf.xc)
    if xctype == 'LDA':
        deriv = 0 
    else: 
        deriv = 1 
    ao_value = mol.eval_gto('GTOval', coords)
    S = mol.intor('int1e_ovlp')

    if not is_r: P_loc = np.concatenate([P_loc_a, P_loc_b])

    totDisp = 0.0
    disp = []
    for dm in P_loc:
        rho = 0.5*dft.numint.eval_rho(mol, ao_value, dm, xctype='LDA')
        rx = sum(rho * weights*coords.T[0] )
        ry = sum(rho * weights*coords.T[1] )
        rz = sum(rho * weights*coords.T[2] )
        rr = rx**2 + ry**2 + rz**2
        rx2 = sum(rho * weights*coords.T[0]**2 )
        ry2 = sum(rho * weights*coords.T[1]**2 )
        rz2 = sum(rho * weights*coords.T[2]**2 )
        r2 = rx2 + ry2 + rz2
        totDisp += r2 - rr 
        disp = np.append(disp,r2 - rr)
#        print(2.*(r2 - rr))
    if is_r: 
        totDisp *= 2.0
        disp    *= 2.0

    return np.sort(disp), totDisp





def hxcsic(mf,mol):
    P = mf.make_rdm1()
    if P.ndim == 2: 
      is_r = True
      P_loc =  mf.ploc
    else: 
      is_r = False
      P_loc_a =  mf.ploca
      P_loc_b =  mf.plocb

    grid = pyscf.dft.gen_grid.Grids(mol)
#    grid.atom_grid =  (200, 350)
    grid.build() 
    coords = grid.coords
    weights = grid.weights

    xctype = dft.xcfun.xc_type(mf.xc)

    if xctype == 'LDA': deriv = 0 
    else: deriv = 1 
    ao_value = dft.numint.eval_ao(mol, coords, deriv=deriv)

    S = mol.intor('int1e_ovlp')

    exchange = mf.xc.split(',')[0]
    correlation = mf.xc.split(',')[1]

    if not is_r: P_loc = np.concatenate([P_loc_a, P_loc_b])

    HXC = 0.0
    J = mf.get_j(mol,dm=P_loc)  # can we do all at once?
    if is_r:  J *=.25
    for i, dm in enumerate(P_loc):
        rho = 0.5*dft.numint.eval_rho(mol, ao_value, dm, xctype=xctype)
        rhoz = np.zeros_like(rho)
        if xctype == 'LDA': n=rho
        else: n=rho[0]
        e_x  = dft.xcfun.eval_xc(exchange+','   , (rho, rhoz), spin=1,deriv=0)[0]
        e_c  = dft.xcfun.eval_xc(','+correlation, (rho, rhoz), spin=1,deriv=0)[0]
        Ex = sum(e_x * weights *n )
        Ec = sum(e_c * weights *n )
        rhoint = sum(n * weights)
        Har= 0.5*np.trace(dm @ J[i])
        HXC += -Ex -Ec -Har
        print('e-count, Hartree, XC = ',rhoint,-Har, -Ex-Ec)
    if is_r:  HXC *= 2.0

    return HXC

import mendeleev

def run_atoms(atom_list,ionize):
    summary = []
    for element in mendeleev.elements.get_all_elements()[atom_list]  :
       symbol = element.symbol
       if ionize == 0 : 
            spin = element.ec.unpaired_electrons()
            EC = element.ec
            ch=''
       elif ionize == 1 : 
            spin = element.ec.ionize().unpaired_electrons()
            EC = element.ec.ionize()
            ch='+1'
       elif ionize == 2 : 
            spin = element.ec.ionize().ionize().unpaired_electrons()
            EC = element.ec.ionize().ionize()
            ch='+2'
       elif ionize == 3 : 
            spin = element.ec.ionize().ionize().ionize().unpaired_electrons()
            EC = element.ec.ionize().ionize().ionize()
            ch='+3'
       else:
            raise Exception("Wrong ionize number. Must be 0-3")
       ne = element.atomic_number - ionize
       nu = (ne + spin)/2
       nd = (ne - spin)/2
#      method can be 'rho', 'max', or 'not'. They will give different FODs
       mf  =  run_dft(symbol+ch,smiles='atom',spin=spin,charge=ionize)
       conv= mf[1].converged
       summary.append([ne,symbol+ch, spin,nu,nd, conv, element.ec] )
       if conv:
           submit_flosic(symbol+ch)
           get_atomic_population(mf[1],symbol+ch)
    

    return summary   




def run_dft(molecule_sm,smiles='smiles',spin=0,charge=0,orbs='flosic'):
    '''
    '''
    from pyscf import sgx
    from pyscf import gto, scf
    mol = Mole()
    string_data =''


    if smiles=='smiles':
       mol.atom, mol.charge, mol.spin = get_coords(molecule_sm)
       mol.spin=spin
       mol.charge=charge
#       mol.unit='Bohr'
    elif smiles=='xyz':
       if molecule_sm[-4:]!='.xyz' : molecule_sm=molecule_sm+'.xyz'
       mol.charge = charge
       mol.spin=spin
       mol.fromfile(filename=molecule_sm,format='xyz')
    elif smiles=='atom':
       mol.charge=charge
       mol.spin=spin
       mol.atom = molecule_sm+'0.0 0.0 0.0\n'
    else:
       print('ERROR') # catch the error here
    mol.basis =  'sto-3G'
    mol.cart=True
    mol.build()
    
    new_name = molecule_sm.replace(".xyz", "")


#   DFT calculation
    print('*************************************************')
    if spin ==0 and smiles!='atom':   # change as needed
        mf = dft.RKS(mol)
        print('This calculation will be RKS')
        string_data+='This calculation will be RKS'+'\n'
    else:
        mf = dft.UKS(mol)
        print('This calculation will be UKS')
        string_data+='This calculation will be UKS'+'\n'


    mf.verbose = 3
    mf.xc = 'pbe,pbe'

    mf.init_guess = 'atoms'
    mf.max_cycle=500
    mf.grids.level = 5
    mf.damp = 0.8
    mf.diis_start_cycle = 5
#    mf.grids.prune = dft.gen_grid.sg1_prune
    mf.conv_tol=1.e-5
    print('Starting '+new_name+" calculation")
    string_data+= 'Starting '+new_name+" calculation"+'\n'
    print('Functional ',mf.xc)
    string_data+= 'Functional ' + mf.xc+'\n'
    print('Basis      ',mol.basis)
    string_data+= 'Basis      '+mol.basis+'\n'
    print('charge ',mol.charge)
    string_data+= 'charge '+str(mol.charge)+'\n'
    print('spin   ',mol.spin)
    string_data+= 'spin   '+str(mol.spin)+'\n'
    mf.kernel()     
#    dm = mf.make_rdm1()
#    mf.xc = 'slater,vwn5'
#    mf.kernel(dm0=dm)     

    dm = mf.make_rdm1()
    S = mol.intor('int1e_ovlp')

    FODs = docsic_g(mf,mol,dm,orbs)
    print('Orbitals evaluated using '+orbs)
    string_data+= 'Orbitals evaluated and tested using '+orbs + '\n'
    if mf.converged: 
        print(new_name+" converged")
        string_data+= new_name+" converged"+'\n'
        try:
          os.mkdir(new_name)
        except FileExistsError:
          os.system("rm -rf '"+new_name+"-old'")
          os.system("mv -f '"+new_name+"' '"+new_name+"-old'")
          os.mkdir(new_name)
        except OSError as err:
          print(f"Operating system error: {err.strerror}.")
          string_data+=f"Operating system error: {err.strerror}."  +'\n'
        else:
          print(f"Successfully created the '{new_name}' folder")
          string_data+=f"Successfully created the '{new_name}' folder" +'\n'
        os.chdir(new_name)
        print_xyz(mol, FODs,new_name+".fods")
        write_cluster(mol)
        write_nrlmol_input(mol)
        write_frmgrp(mol)
        write_frmidt(mol,FODs)
        f = open("ini_fod.txt", "w")
        f.write(string_data)
        f.close()
        os.chdir('..')
    else:
        print('Failed to Converge. '+new_name+".fods.xyz file not generated." )


# here calculate the dispersion
#    disp,totDisp = dispersion(mf,mol)
#    print('Dispersion = ', totDisp)
#    print('array = ', disp)
#
#    hxc = hxcsic(mf,mol)
#    print('SIC Energy = ',hxc)

# do we calculate E_PZ here?



    return mol, mf

# from pyscf
def mulliken_pop(mol, dm, s=None, verbose=2):
    r'''Mulliken population analysis

    .. math:: M_{ij} = D_{ij} S_{ji}

    Mulliken charges

    .. math:: \delta_i = \sum_j M_{ij}

    Returns:
        A list : pop, charges

        pop : nparray
            Mulliken population on each atomic orbitals
        charges : nparray
            Mulliken charges
    '''
    if s is None: s = get_ovlp(mol)
#    log = logger.new_logger(mol, verbose)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        pop0 = np.einsum('ij,ji->i', dm, s).real
        pop1 = pop0
    else: # ROHF
        pop0 = np.einsum('ij,ji->i', dm[0], s).real
        pop1 = np.einsum('ij,ji->i', dm[1], s).real

    print(' ** Mulliken pop  **')
    pop_ang0=np.zeros(6)
    pop_ang1=np.zeros(6)
    print(f'                           up        down')
    for i, s in enumerate(mol.ao_labels(None)):
        print(f'pop of {s[1][:2]:s} {s[2]:s}{s[3]:6s} {pop0[i]:10.5f}  {pop1[i]:10.5f}')
        if 's' in s[2]: 
           pop_ang0[0] +=pop0[i]
           pop_ang1[0] +=pop1[i]
        if 'p' in s[2]: 
           pop_ang0[1] +=pop0[i]
           pop_ang1[1] +=pop1[i]
        if 'd' in s[2]: 
           pop_ang0[2] +=pop0[i]
           pop_ang1[2] +=pop1[i]
        if 'f' in s[2]: 
           pop_ang0[3] +=pop0[i]
           pop_ang1[3] +=pop1[i]
        if 'g' in s[2]: 
           pop_ang0[4] +=pop0[i]
           pop_ang1[4] +=pop1[i]
        if 'h' in s[2]: 
           pop_ang0[5] +=pop0[i]
           pop_ang1[5] +=pop1[i]
    
    print(' ** angular pop  **')
    print(f'ang        up        down')
    if pop_ang0[0]+pop_ang1[0]>0.0: 
         print(f's  {pop_ang0[0]:10.5f}  {pop_ang1[0]:10.5f} ')
    if pop_ang0[1]+pop_ang1[1]>0.0: 
         print(f'p  {pop_ang0[1]:10.5f}  {pop_ang1[1]:10.5f} ')
    if pop_ang0[2]+pop_ang1[2]>0.0: 
         print(f'd  {pop_ang0[2]:10.5f}  {pop_ang1[2]:10.5f} ')
    if pop_ang0[3]+pop_ang1[3]>0.0: 
         print(f'f  {pop_ang0[3]:10.5f}  {pop_ang1[3]:10.5f} ')
    if pop_ang0[4]+pop_ang1[4]>0.0: 
         print(f'g  {pop_ang0[4]:10.5f}  {pop_ang1[4]:10.5f} ')
    if pop_ang0[5]+pop_ang1[5]>0.0: 
         print(f'h  {pop_ang0[5]:10.5f}  {pop_ang1[5]:10.5f} ')
  


#    print(' ** Mulliken atomic charges  **')
#    chg = np.zeros(mol.natm)
#    spn = np.zeros(mol.natm)
#    for i, s in enumerate(mol.ao_labels(fmt=None)):
#        chg[s[0]] += pop0[i] + pop1[i]
#        spn[s[0]] += pop0[i] - pop1[i]
#    chg = mol.atom_charges() - chg
#    for ia in range(mol.natm):
#        symb = mol.atom_symbol(ia)
#        print(symb)
#        print(f'charge of  {ia:3d}{symb:s} =  {chg[ia]:10.5f}')
#        print(f'spin   of  {ia:3d}{symb:s} =  {spn[ia]:10.5f}')
    return 


# atomic population
def get_atomic_population(mf,molecule_sm):
        dm = mf.make_rdm1()
        S = mf.mol.intor('int1e_ovlp')
        os.chdir(molecule_sm)
        sys.stdout = open( molecule_sm+'.population.dat'  , 'w')
        mulliken_pop(mf.mol, dm,S , verbose=5)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        os.chdir('..')
        print('Population from DFT printed in '+ molecule_sm+'.population.dat' )




def submit_flosic(name):
    os.chdir(name)
    sys.stdout = open( 'submit.sb'  , 'w')
    slurmtxt='''#!/bin/bash --login
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --job-name "'''+name+'''"
module load intel/2022b
/mnt/home/peral1jCMICH/Work/FLOSIC/gen-FOD/flosic_code/flosic/nrlmol_exe >& output 

module load GCCcore/12.3.0
module load Python/3.11.3
python3 ../flosic_to_xyz.py 

''' 
 
    print(slurmtxt)   
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print('Wrote '+ 'submit.sb' )
#    os.system("sbatch submit.sb")
    print('Job not submitted')
    os.chdir('..')
    return




