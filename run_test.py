#!/usr/bin/env python
'''
molecule:     sys.argv[1] (read from molecule.xyz)
functional:   sys.argv[2]
basis         sys.argv[3]
method        sys.argv[4]
                 scdm-g 
                 elf (paper)
                 lol 
                 grid
mkorbs        sys.argv[5]
                 scdm-g
                 floa (paper)
charge        int(sys.argv[6])
spin          int(sys.argv[7])-1


usage example:

python run_test_flosic.py  methane pbe,pbe 6-31G lol flo 0 1



needs:
              pyscf             (tested on 2.6.0)
              scipy             (tested on 1.15.1)
              numpy             (tested on 1.25.2) 
              pytictoc
              niflosic (separate module file)
              pytorch           (tested on 2.7.1)
'''

if __name__ == "__main__":

    from niflosic import RSIC, USIC,dump_scf_summary, lowdin_pop, scf_ener_fod
    from pyscf.gto import Mole 
    import sys
    from pyscf import dft, df
    from scipy import optimize
    
    import pyscf
    print('pyscf version:   ',pyscf.__version__)
    import numpy
    print('numpy version :',numpy.__version__)
    import scipy
    print('scipy version :',scipy.__version__)
    from pytictoc import TicToc
 

    molecule = sys.argv[1]
    functional = sys.argv[2]
    basis = sys.argv[3]
    method = sys.argv[4]
    mkorbs = sys.argv[5]
    charge = int(sys.argv[6])
    spin= int(sys.argv[7])-1
    mol = Mole()
    mol.charge=charge
    mol.spin=spin
    mol.fromfile(filename=molecule+'.xyz',format='xyz')
    mol.basis=basis
    mol.build()

 
    force_uks = False

#########################################################################
#                            DFT calculation
#########################################################################

    if spin == 0  and not force_uks: 
              mf0 = dft.RKS(mol)
    else:
              mf0 = dft.UKS(mol)
    mf = df.density_fit(mf0, 'def2-universal-jfit' )
    mf.xc = functional
    mf.grids.level = 5
    mf.verbose = 4
    mf.conv_tol =1.e-9
    mf.max_cycle=200
    t = TicToc()
    t.tic()
    print('*'*81)
    print('************************************ DFT SCF  ***********************************')
    print('*'*81)
    mf.kernel()
    print('*'*81)
    print('********************************* DFT SCF Done  *********************************')
    print('*'*81)
    t.toc()
    print('*'*81)
    C = mf.mo_coeff
    P = mf.make_rdm1()
    lowdin_pop(mol,P, mol.intor('int1e_ovlp') )
    mf.dip_moment()
#########################################################################
#                          NIFLOSIC calculation
#########################################################################


    if spin == 0 and not force_uks:  
              mf0 = RSIC(mol)
    else:
              mf0 = USIC(mol)

    mf = df.density_fit(mf0, auxbasis='def2-universal-jfit')
    mf.conv_tol =1.e-8
    mf.mo_coeff = C
    mf.xc = functional
    mf.method = method
    mf.mkorbs = mkorbs
    mf.sic='flo-sic'
    mf.verbose = 2
    mf.scale_sic = 1.00
    mf.grids.level = 5
    mf.max_cycle=300
    print('Functional ',mf.xc)
    print('SIC        ',mf.sic)
    print('Basis      ',mol.basis)
    print('charge ',mol.charge)
    print('spin   ',mol.spin)
    ener = 0.0
    print('*'*81)
    print('************************************ NI-FLOSIC  *********************************')
    print('*'*81)
    t.tic()
    mf.do_piv = True
    try:
      del mf.d
    except:
      pass
    r  = mf.kernel(dm0=P )
    ener = r[1]
    P = mf.make_rdm1()
   
    print('*'*81)
    print('********************************* NI-FLOSIC Done ********************************')
    print('*'*81)
    t.toc()
    print('*'*81)
    dm = mf.make_rdm1()
    EQR = mf.e_tot
    lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )
    mf.dip_moment()




    if spin == 0 and not force_uks:
          Eig = numpy.sort(mf.mo_energy)
          nocc= mol.nelec[0]
          Ehomo = Eig[nocc-1]
          Elumo = Eig[nocc]
          print(f'HOMO = {Ehomo:12.8f}  LUMO = {Elumo:12.8f} (Hartree)')
    else:
          Eig_a, Eig_b = mf.mo_energy
          Eig_a = numpy.sort(Eig_a)
          Eig_b = numpy.sort(Eig_b)
          nocc_a = mol.nelec[0]
          nocc_b = mol.nelec[1]
          Ehomo_a = Eig_a[nocc_a-1]
          Elumo_a = Eig_a[nocc_a]
          Ehomo_b = Eig_b[nocc_b-1]
          Elumo_b = Eig_b[nocc_b]
          print(f'alpha HOMO = {Ehomo_a:12.8f}  LUMO = {Elumo_a:12.8f} (Hartree)')
          print(f'beta  HOMO = {Ehomo_b:12.8f}  LUMO = {Elumo_b:12.8f} (Hartree)')








    if spin == 0 and not force_uks: 
       FOD = mf.fod
       fod = FOD.copy()
       fod=fod.flatten()
    else:
       FOD_a = mf.fod_a
       FOD_b = mf.fod_b
       fod_a = FOD_a.copy()
       fod_b = FOD_b.copy()
       fod_a = fod_a.flatten()
       fod_b = fod_b.flatten()
       FOD = numpy.vstack((FOD_a,FOD_b))
       

#########################################################################
#                         FLOSIC calculation
#########################################################################


    print('*'*81)
    print('********************************* Relaxing FODs *********************************')
    print('*'*81)
    mf.use_fod=True
    mf.testfd = True

#    mf.damp = 0.5
#    mf.diis_start_cycle = 6




    fod = FOD.flatten().copy()
    mf.max_cycle=120

    t.tic()
    solution =  optimize.minimize(scf_ener_fod, fod, method='L-BFGS-B', jac=True, args=(mf,mol),options={'gtol': 1.e-5 } )
    if solution.success:
       print('*'*81)
       print('*********************************    Success    *********************************')
       print('*'*81)
       t.toc()
       print('*'*81)
       print('displacement RMS = ', numpy.linalg.norm(solution.x -fod) )
       grad = numpy.linalg.norm(solution.jac)
       EFLO = mf.e_tot

       print('Gain in Energy = ', EFLO-EQR)
       print('Gradient FOD   = ', grad)

       dm = mf.make_rdm1()
       lowdin_pop(mol,dm, mol.intor('int1e_ovlp') )
       mf.dip_moment()
       if spin == 0 and not force_uks:
          Eig = numpy.sort(mf.mo_energy)
          nocc= mol.nelec[0]
          Ehomo = Eig[nocc-1]
          Elumo = Eig[nocc]
          print(f'HOMO = {Ehomo:12.8f}  LUMO = {Elumo:12.8f} (Hartree)')
       else:
          Eig_a, Eig_b = mf.mo_energy
          Eig_a = numpy.sort(Eig_a)
          Eig_b = numpy.sort(Eig_b)
          nocc_a = mol.nelec[0]
          nocc_b = mol.nelec[1]
          Ehomo_a = Eig_a[nocc_a-1]
          Elumo_a = Eig_a[nocc_a]
          Ehomo_b = Eig_b[nocc_b-1]
          Elumo_b = Eig_b[nocc_b]
          print(f'alpha HOMO = {Ehomo_a:12.8f}  LUMO = {Elumo_a:12.8f} (Hartree)')
          print(f'beta  HOMO = {Ehomo_b:12.8f}  LUMO = {Elumo_b:12.8f} (Hartree)')



    else: 
       print('*'*81)
       print('*******************************  FOD Opt failed *********************************')
       print('*'*81)
       t.toc()
       print('*'*81)







    
