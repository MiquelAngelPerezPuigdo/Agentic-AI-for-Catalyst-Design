import hashlib
import random

pd_true_ranking = [
    'COc1ccc(OC)c(c1P(C23CC4CC(CC(C4)C2)C3)C56CC7CC(CC(C7)C5)C6)-c8c(cc(cc8C(C)C)C(C)C)C(C)C', 'N1(C2=CC=CC=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)CCOCC1', 'COc1ccc(OC)c(c1P(C(C)(C)C)C(C)(C)C)-c2c(cc(cc2C(C)C)C(C)C)C(C)C', 'CC1=C(C)C(C)=C(C)C(C(C(C(C)C)=CC(C(C)C)=C2)=C2C(C)C)=C1P(C(C)(C)C)C(C)(C)C', 'CC1=C(C2=C(C(C)C)C=C(C(C)C)C=C2C(C)C)C(P(C(C)(C)C)C(C)(C)C)=C(OC)C=C1', 
    'CCCCC1=C(C(=C(C(=C1F)F)C2=C(C=C(C(=C2C(C)C)C3=C(C(=CC=C3)OC)P(C45CC6CC(C4)CC(C6)C5)C78CC9CC(C7)CC(C9)C8)C(C)C)C(C)C)F)F', 'CC1=C(C2=C(C(C)C)C=C(C(C)C)C=C2C(C)C)C(P(C(C)(C)C)C(C)(C)C)=C(C)C(OC)=C1C', 'CN(C)c1ccccc1P([C@@]23C[C@@H]4C[C@@H](C[C@@H](C4)C2)C3)[C@@]56C[C@@H]7C[C@@H](C[C@@H](C7)C5)C6', 'COc1c(P(C2CCCCC2)C3CCCCC3)c(c4c(C(C)C)cc(C(C)C)cc4C(C)C)c(OC)cc1','C1[C@H]2C[C@H]3C[C@@H]1C[C@@](C2)(C3)P(Cc4ccccc4)[C@@]56C[C@@H]7C[C@@H](C[C@@H](C7)C5)C6', 
    'c8ccc(c1ccccc1P(C34CC2CC(CC(C2)C3)C4)C67CC5CC(CC(C5)C6)C7)cc8', 'CC(C)C1=CC(C(C)C)=CC(C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2', 'COc1cccc(OC)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4', 'CC(C)Oc1cccc(OC(C)C)c1-c2ccccc2P(C3CCCCC3)C4CCCCC4', 'CC(C)[Si](Oc1ccccc1P([C@]23C[C@H]4C[C@H](C[C@H](C4)C2)C3)[C@]56C[C@H]7C[C@H](C[C@H](C7)C5)C6)(C(C)C)C(C)C', r'C\C(P(C(C)(C)C)C(C)(C)C)=C(/c1ccccc1)c2ccccc2', 
    'C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C', 'CN(C)c1ccccc1-c2ccccc2P(C(C)(C)C)C(C)(C)C'
]

ni_true_ranking = [
    'CCCC(CCC)[C@H]1COC(C2=N[C@@H](C(CCC)CCC)CO2)=N1', 'CC1(C)CN2C(C3=N[C@@H](C(C)C)CN3C1)=N[C@@H](C(C)C)C2', 'CC([C@H]1CN(C2=CC=CC(C(C)(C)C)=C2)C(C3=N[C@@H](C(C)C)CN3C4=CC(C(C)(C)C)=CC=C4)=N1)C', 
    'C[C@@H](CC)[C@H]1COC(C2=N[C@@H]([C@@H](C)CC)CO2)=N1', 'CC([C@H]1COC(C2=N[C@@H](C(C)C)CO2)=N1)C', 'CC([C@H]1CN2C(C3=N[C@@H](C(C)C)CN3CCC2)=N1)C', 'C1(C2=N[C@@H](C3CCCCC3)CO2)=N[C@@H](C4CCCCC4)CO1', 
    'CC([C@H]1CN2C(C3=N[C@@H](C(C)C)CN3[C@@H](C4=CC=CC=C4)[C@@H]2C5=CC=CC=C5)=N1)C', 'CC([C@H]1CN(C2=CC=CC=C2)C(C3=N[C@@H](C(C)C)CN3C4=CC=CC=C4)=N1)C', 'CC([C@H]1CN(C2=CC=C(C(C)(C)C)C=C2)C(C3=N[C@@H](C(C)C)CN3C4=CC=C(C(C)(C)C)C=C4)=N1)C', 
    'CC([C@H]1CN(C2=CC=C(C(F)(F)F)C=C2)C(C3=N[C@@H](C(C)C)CN3C4=CC=C(C(F)(F)F)C=C4)=N1)C', 'C1(C2=N[C@@H](C3CC3)CO2)=N[C@@H](C4CC4)CO1', 'CC([C@H]1CN(C2=CC=CC(C(F)(F)F)=C2)C(C3=N[C@@H](C(C)C)CN3C4=CC(C(F)(F)F)=CC=C4)=N1)C', 
    'CC([C@H]1CN2C(C3=N[C@@H](C(C)C)CN3CC2)=N1)C', 'C1(C2=N[C@@H](C3=CC=CC=C3)CO2)=N[C@@H](C4=CC=CC=C4)CO1', 'CC(C)C[C@H]1COC(C2=N[C@@H](CC(C)C)CO2)=N1', 'C[C@H]1COC(C2=N[C@@H](C)CO2)=N1', 
    'C1(C2=N[C@@H](CC3=CNC4=C3C=CC=C4)CO2)=N[C@@H](CC5=CNC6=C5C=CC=C6)CO1', 'CCCCCC[C@H]1COC(C2=N[C@@H](CCCCCC)CO2)=N1', 'COC(C=CC=C1)=C1C[C@H]2COC(C3=N[C@@H](CC4=CC=CC=C4OC)CO3)=N2', 
    'COC(C=C1)=CC=C1C[C@H]2COC(C3=N[C@@H](CC4=CC=C(OC)C=C4)CO3)=N2', 'CC(C=C1)=CC=C1C[C@H]2COC(C3=N[C@@H](CC4=CC=C(C)C=C4)CO3)=N2', 'O=[N+](C(C=C1)=CC=C1C[C@H]2COC(C3=N[C@@H](CC4=CC=C([N+]([O-])=O)C=C4)CO3)=N2)[O-]', 
    'C1(C2=N[C@@H](CC3=CC=C(C=CC=C4)C4=C3)CO2)=N[C@@H](CC5=CC(C=CC=C6)=C6C=C5)CO1', 'C1(C2=N[C@@H](CC3=CC=CC=C3)CO2)=N[C@@H](CC4=CC=CC=C4)CO1', 'CC(OC(C=C1)=CC=C1C[C@H]2COC(C3=N[C@@H](CC4=CC=C(OC(C)(C)C)C=C4)CO3)=N2)(C)C', 
    'C1(C2=N[C@@H](CC3=CSC4=C3C=CC=C4)CO2)=N[C@@H](CC5=CSC6=C5C=CC=C6)CO1', 'COC1=CC(OC)=CC(C[C@H]2COC(C3=N[C@@H](CC4=CC(OC)=CC(OC)=C4)CO3)=N2)=C1', 'C1(C2=N[C@@H](C(C3=CC=CC=C3)C4=CC=CC=C4)CO2)=N[C@@H](C(C5=CC=CC=C5)C6=CC=CC=C6)CO1'
]

pd_ch_true_ranking = ordered_smiles = [
    "CCC(C)N(C(C)CC)C(=O)c1cccc(O)n1",
    "CCN(CC)C(=O)c1cccc(O)n1",
    "CC(C)N(C(C)C)C(=O)c1cccc(O)n1",
    "CN(C)C(=O)c1cccc(O)n1",
    "CC(=O)c1cccc(O)n1",
    "CCN(CC)C(=O)CCNC(C)=O",
    "CCN(CC)C(=O)C(C)(C)c1cccc(O)n1",
    "Oc1cccc(-c2ccc3ccccc3n2)n1",
    "Oc1c(C(F)(F)F)cc([N+](=O)[O-])cn1",
    "Cc1ccc(nc1)C(C)(C2CCCCC2)c3cccc(O)n3",
    "Oc1ccccn1",
    "O=C(O)c1cccc(O)n1",
    "CC(=O)NCC(=O)NS(=O)(=O)c1ccc(C)cc1",
    "CC(=O)N(C)C",
    "CC(=O)NCCC(=O)O",
    "CC(=O)NC(CSc1ccccc1)C(C)C",
    "CC(=O)NCC(=O)O",
    "no ligand"
]

pd_dual_true_ranking = [
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + Brettphos COC1=C(P(C2CCCCC2)C2CCCCC2)C(=C(OC)C=C1)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + 2-(Di-tert-butylphosphanyl)-1-(2-methoxyphenyl)-1H-pyrrole COC1=CC=CC=C1N1C=CC=C1P(C(C)(C)C)C(C)(C)C",
    "Tris(3-chlorophenyl)phosphine ClC1=CC=CC(=C1)P(C1=CC(Cl)=CC=C1)C1=CC(Cl)=CC=C1 + BippyPhos CC(C)(C)P(C1=CC=NN1C1=C(N(N=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C(C)(C)C",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + Tri-p-tolyl Phosphite CC1=CC=C(OP(OC2=CC=C(C)C=C2)OC2=CC=C(C)C=C2)C=C1",
    "2-(Di-tert-butylphosphanyl)-1-(2-methoxyphenyl)-1H-pyrrole COC1=CC=CC=C1N1C=CC=C1P(C(C)(C)C)C(C)(C)C + (S)-2,2'-Bis(diphenylphosphino)-4,4',6,6'-tetramethoxybiphenyl COc1cc(OC)c(-c2c(OC)cc(OC)cc2P(c2ccccc2)c2ccccc2)c(P(c2ccccc2)c2ccccc2)c1",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + Tri-p-tolyl Phosphite CC1=CC=C(OP(OC2=CC=C(C)C=C2)OC2=CC=C(C)C=C2)C=C1",
    "Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1 + Taniaphos SL-T001-2 CN(C)[C@@H](c1ccccc1P(c1ccccc1)c1ccccc1)c1c[cH-]cc1P(c1ccccc1)c1ccccc1.[Fe+2].c1cc[cH-]c1",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + (S,S)-Me-Duphos, C[C@H]1CC[C@H](C)P1C1=CC=CC=C1P1[C@@H](C)CC[C@@H]1C",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + Tripropan-2-yl phosphite CC(C)OP(OC(C)C)OC(C)C",
    "Phosphine, bis(1,1-dimethylethyl)- CC(C)(C)PC(C)(C)C + (S)-SDP C=1C=CC(=CC1)P(C=2C=CC=CC2)C3=CC=CC4=C3C5(C=6C(=CC=CC6CC5)P(C=7C=CC=CC7)C=8C=CC=CC8)CC4",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + Tri-p-tolyl Phosphite CC1=CC=C(OP(OC2=CC=C(C)C=C2)OC2=CC=C(C)C=C2)C=C1",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + (S)-2,2'-Bis(diphenylphosphino)-4,4',6,6'-tetramethoxybiphenyl COc1cc(OC)c(-c2c(OC)cc(OC)cc2P(c2ccccc2)c2ccccc2)c(P(c2ccccc2)c2ccccc2)c1",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + (S)-SDP C=1C=CC(=CC1)P(C=2C=CC=CC2)C3=CC=CC4=C3C5(C=6C(=CC=CC6CC5)P(C=7C=CC=CC7)C=8C=CC=CC8)CC4",
    "Tributyl phosphite CCCCOP(OCCCC)OCCCC + (R)-Binam-P N(P(C1=CC=CC=C1)C1=CC=CC=C1)C1=C(C2=CC=CC=C2C=C1)C1=C(NP(C2=CC=CC=C2)C2=CC=CC=C2)C=CC2=CC=CC=C12",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + Dicyclohexylethylphosphine tetrafluoroborate CCP(C1CCCCC1)C1CCCCC1",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "(4-(N,N-Dimethylamino)phenyl)di-tert-butyl phosphine CN(C)C1=CC=C(C=C1)P(C(C)(C)C)C(C)(C)C + Taniaphos SL-T001-2 CN(C)[C@@H](c1ccccc1P(c1ccccc1)c1ccccc1)c1c[cH-]cc1P(c1ccccc1)c1ccccc1.[Fe+2].c1cc[cH-]c1",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + (R)-(S)-BPPFA CC(C)P([c-]1cccc1)C(C)C.CC(C)P(c1ccc[c-]1[C@H](C)N(C)CCN1CCCCC1)C(C)C.[Fe+2]",
    "Tri-p-tolyl Phosphite CC1=CC=C(OP(OC2=CC=C(C)C=C2)OC2=CC=C(C)C=C2)C=C1 + (S,S)-Me-Duphos, C[C@H]1CC[C@H](C)P1C1=CC=CC=C1P1[C@@H](C)CC[C@@H]1C",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F",
    "Di-tert-butyl(2,2-dimethylpropyl)phosphanium tetrafluoroborate F[B-](F)(F)F.CC(C)(C)C[PH+](C(C)(C)C)C(C)(C)C + Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + 2-[2-(Dicyclohexylphosphanyl)phenyl]-1-methyl-1H-indole CN1C(=CC2=CC=CC=C12)C1=CC=CC=C1P(C1CCCCC1)C1CCCCC1",
    "Tributyl phosphite CCCCOP(OCCCC)OCCCC + (S)-SDP C=1C=CC(=CC1)P(C=2C=CC=CC2)C3=CC=CC4=C3C5(C=6C(=CC=CC6CC5)P(C=7C=CC=CC7)C=8C=CC=CC8)CC4",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + (4-(N,N-Dimethylamino)phenyl)di-tert-butyl phosphine CN(C)C1=CC=C(C=C1)P(C(C)(C)C)C(C)(C)C",
    "BippyPhos CC(C)(C)P(C1=CC=NN1C1=C(N(N=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1)C(C)(C)C + Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F",
    "Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1 + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "(Ethane-1,2-diyl)bis[bis(pentafluorophenyl)phosphane] FC1=C(F)C(F)=C(P(CCP(C2=C(F)C(F)=C(F)C(F)=C2F)C2=C(F)C(F)=C(F)C(F)=C2F)C2=C(F)C(F)=C(F)C(F)=C2F)C(F)=C1F + [CF3]8-DPPF C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.[Fe]",
    "Triallyl phosphine C=CCP(CC=C)CC=C + (S)-SDP C=1C=CC(=CC1)P(C=2C=CC=CC2)C3=CC=CC4=C3C5(C=6C(=CC=CC6CC5)P(C=7C=CC=CC7)C=8C=CC=CC8)CC4",
    "Di-tert-butyl(butyl)phosphonium tetrafluoroborate CCCCP(C(C)(C)C)C(C)(C)C + 5-(Dicyclohexylphosphanyl)-1',3',5'-triphenyl-1'H-1,4'-bipyrazole C1CCC(CC1)P(C1CCCCC1)C1=CC=NN1C1=C(N(N=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1",
    "Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1 + Tributyl phosphite CCCCOP(OCCCC)OCCCC",
    "2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2 + Di-tert-butyl(butyl)phosphonium tetrafluoroborate CCCCP(C(C)(C)C)C(C)(C)C",
    "Phosphorodiamidous acid, tetrakis(1-methylethyl)-, 2-cyanoethyl ester CC(C)N(C(C)C)P(OCCC#N)N(C(C)C)C(C)C + Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C",
    "2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2 + Tri(2-furyl)phosphine O1C=CC=C1P(C1=CC=CO1)C1=CC=CO1",
    "Di-tert-butyl(2'-methyl[1,1'-biphenyl]-2-yl)phosphane CC1=CC=CC=C1C1=CC=CC=C1P(C(C)(C)C)C(C)(C)C + Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1",
    "1,3,5-Triaza-7-phosphaadamantane C1N2CN3CN1CP(C2)C3 + Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C",
    "1,3,5-Triaza-7-phosphaadamantane C1N2CN3CN1CP(C2)C3 + 2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2",
    "Di-tert-butyl(butyl)phosphonium tetrafluoroborate CCCCP(C(C)(C)C)C(C)(C)C + Taniaphos SL-T001-2 CN(C)[C@@H](c1ccccc1P(c1ccccc1)c1ccccc1)c1c[cH-]cc1P(c1ccccc1)c1ccccc1.[Fe+2].c1cc[cH-]c1",
    "Tris(2-chloroethyl) phosphite ClCCOP(OCCCl)OCCCl + (S)-2,2'-Bis(diphenylphosphino)-4,4',6,6'-tetramethoxybiphenyl COc1cc(OC)c(-c2c(OC)cc(OC)cc2P(c2ccccc2)c2ccccc2)c(P(c2ccccc2)c2ccccc2)c1",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + Di-tert-butyl(butyl)phosphonium tetrafluoroborate CCCCP(C(C)(C)C)C(C)(C)C",
    "Tris(2-chloroethyl) phosphite ClCCOP(OCCCl)OCCCl + Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C",
    "Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F + Mandyphos SL-M001-1 CN(C)[C@H](c1ccccc1)[c-]1cccc1P(c1ccccc1)c1ccccc1.CN(C)[C@H](c1ccccc1)c1cc[cH-]c1P(c1ccccc1)c1ccccc1.[Fe+2]",
    "Brettphos COC1=C(P(C2CCCCC2)C2CCCCC2)C(=C(OC)C=C1)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C + (4-(N,N-Dimethylamino)phenyl)di-tert-butyl phosphine CN(C)C1=CC=C(C=C1)P(C(C)(C)C)C(C)(C)C",
    "(4-(N,N-Dimethylamino)phenyl)di-tert-butyl phosphine CN(C)C1=CC=C(C=C1)P(C(C)(C)C)C(C)(C)C + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1 + (R)-Binam-P N(P(C1=CC=CC=C1)C1=CC=CC=C1)C1=C(C2=CC=CC=C2C=C1)C1=C(NP(C2=CC=CC=C2)C2=CC=CC=C2)C=CC2=CC=CC=C12",
    "2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2 + Mandyphos SL-M001-1 CN(C)[C@H](c1ccccc1)[c-]1cccc1P(c1ccccc1)c1ccccc1.CN(C)[C@H](c1ccccc1)c1cc[cH-]c1P(c1ccccc1)c1ccccc1.[Fe+2]",
    "Tributyl phosphite CCCCOP(OCCCC)OCCCC + [CF3]8-DPPF C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.[Fe]",
    "Tris(3-chlorophenyl)phosphine ClC1=CC=CC(=C1)P(C1=CC(Cl)=CC=C1)C1=CC(Cl)=CC=C1 + (S,S)-Chiraphos C[C@@H]([C@H](C)P(C1=CC=CC=C1)C1=CC=CC=C1)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Di-tert-butyl(2,2-dimethylpropyl)phosphanium tetrafluoroborate F[B-](F)(F)F.CC(C)(C)C[PH+](C(C)(C)C)C(C)(C)C + (S)-2,2'-Bis(diphenylphosphino)-4,4',6,6'-tetramethoxybiphenyl COc1cc(OC)c(-c2c(OC)cc(OC)cc2P(c2ccccc2)c2ccccc2)c(P(c2ccccc2)c2ccccc2)c1",
    "Di-tert-butyl(2'-methyl[1,1'-biphenyl]-2-yl)phosphane CC1=CC=CC=C1C1=CC=CC=C1P(C(C)(C)C)C(C)(C)C + Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C",
    "(R)-Binam-P N(P(C1=CC=CC=C1)C1=CC=CC=C1)C1=C(C2=CC=CC=C2C=C1)C1=C(NP(C2=CC=CC=C2)C2=CC=CC=C2)C=CC2=CC=CC=C12 + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Triallyl phosphine C=CCP(CC=C)CC=C + (2R,2'R,3R,3'R)-Ph-BIBOP CC(C)(C)P1c2c(cccc2-c2ccccc2)O[C@@H]1[C@H]1Oc2cccc(-c3ccccc3)c2P1C(C)(C)C",
    "Tris(3-chlorophenyl)phosphine ClC1=CC=CC(=C1)P(C1=CC(Cl)=CC=C1)C1=CC(Cl)=CC=C1 + Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F",
    "Brettphos COC1=C(P(C2CCCCC2)C2CCCCC2)C(=C(OC)C=C1)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Dibenzyl N,N-dimethylphosphoramidite CN(C)P(OCC1=CC=CC=C1)OCC1=CC=CC=C1 + [CF3]8-DPPF C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.[Fe]",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + (2R,2'R,3R,3'R)-Ph-BIBOP CC(C)(C)P1c2c(cccc2-c2ccccc2)O[C@@H]1[C@H]1Oc2cccc(-c3ccccc3)c2P1C(C)(C)C",
    "2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2 + Triallyl phosphine C=CCP(CC=C)CC=C",
    "Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F + (S,S)-Chiraphos C[C@@H]([C@H](C)P(C1=CC=CC=C1)C1=CC=CC=C1)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Di-tert-butyl(2,2-dimethylpropyl)phosphanium tetrafluoroborate F[B-](F)(F)F.CC(C)(C)C[PH+](C(C)(C)C)C(C)(C)C + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + Triallyl phosphine C=CCP(CC=C)CC=C",
    "Phosphine, bis(1,1-dimethylethyl)- CC(C)(C)PC(C)(C)C + Triallyl phosphine C=CCP(CC=C)CC=C",
    "Phosphine, bis(1,1-dimethylethyl)- CC(C)(C)PC(C)(C)C + Tributyl phosphite CCCCOP(OCCCC)OCCCC",
    "5-(Dicyclohexylphosphanyl)-1',3',5'-triphenyl-1'H-1,4'-bipyrazole C1CCC(CC1)P(C1CCCCC1)C1=CC=NN1C1=C(N(N=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1 + (S,S)-Chiraphos C[C@@H]([C@H](C)P(C1=CC=CC=C1)C1=CC=CC=C1)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + (S)-DM-BINAP Cc1cc(C)cc(P(c2cc(C)cc(C)c2)c2ccc3ccccc3c2-c2c(P(c3cc(C)cc(C)c3)c3cc(C)cc(C)c3)ccc3ccccc23)c1",
    "Di-tert-butyl(2,2-dimethylpropyl)phosphanium tetrafluoroborate F[B-](F)(F)F.CC(C)(C)C[PH+](C(C)(C)C)C(C)(C)C + Mandyphos SL-M001-1 CN(C)[C@H](c1ccccc1)[c-]1cccc1P(c1ccccc1)c1ccccc1.CN(C)[C@H](c1ccccc1)c1cc[cH-]c1P(c1ccccc1)c1ccccc1.[Fe+2]",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + Tributyl phosphite CCCCOP(OCCCC)OCCCC",
    "5-(Dicyclohexylphosphanyl)-1',3',5'-triphenyl-1'H-1,4'-bipyrazole C1CCC(CC1)P(C1CCCCC1)C1=CC=NN1C1=C(N(N=C1C1=CC=CC=C1)C1=CC=CC=C1)C1=CC=CC=C1 + Tris[4-(trifluoromethyl)phenyl]phosphane FC(F)(F)C1=CC=C(C=C1)P(C1=CC=C(C=C1)C(F)(F)F)C1=CC=C(C=C1)C(F)(F)F",
    "Phosphine, bis(1,1-dimethylethyl)- CC(C)(C)PC(C)(C)C + Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C",
    "Triallyl phosphine C=CCP(CC=C)CC=C + Tributyl phosphite CCCCOP(OCCCC)OCCCC",
    "Phosphine, bis(1,1-dimethylethyl)- CC(C)(C)PC(C)(C)C + (2R,2'R,3R,3'R)-Ph-BIBOP CC(C)(C)P1c2c(cccc2-c2ccccc2)O[C@@H]1[C@H]1Oc2cccc(-c3ccccc3)c2P1C(C)(C)C",
    "2,6,7-Trioxa-1-phosphabicyclo[2.2.2]octane, 4-ethyl- CCC12COP(OC1)OC2 + (S,S)-Me-Duphos, C[C@H]1CC[C@H](C)P1C1=CC=CC=C1P1[C@@H](C)CC[C@@H]1C",
    "Allyldiphenylphosphine C=CCP(C1=CC=CC=C1)C1=CC=CC=C1 + Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C",
    "Brettphos COC1=C(P(C2CCCCC2)C2CCCCC2)C(=C(OC)C=C1)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C + 2-(Di-tert-butylphosphanyl)-1-(2-methoxyphenyl)-1H-pyrrole COC1=CC=CC=C1N1C=CC=C1P(C(C)(C)C)C(C)(C)C",
    "Tributyl phosphite CCCCOP(OCCCC)OCCCC + (2R,2'R,3R,3'R)-Ph-BIBOP CC(C)(C)P1c2c(cccc2-c2ccccc2)O[C@@H]1[C@H]1Oc2cccc(-c3ccccc3)c2P1C(C)(C)C",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "Trineopentyl phosphite CC(C)(C)COP(OCC(C)(C)C)OCC(C)(C)C + (S,S)-Me-Duphos, C[C@H]1CC[C@H](C)P1C1=CC=CC=C1P1[C@@H](C)CC[C@@H]1C",
    "[CF3]8-DPPF C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.C1=C(C=C(C=C1C(F)(F)F)P([C-]2[C-]=[C-][C-]=[C-]2)C3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)C(F)(F)F.[Fe] + (S)-BINAPINE CC(C)(C)P1CC2=C(C3=CC=CC=C3C=C2)C2=C(C=CC3=CC=CC=C23)[C@H]1[C@H]1P(CC2=C(C3=CC=CC=C3C=C2)C2=C1C=CC1=CC=CC=C21)C(C)(C)C",
    "Brettphos COC1=C(P(C2CCCCC2)C2CCCCC2)C(=C(OC)C=C1)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C + Josiphos SL-J002-1 [Fe++].[CH-]1C=CC=C1.C[C@H]([C-]1C=CC=C1P(C1=CC=CC=C1)C1=CC=CC=C1)P(C(C)(C)C)C(C)(C)C",
    "Tris(trimethylsilyl)phosphine C[Si](C)(C)P([Si](C)(C)C)[Si](C)(C)C + Triallyl phosphine C=CCP(CC=C)CC=C",
    "rac-BI-DIME COc1cccc(OC)c1-c1cccc2c1P(C(C)(C)C)CO2 + Di-tert-butyl(2',4',6'-triisopropyl-4-methoxy-3,5,6-trimethyl-[1,1'-biphenyl]-2-yl)phosphine COC1=C(C)C(P(C(C)(C)C)C(C)(C)C)=C(C(C)=C1C)C1=C(C=C(C=C1C(C)C)C(C)C)C(C)C",
    "(2R,2'R,3R,3'R)-Ph-BIBOP CC(C)(C)P1c2c(cccc2-c2ccccc2)O[C@@H]1[C@H]1Oc2cccc(-c3ccccc3)c2P1C(C)(C)C + (S)-SDP C=1C=CC(=CC1)P(C=2C=CC=CC2)C3=CC=CC4=C3C5(C=6C(=CC=CC6CC5)P(C=7C=CC=CC7)C=8C=CC=CC8)CC4",
    "(S,S)-Me-Duphos, C[C@H]1CC[C@H](C)P1C1=CC=CC=C1P1[C@@H](C)CC[C@@H]1C + Xantphos CC1(C)C2=C(OC3=C1C=CC=C3P(C1=CC=CC=C1)C1=CC=CC=C1)C(=CC=C2)P(C1=CC=CC=C1)C1=CC=CC=C1",
    "(S,S)-Chiraphos C[C@@H]([C@H](C)P(C1=CC=CC=C1)C1=CC=CC=C1)P(C1=CC=CC=C1)C1=CC=CC=C1 + (S)-(6,6'-Dimethoxy-[1,1'-biphenyl]-2,2'-diyl)bis(bis(3,5-bis(trimethylsilyl)phenyl)phosphine) COc1cccc(P(c2cc([Si](C)(C)C)cc([Si](C)(C)C)c2)c2cc([Si](C)(C)C)cc([Si](C)(C)C)c2)c1-c1c(OC)cccc1P(c1cc([Si](C)(C)C)cc([Si](C)(C)C)c1)c1cc([Si](C)(C)C)cc([Si](C)(C)C)c1",
    "(S)-BINAPINE CC(C)(C)P1CC2=C(C3=CC=CC=C3C=C2)C2=C(C=CC3=CC=CC=C23)[C@H]1[C@H]1P(CC2=C(C3=CC=CC=C3C=C2)C2=C1C=CC1=CC=CC=C21)C(C)(C)C + Taniaphos SL-T001-2 CN(C)[C@@H](c1ccccc1P(c1ccccc1)c1ccccc1)c1c[cH-]cc1P(c1ccccc1)c1ccccc1.[Fe+2].c1cc[cH-]c1",
    "Tris(3-chlorophenyl)phosphine ClC1=CC=CC(=C1)P(C1=CC(Cl)=CC=C1)C1=CC(Cl)=CC=C1 + (S)-(6,6'-Dimethoxy-[1,1'-biphenyl]-2,2'-diyl)bis(bis(3,5-bis(trimethylsilyl)phenyl)phosphine) COc1cccc(P(c2cc([Si](C)(C)C)cc([Si](C)(C)C)c2)c2cc([Si](C)(C)C)cc([Si](C)(C)C)c2)c1-c1c(OC)cccc1P(c1cc([Si](C)(C)C)cc([Si](C)(C)C)c1)c1cc([Si](C)(C)C)cc([Si](C)(C)C)c1"
]

def generate_id_map(smiles_list):
    id_map = {}
    for lig in smiles_list:
        id_map[f"Catalyst_{hashlib.md5(lig.encode()).hexdigest()[:6]}"] = lig
    return id_map

def get_shuffled_catalyst_list(id_map):
    items = list(id_map.items())
    random.seed(42)
    random.shuffle(items)
    return "\n".join([f"- {k}: {v}" for k, v in items])

# 3. INITIALIZE MAPS
pd_id_map = generate_id_map(pd_true_ranking)
ni_id_map = generate_id_map(ni_true_ranking)
pd_ch_id_map = generate_id_map(pd_ch_true_ranking)
pd_dual_id_map = generate_id_map(pd_dual_true_ranking)

# 4. PASTE THE RANKING_TASKS DICTIONARY YOU JUST ASKED ABOUT HERE:
RANKING_TASKS = {
    "Pd_Fluorination": {
        "reaction_definition": "A palladium-catalyzed, exogenous-fluoride-free cross-coupling that transforms electron-deficient (hetero)aryl sulfonyl fluorides (ArSO2F) into (hetero)aryl fluorides.",
        "catalyst_list": get_shuffled_catalyst_list(pd_id_map),  
        "id_map": pd_id_map, 
        "true_ranking": pd_true_ranking, 
        "num_catalysts": len(pd_id_map),
        "mechanism_text": "The catalytic cycle initiates with the oxidative addition of the C-S bond of the sulfonyl fluoride to an electron-rich Pd(0) species. This is followed by a desulfonylation step (extrusion of SO2 gas) to generate a critical monomeric [Ar-Pd(II)-F] intermediate. Finally, a challenging C(sp2)-F reductive elimination occurs. Success heavily relies on bulky, specialized phosphine ligands that can both stabilize the Pd(II) intermediate against off-cycle dimerization and sterically force the demanding C-F reductive elimination.",
        "related_paper_paths": ["publications/Angew Chem Int Ed - 2021 - Chatelain - Desulfonative Suzuki Miyaura Coupling of Sulfonyl Fluorides.pdf", "publications/pd-catalyzed-nucleophilic-fluorination-of-aryl-bromides.pdf"],
        "actual_paper_paths": ["publications/Pd-catalyzed desulfonylative fluorination of electron-deficient (hetero)aryl sulfonyl fluorides.pdf"]
    },
    "Ni_Epoxide_Coupling": {
        "reaction_definition": "A stereoconvergent, dual nickel/photoredox-catalyzed reductive cross-electrophile coupling that merges racemic styrene oxides with aryl iodides.",
        "catalyst_list": get_shuffled_catalyst_list(ni_id_map), 
        "id_map": ni_id_map, 
        "true_ranking": ni_true_ranking, 
        "num_catalysts": len(ni_id_map),
        "mechanism_text": "The transformation relies on dual catalysis. The photoredox cycle generates an aryl radical from the aryl iodide, which is intercepted by the chiral Ni(0) catalyst to form an [Ar-Ni(I)] intermediate. Concurrently, epoxide ring-opening occurs to yield a stabilized benzylic radical. Recombination of this benzylic radical with the nickel complex yields a high-valent, chiral [Ar-Ni(III)-alkyl] species. The final, enantiodetermining step is a rapid reductive elimination that sets the stereocenter and releases the chiral alcohol product, highlighting the extreme sensitivity of this step to the electronic and steric tuning of the bioxazoline/biimidazoline ligands.",
        "related_paper_paths": ["publications/nickel-catalyzed-enantioselective-reductive-cross-coupling-of-styrenyl-aziridines.pdf", "publications/regioselective-cross-electrophile-coupling-of-epoxides-and-(hetero)aryl-iodides-via-ni-ti-photoredox-catalysis.pdf"],
        "actual_paper_paths": ["publications/ni-photoredox-catalyzed-enantioselective-cross-electrophile-coupling-of-styrene-oxides-with-aryl-iodides.pdf"]
    },
    "Pd_CH_Fluorination": {
        "reaction_definition": "A ligand-accelerated, palladium-catalyzed transformation that enables the stereo- and site-selective methylene beta-C(sp3)-H fluorination of weakly coordinating native amides using Selectfluor.",
        "catalyst_list": get_shuffled_catalyst_list(pd_ch_id_map), 
        "id_map": pd_ch_id_map, 
        "true_ranking": pd_ch_true_ranking, 
        "num_catalysts": len(pd_ch_id_map),
        "mechanism_text": "The catalytic cycle initiates with the coordination of the native amide to a cationic Pd(II) center assembled with a bidentate ligand. An irreversible, rate-determining concerted metalation-deprotonation process activates the beta-methylene C-H bond to form a five-membered palladacycle intermediate. Oxidative addition by Selectfluor then accesses a high-valent Pd(IV)-fluoride intermediate, which undergoes direct C-F reductive elimination to yield the stereoselective product and regenerate the active catalyst complex.",
        "related_paper_paths": ["publications/ligand-enabled-palladium-catalyzed-beta-c(sp3)-h-arylation-of-weinreb-amides.pdf", "publications/Pd-cat-direct-betaC(sp3)H-fluorination-aliphatic-carboxylic-acids.pdf"],
        "actual_paper_paths": ["publications/palladium-catalyzed-methylene-beta-c-h-fluorination-of-native-amides.pdf"]
    },
    "Pd_Dual_Catalysis": {
        "reaction_definition": "Dual-Ligand System for Mild Decarbonylative Suzuki–Miyaura Cross-Coupling of Aroyl Chlorides",
        "id_map": pd_dual_id_map, 
        "true_ranking": pd_dual_true_ranking, 
        "mechanism_text":"The reaction mechanism for the dual-ligand decarbonylative Suzuki-Miyaura cross-coupling of aroyl chlorides is a dynamic ligand-relay process where two distinct ligands are involved and the assignation to L1 and L2 cannot be known a priori. It starts with the oxidative addition of the aroyl chloride to a Pd(0) catalyst to form a Pd(II)-acyl complex. This step is followed by a quantitative decarbonylation (CO extrusion), which is exclusively facilitated by Ligand 2 to form a (L2)2Pd(aryl)(Cl) intermediate. Meanwhile, the complex supported by Ligand 1 remains stable and does not decarbonylate, but rapidly exchanges with free L2 to enter the productive pathway. Once Complex is formed, a rapid ligand exchange occurs where L1 replaces L2. L1 then preferentially accelerates the subsequent transmetalation with an arylboronic acid and reductive elimination steps to form the final biaryl product. The cooperativity relies on this dynamic exchange, using L2 for decarbonylation and L1 for transmetalation/reductive elimination, overcoming the kinetic barriers of using a single ligand.",
        "related_paper_paths": ["publications/decarbonylative-suzuki-miyaura-cross-coupling-of-aroyl-chlorides.pdf", "publications/pd-catalyzed-decarbonylative-cross-couplings-of-aroyl-chlorides.pdf"],
        "actual_paper_paths": ["publications/dual-ligand-system-for-mild-decarbonylative-suzuki-miyaura-cross-coupling-of-aroyl-chlorides.pdf"]
    }
}

MASTER_DATASET_RAW = r"""
6%  CC(C)(C1=NC(=CC=C1)O)C(=O)NC2=C(C(=C(C(=C2F)F)C(F)(F)F)F)F
1%  C1=CC(=NC(=C1)O)C(=O)NC2=C(C(=C(C(=C2F)F)C(F)(F)F)F)F
1%  CC(C)(C1=NC2=CC=CC=C2C=C1)C3=NC(=CC=C3)O
4%  C1=CC=C2C(=C1)C=CC(=N2)C3=NC(=CC=C3)O
1%  C1CCN(CC1)CC2=NC(=CC=C2)O
0%  CC(C)(CN1CCCCC1)C2=NC(=CC=C2)O
1%  C1=CC(=NC(=C1)O)C(=O)O
0%  CC(C)(C1=NC(=CC=C1)O)C(=O)O
4%  C/C(=N\OC)/C1=NC(=CC=C1)O
1%  CC(C)(C)[C@@H]1COC(=N1)C2=NC(=CC=C2)O
2%  CC(C)[C@@H]1COC(=N1)C(C)(C)C2=NC(=CC=C2)O
1%  CC(C)C(C(=O)O)NC(=O)C
1%  CC(=O)N[C@@H](CC1=CC=CC=C1)CSCC2=CC=CC=C2
0%  CC(=O)N[C@@H](CC1=CC=CC=C1)CN(C)C
0%  CC(=O)N[C@H](C1=N[C@@H](CC2=CC=CC=C2)CO1)C(C)(C)C
0%  CC(=O)N[C@@H](CC1=NC2=CC=CC=C2C=C1)C3=CC=CC=C3
0%  C1=C(C(=NC=C1C(F)(F)F)O)C(F)(F)F
0%  CC1CCC2=C(C)C3=C(C=CC=C3)N=C2O1
37% C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3)O)N=N2
40% C1=CC(=NC(=C1)O)C2=CN(CC3=CC=C(C=C3)C(F)(F)F)N=N2
10% C1=CC(=NC(=C1)O)C2=CN(CC3=CC=C(C=C3)C#N)N=N2
40% C1=CC(=NC(=C1)O)C2=CN(CC3=CC=C(C=C3)[N+](=O)[O-])N=N2
17% C1=CC(=C(CN2C=C(C3=NC(=CC=C3)O)N=N2)C(=C1)F)F
14% CC1=CC=CC(=C1CN2C=C(C3=NC(=CC=C3)O)N=N2)C
36% CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)CN2C=C(C3=NC(=CC=C3)O)N=N2
21% COC1=CC(=CC(=C1)OC)CN2C=C(C3=NC(=CC=C3)O)N=N2
37% C1=CC(=NC(=C1)O)C2=CN(CC3=CC(=CC(=C3)C(F)(F)F)C(F)(F)F)N=N2
35% C1=CC(=NC(=C1)O)C2=CN(CC3=C(C=C(C=C3)C(F)(F)F)C(F)(F)F)N=N2
27% C1=CC(=NC(=C1)O)C2=CN(CC3=C(C(=C(C(=C3F)F)F)F)F)N=N2
28% C1=CC2=C(C=C1)C(=C3C=CC=CC3=C2)CN4C=C(C5=NC(=CC=C5)O)N=N4
31% C1=CC2=C(C=C1)C(C3=C2C=CC=C3)N4C=C(C5=NC(=CC=C5)O)N=N4
32% C1=CC=C(C=C1)C(C2=CC=CC=C2)N3C=C(C4=NC(=CC=C4)O)N=N3
34% CC(C1=CC=CC=C1)N2C=C(C3=NC(=CC=C3)O)N=N2
37% C1CCC(CC1)CN2C=C(C3=NC(=CC=C3)O)N=N2
35% C1=CC(=NC(=C1)O)C2=CN(CC3C4CC5CC(C4)CC3C5)N=N2
0%  CC1=CC=C(C=C1)S(=O)(=O)N2C=C(C3=NC(=CC=C3)O)N=N2
31% CCCCCCCCCCCCCN1C=C(C2=NC(=CC=C2)O)N=N1
54% C1=CC=C(C=C1)CN2C=C(C3=NC(=C(C=C3)C(F)(F)F)O)N=N2
52% C1=CC=C(C=C1)CN2C=C(C3=NC(=C(C=C3)Cl)O)N=N2
56% COC1=C(N=C(C=C1)C2=CN(CC3=CC=CC=C3)N=N2)O
6%  CC1=CN(C=N1)C2=C(N=C(C=C2)C3=CN(CC4=CC=CC=C4)N=N3)O
80% C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3C(F)(F)F)O)N=N2
46% C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3[N+](=O)[O-])O)N=N2
36% C1=CC=C(C=C1)CN2C=C(C3=NC(=C(C=C3Cl)Cl)O)N=N2
40% C1=CC=C(C=C1)CN2C=C(C3=NC(=C(C=C3F)F)O)N=N2
77% CC1=CC(=C(N=C1C2=CN(CC3=CC=CC=C3)N=N2)O)OC
87% COC1=C(N=C(C(=C1)C(F)(F)F)C2=CN(CC3=CC=C(C=C3)C(F)(F)F)N=N2)O
78% COC1=C(N=C(C(=C1)C2=CC=CC=C2)C3=CN(CC4=CC=C(C=C4)C(F)(F)F)N=N3)O
56% CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)C2=CC(=C(N=C2C3=CN(CC4=CC=C(C=C4)C(F)(F)F)N=N3)O)OC
89% COC1=C(N=C(C(=C1)C2=CC(=CC(=C2)C(F)(F)F)C(F)(F)F)C3=CN(CC4=CC=C(C=C4)C(F)(F)F)N=N3)O
69% COC1=CC(=CC(=C1)OC)C2=CC(=C(N=C2C3=CN(CC4=CC=C(C=C4)C(F)(F)F)N=N3)O)OC
37% C1=CC=C(C=C1)CN2C(=C(C3=NC(=CC=C3)O)N=N2)C4=CC=CC=C4
38% CC(C)(C)C1=CC(=CC(=C1)C(C)(C)C)C2=C(C3=NC(=CC=C3)O)N=NN2CC4=CC=CC=C4
37% COC1=CC(=CC(=C1)OC)C2=C(C3=NC(=CC=C3)O)N=NN2CC4=CC=CC=C4
28% C1=CC=C(C=C1)CN2C(=C(C3=NC(=CC=C3)O)N=N2)C4=CC(=CC(=C4)C(F)(F)F)C(F)(F)F
41% CC1=C(C2=NC(=CC=C2)O)N=NN1CC3=CC=CC=C3
33% C1=CC=C(C=C1)CN2C(=C(C3=NC(=CC=C3)O)N=N2)C(F)(F)F
52% CC1=C(C2=NC(=C(C=C2)OC)O)N=NN1CC3=CC=C(C=C3)C(F)(F)F
54% CC(C)C1=C(C2=NC(=C(C=C2)OC)O)N=NN1CC3=CC=C(C=C3)C(F)(F)F
48% COC1=C(N=C(C(=C1)C2=C(C3CCCCC3)N(CC4=CC=C(C=C4)C(F)(F)F)N=N2)O)
21% CC(C)(C1=NC(=CC=C1)O)C2=CN(CC3=CC=CC=C3)N=N2
78% Cc1cc(C(F)(F)F)c(-c2cn(Cc3ccccc3)nn2)nc1O     #added manually to tune the surrogate's extrapolation on a novel core with a expected active substituent pattern
74% COc1cc(Cl)c(-c2cn(Cc3ccccc3)nn2)nc1O       #added manually to tune the surrogate's extrapolation on a novel core with a expected active substituent pattern
"""

INITIAL_SEED = r"""
37% C1=CC=C(C=C1)CN2C=C(C3=NC(=CC=C3)O)N=N2
6%  CC(C)(C1=NC(=CC=C1)O)C(=O)NC2=C(C(=C(C(=C2F)F)C(F)(F)F)F)F
1%  C1=CC(=NC(=C1)O)C(=O)NC2=C(C(=C(C(=C2F)F)C(F)(F)F)F)F
1%  CC(C)(C1=NC2=CC=CC=C2C=C1)C3=NC(=CC=C3)O
4%  C1=CC=C2C(=C1)C=CC(=N2)C3=NC(=CC=C3)O
1%  C1CCN(CC1)CC2=NC(=CC=C2)O
0%  CC(C)(CN1CCCCC1)C2=NC(=CC=C2)O
1%  C1=CC(=NC(=C1)O)C(=O)O
0%  CC(C)(C1=NC(=CC=C1)O)C(=O)O
4%  C/C(=N\OC)/C1=NC(=CC=C1)O
1%  CC(C)(C)[C@@H]1COC(=N1)C2=NC(=CC=C2)O
2%  CC(C)[C@@H]1COC(=N1)C(C)(C)C2=NC(=CC=C2)O
1%  CC(C)C(C(=O)O)NC(=O)C
1%  CC(=O)N[C@@H](CC1=CC=CC=C1)CSCC2=CC=CC=C2
0%  CC(=O)N[C@@H](CC1=CC=CC=C1)CN(C)C
0%  CC(=O)N[C@H](C1=N[C@@H](CC2=CC=CC=C2)CO1)C(C)(C)C
0%  CC(=O)N[C@@H](CC1=NC2=CC=CC=C2C=C1)C3=CC=CC=C3
0%  C1=C(C(=NC=C1C(F)(F)F)O)C(F)(F)F
0%  CC1CCC2=C(C)C3=C(C=CC=C3)N=C2O1 
"""