#!/usr/bin/env python3

import random
import os
import sys
import vegas
import math
import torch
import logging
import numpy as np
from inspect import getsourcefile
import numpy as np
from torchps.flat_phase_space_generator import *
from utils.integrands.abstract import Integrand
from utils.integrands import sanitize_variable

def sanitize_variable(a,b):
    return a
logger = logging.getLogger(__name__)

class CrossSection(Integrand):
    """Stuff"""
    def __init__(self, e_cm=1000, pdf=False, delr_cut=0.4,pt_cut=10, rap_maxcut=2.4, pdf_type=None, pdf_dir=None, lhapdf_dir=None, process=None, device=None, *args, **kwargs):  
        #Parameterinfo
        logger.debug(process)
        self.pdf=pdf
        if self.pdf and lhapdf_dir==None:
            logger.debug("The directory of the LHAPDF module was not given. PDFs are not included")
            self.pdf=False
        if self.pdf:
            if lhapdf_dir not in sys.path:
                sys.path.append(lhapdf_dir)
            try:
                import lhapdf
            except Exception as e:
                logger.debug("LHAPDF could not be imported")
                
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if BASE_DIR+"/integrands/mg/"+process not in sys.path:
            sys.path.append(BASE_DIR+"/integrands/mg/"+process)
        try:
            import matrix2py
        except:
            logger.debug("The matrix elements could not be imported")
            
        self.process=process
        self.E_cm=sanitize_variable(e_cm,device)
        self.delR_cut=sanitize_variable(delr_cut,device)
        self.pT_cut=sanitize_variable(pt_cut,device)
        self.rap_maxcut=sanitize_variable(rap_maxcut,device)
        self.default_device=device


        names=["d","u","s","c","b","t","g"]
        pdgs=[1,2,3,4,5,6,21]
        
        matrix2py.initialisemodel(BASE_DIR+"/integrands/mg/"+process+"/param/param_card.dat")
        file = open(BASE_DIR+"/integrands/mg/"+process+"/param/nexternal.inc", "r") 
        file.readline()

        n_external=file.readline()
        n_external = int((n_external.split("=")[1]).split(")")[0])
        file.readline()
        n_incoming=file.readline()
        n_incoming = int((n_incoming.split("=")[1]).split(")")[0])
        file.close
        
        file = open(BASE_DIR+"/integrands/mg/"+process+"/param/pmass.inc", "r") 
        z=file.readlines()

        z=[x.split("=")[1].split("\n")[0] for x in z]
        z=[x[::-1].split("(")[0][::-1].split(")")[0] for x in z]
        file.close

        file = open(BASE_DIR+"/integrands/mg/"+process+"/param/param.log", "r")
        p=file.readlines()
        file.close

        masses=[0]*len(z)
        external_masses=[0]*2
        Gf=  float([i for i in p if " mdl_gf " in i][0].split()[7])
        aEW=  1/float([i for i in p if " aewm1 " in i][0].split()[7])
        MZ= float( [i for i in p if " mdl_mz " in i][0].split()[7])
        
        for ider,x in enumerate(z):
            if x=="ZERO":
                masses[ider]=0.0
            elif x=="MDL_MW":
                
                masses[ider]=np.sqrt(MZ**2/2. + np.sqrt(MZ**4/4. - (aEW*np.pi*MZ**2)/(Gf*np.sqrt(2))))
            
            else:
                res = [i for i in p if " "+x.lower()+" " in i][0]
                masses[ider]=float(res.split()[7])
        external_masses[0]=masses[:n_incoming]
        external_masses[1]=masses[n_incoming:]

        
        process_name=process.split("P1_")[1]
        particles=process_name.split("_")[0]

        pdg=[0]*len(external_masses[0])
        offset1=0
        offset2=0
        
        for ide,x in enumerate(names):
            ider=ide-offset1-offset2
            marker=particles.find(x)
            
            if marker==0 and (x!='t' or(len(particles)<=2 or (particles[2]!='-' and particles[2]!='+' ))):
                
                pdg[offset1]=pdgs[ider]
                particles=particles[1:]
                
                if len(particles)>0 and particles[0]=="x":
                    pdg[offset1]*=-1
                    particles=particles[1:]
                    
                names.insert(ide,x)
                offset1+=1
            
                
            
            elif marker!=-1 and (x!='t' or(len(particles)<=2+marker or (particles[marker+2]!='p' and particles[marker+2]!='m' ))) :
                particles=particles[:marker]+particles[marker+1 :]
                pdg[1]=pdgs[ider]
                
                if len(particles)>marker and particles[marker]=="x":
                    particles=particles[:marker]+particles[marker+1 :]
                    pdg[1]*=-1
                names.insert(ide,x)
                offset2+=1
            if offset1+offset2==2:
                break
    
        self.pdg=pdg
                

        logger.info("Ingoing particles: "+str(len(external_masses[0])))
        logger.info("Ingoing pdg codes: "+str(pdg[0])+" "+str(pdg[1]))
        logger.info("PDFs active: "+str(self.pdf))
        logger.info("Outgoing particles: "+str(len(external_masses[1])))

        if((len(external_masses[0]))!=2 and self.pdf):
            logger.info("No PDF support for other cases than 2 body collision")
            self.pdf=False
        
        self.this_process_E_cm = max( self.E_cm, sum(external_masses[1])*2. )
        self.my_ps_generator=FlatInvertiblePhasespace(external_masses[0], external_masses[1],pdf=p,pdf_active=self.pdf, lhapdf_dir=lhapdf_dir)
        
        
        if not self.pdf:
            self.d = self.my_ps_generator.nDimPhaseSpace() # number of dimensions
        else:
            self.d = self.my_ps_generator.nDimPhaseSpace()+2 # number of dimensions
        
        super(CrossSection, self).__init__(self.d)


    def evaluate_integrand(self, x):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if BASE_DIR+"/integrands/mg/"+self.process not in sys.path:
            sys.path.append(BASE_DIR+"/integrands/mg/"+self.process)
        try:
            import matrix2py
        except:
            logger.debug("The matrix elements could not be imported")
            
        momenta, jac = self.my_ps_generator.generateKinematics_batch(self.this_process_E_cm, x,pT_mincut=self.pT_cut, delR_mincut=self.delR_cut, rap_maxcut=self.rap_maxcut, pdgs=self.pdg)
        momenta=momenta.cpu()
        jac=jac.cpu()
        element=[0]*momenta.shape[0]
    
        element=[matrix2py.smatrix(momenta[ind,:,:].t().tolist())*jac[ind]
                for ind, q in enumerate(element)]
        
    
        return torch.tensor(element,device=x.device)/(2.5681894616*10**(-9))
        
        
#'/home/ngoetz/project/zunis/experiments/benchmarks/utils/integrands/mg/P1_epem_emep', '/home/ngoetz/project/MG5_aMC_v2_8_1/HEPTools/lhapdf6_py3/lib/python3.7/site-packages'
    

