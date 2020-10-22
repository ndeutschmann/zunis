import os


import torch
import logging
import math
import copy
import datetime
import sys
from torchps.utils import *

class PhaseSpaceGeneratorError(Exception):
    pass





#=========================================================================================
# Phase space generation
#=========================================================================================

class VirtualPhaseSpaceGenerator(object):

    def __init__(self, initial_masses, final_masses,
                  pdf=None, pdf_active=False,tau=True, lhapdf_dir=None):
        
        dev = torch.device("cuda:"+str(0)) if torch.cuda.is_available() else torch.device("cpu")
        self.initial_masses  = initial_masses
        self.masses_t        = torch.tensor(final_masses,requires_grad=False, dtype=torch.double, device=dev)
        self.n_initial       = len(initial_masses)
        self.n_final         = len(final_masses)
        
        self.pdf             = pdf
        self.pdf_active      = pdf_active
        self.tau             = tau
        if lhapdf_dir not in sys.path:
                sys.path.append(lhapdf_dir)
        import lhapdf
                    
       
      
    def generateKinematics(self, E_cm, random_variables):
        """Generate a phase-space point with fixed center of mass energy."""

        raise NotImplementedError
    

    def nDimPhaseSpace(self):
        """Return the number of random numbers required to produce
        a given multiplicity final state."""

        if self.n_final == 1:
            return 0
        return 3*self.n_final - 4


class FlatInvertiblePhasespace(VirtualPhaseSpaceGenerator):
    """Implementation following S. Platzer, arxiv:1308.2922"""

    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    epsilon_border = 1e-10

    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymore
    absolute_Ecm_min = 1.
    

   

    def __init__(self, *args, **opts):
        
        super(FlatInvertiblePhasespace, self).__init__(*args, **opts)
        if self.n_initial == 1:
            raise PhaseSpaceGeneratorError("This basic generator does not support decay topologies.")
        if self.n_initial >2:
            raise PhaseSpaceGeneratorError("This basic generator does not support more than 2 incoming particles.")
 
    @staticmethod
    def get_flatWeights(E_cm, n):
        """ Return the phase-space volume for a n massless final states.
        Vol(E_cm, n) = (2*pi)^(4-3n)*(pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
        """
        #includes full phase space factor
        if n==1: 
            # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
            # must typically be accounted for in the MC integration framework since we
            # don't have access to that here, so we just return 1.
            return 1.
        if not torch.is_tensor(E_cm):
            return math.pow(2*math.pi, 4-3*n)*math.pow((math.pi/2.0),n-1)*\
                (math.pow((E_cm**2),n-2)/(math.factorial(n-1)*math.factorial(n-2)))
        else:
            return math.pow(2*math.pi, 4-3*n)*math.pow((math.pi/2.0),n-1)*\
                (torch.pow((E_cm**2),n-2)/(math.factorial(n-1)*math.factorial(n-2)))
    
   
    
    def massless_map(self,x,exp):

        return (x**(exp))*((exp+1)-(exp)*x)

  
    
    @staticmethod
    def rho(M, N, m):
        """Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))"""
        
        Msqr = M**2
        
        return ((Msqr-(N+m)**2) * (Msqr-(N-m)**2) )**0.5 / (8.*Msqr)
    
    


   
    
    def get_pdfQ2(self, pdf, pdg, x, scale2):
        """ Call the PDF and return the corresponding density."""
   

        if pdf is None:
            return torch.ones_like(x)
       
        if pdg not in [21] and abs(pdg) not in range(1,7):
            return torch.ones_like(x)
      
        
        # Call to lhapdf API
        
        f = pdf.xfxQ2(pdg, x, scale2)
        
        

        return torch.tensor(f,dtype=torch.double, device=x.device)/x
    
    def generateKinematics_batch(self, E_cm, random_variables_full, pT_mincut=-1, delR_mincut=-1, rap_maxcut=-1,pdgs=[0,0]):
        """Generate a self.n_initial -> self.n_final phase-space point
        using the random variables passed in argument, including phase space cuts and PDFs
        
        """
        self.masses_t=self.masses_t.to(random_variables_full.device)
        self.collider_energy=E_cm
        # Make sure that none of the random_variables is NaN.
        if torch.is_tensor(random_variables_full) and torch.isnan(random_variables_full).any():
            raise PhaseSpaceGeneratorError("Some of the random variables passed "+
              "to the phase-space generator are NaN: %s"%str(random_variables.data.tolist()))
        
        wgt_jac=torch.ones(random_variables_full.shape[0],dtype=torch.double, device=random_variables_full.device)
        xb_1=torch.ones(random_variables_full.shape[0],dtype=torch.double, device=random_variables_full.device)
        xb_2=torch.ones(random_variables_full.shape[0],dtype=torch.double, device=random_variables_full.device)
        if not self.pdf_active:
            random_variables=random_variables_full
       
        else:
            
            
            random_variables=random_variables_full[:,:-2]
            if(self.tau):
                tot_final_state_masses = torch.sum(self.masses_t).tolist()

                tau_min = (max(tot_final_state_masses, 
                                  self.absolute_Ecm_min)/E_cm)**2*torch.ones_like(random_variables_full[:,-1])
                tau_max =torch.ones_like(random_variables_full[:,-1])
                tau, wgt_jac1 = uniform_distr(random_variables_full[:,-2],tau_min,tau_max)

                ycm_min = 0.5 * torch.log(tau)
                ycm_max = -ycm_min
                ycm, wgt_jac2 = uniform_distr(random_variables_full[:,-1],ycm_min,ycm_max)
                xb_1 = torch.sqrt(tau)*torch.exp(ycm)
                xb_2 = torch.sqrt(tau)*torch.exp(-ycm)

                E_cm = torch.sqrt(tau)*E_cm
                wgt_jac*=wgt_jac1*wgt_jac2
            else:
            
                xb_1=random_variables_full[:,-1]
                xb_2=random_variables_full[:,-2]
               
                E_cm=torch.sqrt(xb_1*xb_2)*E_cm
            
            p_energy=(torch.ones_like(xb_1)*(91.188)**2).to(random_variables.device)
            x_cut=torch.where(xb_1<1e-4,torch.zeros_like(xb_1),torch.ones_like(xb_1))
            x_cut=torch.where(xb_2<1e-4,torch.zeros_like(x_cut),x_cut)
            wgt_jac*=self.get_pdfQ2(self.pdf,pdgs[0],xb_1,p_energy)*self.get_pdfQ2(self.pdf,pdgs[1],xb_2,p_energy)*x_cut
            
            
        
        assert ( random_variables.shape[1]==self.nDimPhaseSpace())
        
        
       
        # The distribution weight of the generate PS point
        weight = torch.ones(random_variables.shape[0],dtype=torch.double, device=random_variables.device)
        weight*=wgt_jac
        output_momenta_t=[]
        self.masses_t=self.masses_t.to(random_variables.device)
        mass = self.masses_t[0]
        
        
        if not self.pdf_active:
            M    = [ 0. ]*(self.n_final-1)
            
            M[0] = E_cm
            
            M=torch.tensor(M,requires_grad=False, dtype=torch.double, device=random_variables.device)
            
            M=torch.unsqueeze(M,0).repeat(random_variables.shape[0],1)
            
            
        else:
            M =[[ 0. ]*(self.n_final-1)]*random_variables.shape[0]
            M=torch.tensor(M,requires_grad=False, dtype=torch.double, device=random_variables.device)
            M[:,0]=E_cm
            M.to(random_variables.device)
            E_cm.to(random_variables.device)
            self.masses_t.to(random_variables.device)
            
        weight *= self.generateIntermediatesMassive_batch(M, E_cm, random_variables)
       
        Q_t=torch.tensor([0., 0., 0., 0.],requires_grad=False, dtype=torch.double, device=random_variables.device)
        Q_t=Q_t.unsqueeze(0).repeat(random_variables.shape[0],1)
        Q_t[:,0]=M[:,0]
        M=torch.cat((M,self.masses_t.unsqueeze(0).repeat(random_variables.shape[0],1)[:,-1:]),-1)
        
        q_t=(4.*M[:,:-1]*self.rho(M[:,:-1],M[:,1:],self.masses_t[:-1].to(M.device)))
        
        rnd=random_variables[:,self.n_final-2:3*self.n_final-4]
        
        
        cos_theta_t=(2.*rnd[:,0::2]-1.)
        theta_t=torch.acos(cos_theta_t)
        sin_theta_t=(torch.sqrt(1.-cos_theta_t**2))
        
        
        phia_t=2*math.pi*rnd[:,1::2]
       
        
        cos_phi_t=torch.cos(phia_t)
        sqrt=torch.sqrt(1.-cos_phi_t**2)
        sin_phi_t=(torch.where(phia_t>math.pi,-sqrt,sqrt))
        a=torch.unsqueeze((q_t*sin_theta_t*cos_phi_t),0)
        b=torch.unsqueeze((q_t*sin_theta_t*sin_phi_t),0)
        c=torch.unsqueeze((q_t*cos_theta_t),0)
       
        lv=torch.cat((torch.zeros_like(a),a,b,c),0)
        
        output_returner=torch.zeros((random_variables.shape[0],self.n_initial+self.n_final,4),
                                    dtype=torch.double,device=random_variables.device)
        for i in range(self.n_initial+self.n_final-1):
            
            if i < self.n_initial:
                
                output_returner[:,i,:]=0
                continue

           
            p2 =(lv[:,:,i-self.n_initial].t())
           
            p2=set_square_t(p2,self.masses_t[i-self.n_initial]**2)
            
            p2=boost_t(p2,boostVector_t(Q_t)) 
            p2=set_square_t(p2,self.masses_t[i-self.n_initial]**2)
           
            output_returner[:,i,:]=p2
        
           
            
            nextQ_t=Q_t-p2
            
            nextQ_t=set_square_t(nextQ_t,M[:,i-self.n_initial+1]**2)
           
            Q_t = nextQ_t
        
        
        output_returner[:,-1,:]=Q_t
        
        self.setInitialStateMomenta_batch(output_returner,E_cm)
        
        output_returner_save=output_returner.clone()
        output_returner=boost_to_lab_frame(output_returner,xb_1,xb_2)
        
        q_theta2=torch.min(torch.abs(torch.sqrt(output_returner[:,2:,1]**2+output_returner[:,2:,2]**2)),axis=1).values

        factor2=torch.where(q_theta2<torch.ones_like(q_theta2)*pT_mincut,
                            torch.zeros_like(weight),torch.ones_like(weight))
      
        for i in range(output_returner[:,2:,:].shape[1]):
            for j in range(output_returner[:,2:,:].shape[1]):
                if i>j:
                    
           
                    factor2*=torch.where(torch.abs(deltaR(output_returner[:,i+2,:], output_returner[:,j+2,:]))
                                        <torch.ones_like(weight)*delR_mincut,torch.zeros_like(weight),torch.ones_like(weight))
 
        if(rap_maxcut>0):
            factor2*=torch.where(rap_maxcut<
                                torch.abs(torch.max(pseudoRap_t(output_returner[:,2:,:]),axis=1).values),
                                torch.zeros_like(weight),torch.ones_like(weight))
                           
           
        weight=weight*factor2
        
        
        shat=xb_1*xb_2*self.collider_energy**2
        return output_returner_save, weight/(2*shat)
        
       
    
    
    def bisect_vec_batch(self,v_t, target=1.e-16, maxLevel=600):
        """Solve v = (n+2) * u^(n+1) - (n+1) * u^(n+2) for u. Vectorized, batched"""
        if(v_t.size(1)==0):
            return
       
        exp=torch.arange(self.n_final-2,0,step=-1, device=v_t.device, dtype=torch.double)
        
        exp=exp.unsqueeze(0).repeat(v_t.shape[0],1)
        level = 0
        left  = torch.zeros_like(v_t)
        right = torch.ones_like(v_t)
            
        checkV = torch.ones_like(v_t)*-1
        u =torch.ones_like(v_t)*-1
        error=torch.ones_like(v_t)
        maxLevel=maxLevel/10
        ml=maxLevel
        oldError=100
        while(torch.max(error)>target and ml<10*maxLevel):
            
            while (level < ml):
                u = (left + right) * (0.5**(level + 1))

                checkV = self.massless_map(u,exp)


                left *= 2.
                right *= 2.
                con=torch.ones_like(left)*0.5
                adder=torch.where(v_t<=checkV, con*-1.,con)


                left=left+(adder+0.5)
                right=right+(adder-0.5)

                level += 1
           
            error=torch.abs(1. - checkV / v_t)
            
            ml=ml+maxLevel
            newError=torch.max(error)
            if(newError>=oldError):
                break
            else:
                oldError=newError
        
        return u
            

    
    def generateIntermediatesMassless_batch(self, M_t, E_cm, random_variables): 
        """Generate intermediate masses for a massless final state. Batch version"""
        
        
        u = self.bisect_vec_batch(random_variables[:,:self.n_final-2])
        
        for i in range(2, self.n_final):
            M_t[:,i-1] = torch.sqrt(u[:,i-2]*(M_t[:,i-2]**2))
        if not torch.is_tensor(E_cm):
            return torch.tensor([self.get_flatWeights(E_cm,self.n_final)]*random_variables.shape[0],
                            dtype=torch.double, device=random_variables.device)
        else:
            return self.get_flatWeights(E_cm,self.n_final)



    def generateIntermediatesMassive_batch(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massive final state. Batch version"""
        
       
        
        M[:,0] -= torch.sum(self.masses_t)
        
        weight = self.generateIntermediatesMassless_batch(M, E_cm, random_variables)
       
        
        K_t=M.clone()
        
        masses_sum=torch.flip(torch.cumsum(torch.flip(self.masses_t,(-1,)),-1),(-1,))
        M+=masses_sum[:-1].to(M.device)
        
        weight[:]*= 8.*self.rho(
            M[:,self.n_final-2],
            self.masses_t[self.n_final-1],
            self.masses_t[self.n_final-2] )
        
        
        weight[:]*=torch.prod((self.rho(M[:,:self.n_final-2],M[:,1:],self.masses_t[:self.n_final-2].to(M.device))/
                            self.rho(K_t[:,:self.n_final-2],K_t[:,1:],0.)) * (M[:,1:self.n_final-1]/K_t[:,1:self.n_final-1]),-1)
        
        weight[:] *= torch.pow(K_t[:,0]/M[:,0],2*self.n_final-4)
        
        
        return weight
 
    def setInitialStateMomenta_batch(self, output_momenta, E_cm):
        """Generate the initial state momenta. Batch version"""
        if self.n_initial not in [2]:
            raise PhaseSpaceGeneratorError(
               "This PS generator only supports 2 initial states")

        if self.n_initial == 2:
            if self.initial_masses[0] == 0. or self.initial_masses[1] == 0.:
                if not torch.is_tensor(E_cm):
                
                    output_momenta[:,0,:] = torch.tensor([E_cm/2.0 , 0., 0., +E_cm/2.0],dtype=torch.double, device=output_momenta[0].device)
                    output_momenta[:,1,:] = torch.tensor([E_cm/2.0 , 0., 0., -E_cm/2.0],dtype=torch.double, device=output_momenta[0].device)
                else:
                    E_cm_p=E_cm.unsqueeze(-1)
                    output_momenta[:,0,:]=torch.cat((E_cm_p/2, torch.zeros_like(E_cm_p),torch.zeros_like(E_cm_p), 1*E_cm_p/2),-1)
                    output_momenta[:,1,:]=torch.cat((E_cm_p/2, torch.zeros_like(E_cm_p),torch.zeros_like(E_cm_p), -1*E_cm_p/2),-1)
            else:
                M1sq = self.initial_masses[0]**2
                M2sq = self.initial_masses[1]**2
                E1 = (E_cm**2+M1sq-M2sq)/ E_cm
                E2 = (E_cm**2-M1sq+M2sq)/ E_cm
               
                if not torch.is_tensor(E_cm):
                    Z = math.sqrt(E_cm**4 - 2*E_cm**2*M1sq - 2*E_cm**2*M2sq + M1sq**2 - 2*M1sq*M2sq + M2sq**2) / E_cm
                    output_momenta[:,0,:] = torch.tensor([E1/2.0 , 0., 0., +Z/2.0],dtype=torch.double, device=output_momenta[0].device)
                    output_momenta[:,1,:] = torch.tensor([E2/2.0 , 0., 0., -Z/2.0],dtype=torch.double, device=output_momenta[0].device)
                else:
                    Z = torch.sqrt(E_cm**4 - 2*E_cm**2*M1sq - 2*E_cm**2*M2sq + M1sq**2 - 2*M1sq*M2sq + M2sq**2) / E_cm
                    E1_p=E1.unsqueeze(-1)
                    E2_p=E2.unsqueeze(-1)
                    Z_p=Z.unsqueeze(-1)
                    output_momenta[:,0,:]=torch.cat((E1_p/2, torch.zeros_like(E1_p),torch.zeros_like(E1_p), 1*Z_p/2),-1)
                    output_momenta[:,1,:]=torch.cat((E2_p/2, torch.zeros_like(E1_p),torch.zeros_like(E1_p), -1*Z_p/2),-1)
        return
     


