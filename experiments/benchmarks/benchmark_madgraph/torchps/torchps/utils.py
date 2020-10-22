import torch
import math
import numpy as np

def set_square_t(inputt, square, negative=False):
        """Change the time component of this LorentzVector
        in such a way that self.square() = square.
        If negative is True, set the time component to be negative,
        else assume it is positive.
        """
        
        ret=torch.zeros_like(inputt)
       
        ret[:,0] = (rho2_t(inputt) + square) ** 0.5
        
        if negative: ret[:,0] *= -1
        ret[:,1:]=inputt[:,1:]
       
        return ret
    
def rho2_t(inputt):
        """Compute the radius squared. Vectorized."""
        
        return torch.sum(inputt[:,1:]*inputt[:,1:],-1)
    
def rho2_tt(inputt):
        """Compute the radius squared. Vectorized and for batches."""
        
        return torch.sum(inputt[:,: ,1:]**2,-1)
    
def boostVector_t(inputt):
        
        if torch.min(inputt[:,0]) <= 0. or torch.min(square_t(inputt)) < 0.:
            print("Invalid boost")
        
        return inputt[:,1:]/inputt[:,0].unsqueeze(1)
    
def square_t(inputt):
    if(inputt.shape[1]==4 or inputt.shape[0]==4):
        
        return dot_t(inputt,inputt)
    else:
        
        return torch.sum(inputt*inputt,-1)
    
def dot_t(inputa,inputb):
    """Dot product for four vectors"""
    return inputa[:,0]*inputb[:,0] - inputa[:,1]*inputb[:,1] - inputa[:,2]*inputb[:,2] - inputa[:,3]*inputb[:,3]

def dot_fb(inputa,inputb):
    """Dot product for four vectors in batches vectors"""
    return inputa[:,:,0]*inputb[:,:,0] + inputa[:,:,1]*inputb[:,:,1] + inputa[:,:,2]*inputb[:,:,2] 

def dot_s(inputa,inputb):
    """Dot product for space vectors"""
    return inputa[:,0]*inputb[:,0] + inputa[:,1]*inputb[:,1] + inputa[:,2]*inputb[:,2] 
    
def boost_t(inputt, boost_vector, gamma=-1.):
        """Transport inputt into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            boost_t(p,-boostVector(p))
        returns to (M,0,0,0).
        Version for a single phase pace point
        """
        
        b2 = square_t(boost_vector)
        if gamma < 0.:
            gamma = 1.0 / torch.sqrt(1.0 - b2)
        inputt_space = inputt[:,1:]
        
        bp = torch.sum(inputt_space*boost_vector,-1)
        
        gamma2=torch.where(b2>0, (gamma-1.0)/b2,torch.zeros_like(b2))
        
        factor = gamma2*bp + gamma*inputt[:,0]
        
        inputt_space+= factor.unsqueeze(1)*boost_vector
        
        inputt[:,0] = gamma*(inputt[:,0] + bp)
        
        return inputt
    
def boost_tt(inputt, boost_vector, gamma=-1.):
        """Transport inputt into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            boost_t(p,-boostVector(p))
        returns to (M,0,0,0).
        Version for a batch
        """
        
        b2 = square_t(boost_vector)
        if gamma < 0.:
            gamma = 1.0 / torch.sqrt(1.0 - b2)
        inputt_space = inputt[:,:,1:]
        
        bp = torch.sum(inputt_space*boost_vector,-1)
        
        gamma2=torch.where(b2>0, (gamma-1.0)/b2,torch.zeros_like(b2))
        
        factor = gamma2*bp + gamma*inputt[:,:,0]
        
        inputt_space+= factor.unsqueeze(-1)*boost_vector
        
        inputt[:,:,0] = gamma*(inputt[:,:,0] + bp)
        
        return inputt
    
def cosTheta_t(inputt):

        ptot =torch.sqrt(torch.dot(inputt[1:],inputt[1:]))
        assert (ptot > 0.)
        return inputt[3] / ptot
    


    

def phi_t(inputt):

    return torch.atan2(inputt[2], inputt[1])



def uniform_distr(r,minv,maxv):
        """distributes r uniformly within (min, max), with jacobian dvariable"""
        minv=torch.ones_like(r)*minv
        maxv=torch.ones_like(r)*maxv
        dvariable = (maxv-minv)
        variable = minv + dvariable*r
       
        #print(dvariable)
        return variable, dvariable
    
def boost_to_lab_frame( momenta, xb_1, xb_2):
    """Boost a phase-space point from the COM-frame to the lab frame, given Bjorken x's."""
    def boost_lf(momenta, xb_1, xb_2):
        ref_lab = (momenta[:,0,:]*xb_1.unsqueeze(-1) + momenta[:,1,:]*xb_2.unsqueeze(-1))
       
        if not ((rho2_t(ref_lab) == 0).any()):
            lab_boost = boostVector_t(ref_lab)
            
            return boost_tt(momenta,lab_boost.unsqueeze(1))
        else:
            return momenta
   
    return torch.where(( (xb_1!=torch.ones_like(xb_1))| (xb_2!=torch.ones_like(xb_2))).unsqueeze(-1).unsqueeze(-1),boost_lf(momenta, xb_1,xb_2) , momenta)
    
    
    
    
def pseudoRap(inputt, eps=np.finfo(float).eps**0.5, huge=np.finfo(float).max):
        """Compute pseudorapidity. Single PS point"""

        pt = torch.sqrt(torch.sum(inputt[:,1:3]**2,axis=-1))
       
        th = torch.atan2(pt, inputt[:,3])
        return torch.where((pt<eps) & (torch.abs(inputt[:,3])<eps), huge*torch.ones_like(inputt[:,3]),-torch.log(torch.tan(th/2.)))
    
    
def pseudoRap_t(inputt, eps=np.finfo(float).eps**0.5, huge=np.finfo(float).max):
        """Compute pseudorapidity. Batch"""

        pt = torch.sqrt(torch.sum(inputt[:,:,1:3]**2,axis=-1))
       
        th = torch.atan2(pt, inputt[:,:,3])
        return torch.where((pt<eps) & (torch.abs(inputt[:,:,3])<eps), huge*torch.ones_like(inputt[:,:,3]),-torch.log(torch.tan(th/2.)))
       

   
def getdelphi(inputt1, inputt2,eps=np.finfo(float).eps**0.5,huge=np.finfo(float).max):
    """Compute the phi-angle separation with inputt2."""
    
    pt1 = torch.sqrt(torch.sum(inputt1[:,1:3]**2,axis=-1))
    pt2 = torch.sqrt(torch.sum(inputt2[:,1:3]**2,axis=-1))
  
    tmp = inputt1[:,1]*inputt2[:,1]+ inputt1[:,2]*inputt2[:,2]
    tmp /= (pt1*pt2)
    returner=torch.where(torch.abs(tmp) > torch.ones_like(tmp),torch.acos(tmp/torch.abs(tmp)),torch.acos(tmp))
    returner=torch.where((pt1==0.) | (pt2==0.),huge*torch.ones_like(returner),returner)
    return returner

def deltaR(inputt1, inputt2):
    """Compute the deltaR separation with momentum p2."""

    delta_eta = pseudoRap(inputt1) - pseudoRap(inputt2)
    delta_phi = getdelphi(inputt1,inputt2)
    return torch.sqrt(delta_eta**2 + delta_phi**2)