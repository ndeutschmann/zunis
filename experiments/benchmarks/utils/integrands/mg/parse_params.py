import logging

logger = logging.getLogger(__name__)

class ParseParams():
    """
    This class parses the fortran output files of MG5 in order
    to extract the PDG codes and masses of external particles.
        Parameters
        ----------
        process : String
            The name of the subprocesses "P1_x_x"
        exec_dir:      String
            The base directory of the execution file
    """

    def __init__(self,process, exec_dir):

        self.process=process
        self.exec_dir=exec_dir

        return

    def parse(self):
        """
            Main parsing method, executes the substeps.
        """

        #extract the number of ingoing/external particles
        (n_external, n_incoming)=self.nexternal_parse()

        #extract the name of the mass parameters
        mass_tags=self.pmass_parse()

        #external masses will contain two arrays:
        #the array of incoming and outgoing masses
        external_masses=[0]*2

        #determine the value of the masses by parsing the model parameters
        (external_masses[0],external_masses[1])=self.param_parse(n_incoming,mass_tags)

        #parse the process name to determine the pdg codes
        pdg=self.pdg_parse(external_masses)

        return (pdg,external_masses)

    def nexternal_parse(self):
        """

            nexternal.inc has the structure:
            1 datatype    "NEXTERNAL"
            2 "PARAMETER (NEXTERNAL=" number of external particles ")"
            3 datatype    "NINCOMING"
            4 "PARAMETER (NINCOMING=" number of incoming particles ")"
            this parses line 2 and and 4 to extract the number of in- and outgoing particles

        """
        file = open(self.exec_dir+"/integrands/mg/"+self.process+"/param/nexternal.inc", "r")

        #skip first line
        file.readline()
        n_external=file.readline()
        #parse line 2
        # The string is split at the "=" to extract the float after it
        n_external = int((n_external.split("=")[1]).split(")")[0])
        #skip third line
        file.readline()
        n_incoming=file.readline()
        #parse line 4
        n_incoming = int((n_incoming.split("=")[1]).split(")")[0])
        file.close()

        return (n_external, n_incoming)

    def pmass_parse(self):
        """

            pmass.inc has the structure
            "PMASS(" number of particle ")=" mass
            if the mass is zero, mass is the string "ZERO"
            if the particle is a W-boson, mass ist the string "MDL_MW"
            else the mass is a string "MDL_M"+particle name in caps
            the order of the particles in the file is identical to the order of the particles
            in the process name
            all lines are read in and the value of the mass is extracted.

        """
        file = open(self.exec_dir+"/integrands/mg/"+self.process+"/param/pmass.inc", "r")
        z=file.readlines()
        # The string is split at the "=" to extract the float after it
        z=[x.split("=")[1].split("\n")[0] for x in z]
        z=[x[::-1].split("(")[0][::-1].split(")")[0] for x in z]
        file.close()

        return z

    def param_parse(self,n_incoming,mass_tags):
        """

            param.log contains the parameters of the model
            it has the structure:
            "Parameter" paramtername  "              has been read with value " paramter_value
            all lines are read and the values for "mdl_gf", "aewm1" and "mdl_mz" are used
            to calculate the vale of the mass for the W-Boson, which is not contained in the file.

        """
        file = open(self.exec_dir+"/integrands/mg/"+self.process+"/param/param.log", "r")
        p=file.readlines()
        file.close()

        #an array of particle masses is created
        masses=[0]*len(mass_tags)

        #the relevant strings are split after the spaces
        #the 7th subarray contains then the value
        Gf=  float([i for i in p if " mdl_gf " in i][0].split()[7])
        aEW=  1/float([i for i in p if " aewm1 " in i][0].split()[7])
        MZ= float( [i for i in p if " mdl_mz " in i][0].split()[7])

        #the values read from "pmass_inc" stored in array z are translated to floats
        for j,x in enumerate(mass_tags):
            if x=="ZERO":
                masses[j]=0.0
            elif x=="MDL_MW":

                masses[j]=np.sqrt(MZ**2/2. + np.sqrt(MZ**4/4. - (aEW*np.pi*MZ**2)/(Gf*np.sqrt(2))))

            else:
                #the masses of all massive particles except the W boson are readily availbe in
                #this file. In order to find the corresponding line, each line is scanned if it
                #contains the particle name. The two spaces are necessary to not mix up the mass
                # of the top-quark (" mdl_mt ") and the tau (" mdl_mta ")
                res = [i for i in p if " "+x.lower()+" " in i][0]
                masses[j]=float(res.split()[7])
        #the masses of the incoming and outgoing particles are separeated
        return(masses[:n_incoming],masses[n_incoming:])

    def pdg_parse(self,external_masses):
        """

            Parsing the involved particles from the process name and mapping on pdg codes
            pdg code 0 refers to color-neutral particles
            the automatic process names start with "P1_". The incoming and outgoing
            particles are seperated by a "_". Processing the first part of theprocess code
            allows to identify the incoming particles which is needed to include the PDFs.

        """

        #the mapping between particles and pdg-codes
        names=["d","u","s","c","b","t","g"]
        pdgs=[1,2,3,4,5,6,21]

        process_name=self.process.split("P1_")[1]
        #take only the part for incoming particles
        particles=process_name.split("_")[0]

        pdg=[0]*len(external_masses[0])
        #the offset flags that a particle has been identified at the beginning or
        #the middle of the string. offset1 ensures that the information on pdg[0]
        #is not overwritten if both particles were found at the beginning of the string
        #We need a separate offset2 as the particle can be found at the second half of
        #the string first too
        offset1=0
        offset2=0

        #we go through the list of all particles which are relevant for PDFs
        for k,x in enumerate(names):
            l=k-offset1-offset2
            #we try to find the name in the string
            marker=particles.find(x)

            #if the particle name is on the first position and it is either not t (which could be t(op) or ta+/ta-),
            #or the length is maximum 3 (which is not the cause if tau's are involved) or the 3 position is a +/-,
            #we know that the first position is a quark
            if marker==0 and (x!='t' or(len(particles)<=2 or (particles[2]!='-' and particles[2]!='+' ))):
                #convert the name in pdg code
                pdg[offset1]=pdgs[l]
                #erase the char from the string
                particles=particles[1:]

                #if an antiparticle flag is detected
                if len(particles)>0 and particles[0]=="x":
                    #change the pdg code to anti
                    pdg[offset1]*=-1
                    particles=particles[1:]
                #insert the name again in the names list, so that each quark can be found twice
                names.insert(k,x)
                offset1+=1


            #if the particle name is found at the second position, it is a quark if the name does either not start with t
            #or the length of the array after the start of the name is smaller than 2
            elif marker!=-1 and (x!='t' or (len(particles)<2+marker) ) and ( marker==0 or particles[marker-1]!='m'):
                print(marker)
                print(particles[marker-1])
                #remove the char from the string
                particles=particles[:marker]+particles[marker+1 :]
                #convert the name in pdg code
                pdg[1]=pdgs[l]

                #if an antiparticle flag is detected
                if len(particles)>marker and particles[marker]=="x":
                        #change the pdg code to anti
                    pdg[1]*=-1
                    #remove the flag
                    particles=particles[:marker]+particles[marker+1 :]
                names.insert(k,x)
                #insert the name again in the names list, so that each quark can be found twice
                offset2+=1
            #if two quarks/gluons were found, the search can be stopped
            if offset1+offset2==2:
                break
        #if less than two names where found, the pdg code remains default 0
        return pdg
