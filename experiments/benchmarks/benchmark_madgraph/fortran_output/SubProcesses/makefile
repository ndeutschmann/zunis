
# For python linking (require f2py part of numpy)
ifeq ($(origin MENUM),undefined)
  MENUM=2
endif
P1_epem_emep/matrix$(MENUM)py.so:
	cd P1_epem_emep;make matrix$(MENUM)py.so
P1_epem_mummup/matrix$(MENUM)py.so:
	cd P1_epem_mummup;make matrix$(MENUM)py.so
all:  P1_epem_emep/matrix$(MENUM)py.so  P1_epem_mummup/matrix$(MENUM)py.so 
	echo 'done'