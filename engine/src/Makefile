#
# MaBoSS project Makefile wrapper
#
# Institut Curie, France
#
# Eric Viara, Gautier Stoll March 2011
#

MAKEFILE = Makefile.maboss

maboss:
%:
	@rm -f depend.mk
	@touch depend.mk
	$(MAKE) -f $(MAKEFILE) init
	$(MAKE) -f $(MAKEFILE) depend
	$(MAKE) -f $(MAKEFILE) $@
	
.PHONY: dn_exp

dn_exp:
	@rm -f depend.mk
	@touch depend.mk
	$(MAKE) -f $(MAKEFILE) init
	$(MAKE) -f $(MAKEFILE) USE_DYNAMIC_BITSET=1 depend
	$(MAKE) -f $(MAKEFILE) USE_DYNAMIC_BITSET=1 maboss

tar_perform:
	$(MAKE) -f $(MAKEFILE) $@

clean:
	@rm -f depend.mk
	@touch depend.mk
	$(MAKE) -f $(MAKEFILE) $@
	rm -f *.o MaBoSS_[0-9]*n MaBoSS_[0-9]*n-server MaBoSS_[0-9]*n-client
