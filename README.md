# Scattering from graphene plasmonic nanoresonator
This is a coordinate transform example for obtaining scattering spectra from a graphene plasmonic resonator with Ef=0.64 and μ=1000cm^2/V.s.

The resonator is square shaped with a size of 75nm*60nm and z=1/(resolution*stretch), giving a single pixel thickness after the coordinate transform. Detailed convergence testing hasn't been completed but the results for a thickness of 5nm (resolution=10, stretching=20, alpha=1um) are already very good.  

In order to obtain correct results, careful testing of the distance of the structure from the PML is required. The air block surrounding the graphene structure must also be carefully defined when changing the stretch factor.

Normalising to get a proper value of the scattering cross section might need some tuning, but the shape and frequency of the plasmon peak is pretty much correct.

Please verify that your results are correct when using this code. As the author I claim no responsibility for any errors.
