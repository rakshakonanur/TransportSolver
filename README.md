# TransportSolver

### Implementation of advection-diffusion, perfusion equations on vasculature

How to run:
1. In main of 1d_mesh.py, set path = output directory of 1D simulation (see syntheticVasculature repo)
2. In test_full_model folder, run 1D_mesh.py to generate .geo file of the geometry.
3. Inputs: branchingData.csv (located in C:\Users\rkona\Documents\syntheticVasculature\1D Output)
4. Using command-line: convert branched_network.geo to branched_network.msh (gmsh branched_network.geo -1 -format msh2 -o branched_network.msh)
5. Run bifurcation.py (Converts mesh to .xdmf, computes centerline velocity from flowrate/cross-sectional area, interpolates to bifrucation.xdmf using nearest neighbor, solves advection-diffusion)
