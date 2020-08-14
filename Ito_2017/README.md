
## Directory organization:
**Module:** informationtransfermapping.py
* Contains demo code and demo functions for performing information transfer mapping. See EmpiricalResults demo for example implementation. 

**Directory:** TheoreticalResults/
* Contains demos for our computational (simulation) model results. Specifically, replicates Fig. 4 and Supplementary Fig. 3. See subdirectory for other details. 

**Directory:** EmpiricalResults/
* Contains demos for our empirical network-to-network information transfer mapping results. Specifically, replicates Supplementary Fig. 1. See subdirectory for other details.

**Directory:** MasterScripts/
* Contains all raw code used to run analyses and generate figures from HCP minimally preprocessed material. Code is not organized as "tutorial" code. See subdirectory for more details.

**Directory:** utils/
* **File: multregressionconnectivity.py** Contains a function to estimate functional connectivity using multiple linear regression.
* **File: permutationTesting.py** Contains a function to run FWE-correction using permutation testing.
