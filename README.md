# kde_corner
Makes corner plots from MCMC samples using kernel density estimation. Reflects samples around minimum/maximum sample to better handle samples distributed against a parameter limit. I recommend setting bw_method to 0.1. The scipy default oversmoothes.
