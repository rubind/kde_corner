# kde_corner
Makes corner plots from MCMC samples using kernel density estimation. Reflects samples around minimum/maximum sample to better handle samples distributed against a parameter limit. I recommend setting bw_method to 0.1. The scipy default oversmoothes.

Also useful for making your own contour plots outside of the corner plots:

```import matplotlib.pyplot as plt
kernel_eval, levels, xvals, yvals, kernel_FN = run_2D_KDE(samples_of_x, samples_of_x, contours = [0.317311, 0.0455003], steps = 100)
plt.contourf(xvals, yvals, kernel_eval, levels = levels)
```
