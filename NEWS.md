# proteus 1.0.0

* Added a `NEWS.md` file to track changes to the package.


# proteus 1.1.0
* Automated estimation of differentiation variable (deriv).
* Automated detection of cuda compliant device.
* Various improvement and code consolidation.
* Added function for hyper-param random search.
* Added another loss function (score).

# proteus 1.1.1
* Added threshold parameter to control the automatic differentiation.
* Besides prediction, now you get also a sampler for each prediction point.

# proteus 1.1.2
* Substitution of threshold parameter with min_default.

# proteus 1.1.3
* Added option doing block validation in parallel. Improved numeric stability.

# proteus 1.1.4
* Solve issued about saturation of workers during parallelization. 
* Fixed default parameters setting for stride.
* Added skewed normal to the list of possible distribution.
