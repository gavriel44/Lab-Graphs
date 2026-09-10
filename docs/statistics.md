# Statistical conventions

LabGraphs fits explicit scalar-response models `y = f(x, β)` with independent
observations. The supplied `sx` and `sy` are standard deviations in the same units
as x and y, not variances or confidence-interval widths.

## Objective and residuals

When `sx` is present, ODR jointly estimates β and corrections δ to x by minimizing

```text
χ² = Σᵢ [(δᵢ / sxᵢ)² + ((f(xᵢ + δᵢ, β) − yᵢ) / syᵢ)²]
```

The result stores `x_adjustments = δ` and
`y_adjustments = f(x + δ, β) − y`. These are corrections **added** to the observed
points to reach the fitted curve. The weighted squares of these corrections sum
to `chi_square`.

With `sx=None`, x is fixed, δ is zero, and the objective reduces to weighted least
squares in y. Zero sx values are rejected rather than assigned an ambiguous weight.
Mixed exact/uncertain x observations are not currently supported.

For the plot, `residuals = y − f(x, β)`. The plotted error bars are the original
measurement errors. They do not incorporate fitted-parameter uncertainty or
constitute a normalized orthogonal residual test.

## Covariance

ODRPACK returns an unscaled parameter covariance `cov_beta` plus standard errors
scaled by the residual variance. LabGraphs deliberately exposes a consistent pair:

```text
dof = N − P
reduced_chi_square = χ² / dof

absolute_sigma=True:   covariance = cov_beta
absolute_sigma=False:  covariance = cov_beta × reduced_chi_square

standard_errors = sqrt(diag(covariance))
```

This covariance is a local linear approximation around the solution. Parameter
nonlinearity, poor identifiability, small samples, and multiple optima can make
that approximation inadequate. Solver convergence alone does not assess those
assumptions. LabGraphs rejects reported rank deficiency and unsuccessful stopping
codes instead of presenting them as valid fits. It also rejects a solver-reported
inverse condition number at or below `sqrt(machine epsilon)` (about `1.5e-8` for
float64), where parameter separation becomes numerically unreliable. A poorly
scaled but otherwise valid model may need its data or parameters expressed in
different units to pass this check.

## Goodness of fit

The reported p-value is `scipy.stats.chi2.sf(χ², dof)`, evaluated on the original
weighted objective even when covariance scaling is requested. Its interpretation
assumes independent Gaussian errors and a correctly specified model and uncertainty
scale. The chi-square approximation may be unreliable for nonlinear models or
small samples. An arbitrary error scale produces an arbitrary p-value.

Input validation checks shape, numeric finiteness, positive uncertainties, and
positive degrees of freedom. It cannot establish physical validity or detect all
systematic errors. The API does not estimate unknown measurement uncertainties,
discard outliers, or automatically choose a model.

## References

- [ODRPACK Python API](https://hugomvale.github.io/odrpack-python/reference/)
- [ODRPACK95 user guide](https://github.com/HugoMVale/odrpack95/blob/main/original/Doc/guide.pdf)
- [SciPy covariance scaling convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html)

The numerical tests independently verify absolute covariance against the inverse
weighted linear normal matrix, and ODR coefficients against the closed-form
constant-variance Deming solution.
