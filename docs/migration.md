# Migrating from the original scripts

Version 0.1 replaces the experiment-specific source layout with the installable
`labgraphs` package. The old imports are intentionally not maintained.

| Earlier interface | Current interface |
| --- | --- |
| `DataTable.from_excel(path, column_indexes)` | `read_excel(path, x="…", y="…", sx="…", sy="…")` |
| `DataTable.from_list(...)` | `Dataset(x, y, sx=…, sy=…)` |
| `LinearFit()` | `models.linear` |
| `PolynomialFit()` | `models.quadratic` |
| `DataFitter(data, function).fit()` | `fit(data, model)` |
| Result dictionary | `FitResult` attributes |
| `print_fit_output(results)` | `print(result.summary())` |
| `PlotterParams` and `Plotter` | `fig, axes = plot_fit(result)` |

Model functions now take `function(x, *parameters)`. Adapt an older
`function(parameters, x)` with a wrapper, for example:

```python
model = Model("My model", lambda x, *p: old_function(p, x), parameters=("a", "b"))
```

The built-in sinusoidal model now orders parameters as **offset, amplitude,
angular frequency, phase**. Its former order was offset, angular frequency,
phase, amplitude. Linear, quadratic, and the newer exponential equation keep their
original parameter order. The procedural `old_main.py` exponential used a lifetime
instead of a rate; convert with `rate = -1 / lifetime`.

Optics and magnetism equations can be supplied as custom `Model` instances. For
example, the former optics equation becomes:

```python
model = Model(
    "Thin lens",
    lambda x, offset, focal_length: offset + focal_length * x / (x - focal_length),
    parameters=("offset", "focal_length"),
)
```

Use starting values and x values away from the equation's singularity.

The old parameter errors were scaled by residual variance, but the covariance
was not. Use `absolute_sigma=False` to recover the old error-scaling convention;
the returned covariance now receives the same scaling. The default uses absolute
measurement uncertainties. Missing sx now explicitly means exact x.

The CRT photo scripts, hardcoded entry points, and IDE metadata were removed from
the public project. They remain in Git history. Original local spreadsheets are
kept in the ignored `data/local/` directory. Reproducible synthetic examples replace
machine-specific inputs.
