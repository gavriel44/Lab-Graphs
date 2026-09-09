"""Fit a Navier-Frenk-White dark matter density profile with a custom model.

Data: Milky Way dark matter density derived from HI 21 cm rotation-curve
measurements, spanning the full galactic radius range (not restricted to
outside the bulge). See ``data/milky_way_dark_matter_density.xlsx`` and the
accompanying lab report (``Dark Matter (1).pdf``) for the measurement and
data-reduction pipeline. Data extraction/analysis source:
https://github.com/gavriel44/dark-matter
"""

from pathlib import Path

from labgraphs import Model, fit, plot_fit, read_excel


def nfw_density(r, rho0, rs):
    x = r / rs
    return rho0 / (x * (1 + x) ** 2)


def main() -> None:
    data = read_excel(
        Path(__file__).parent.parent / "data" / "milky_way_dark_matter_density.xlsx",
        x="r (kpc)",
        y="ρ (M☉/kpc³)",
        sx="r_error (kpc)",
        sy="ρ_error (M☉/kpc³)",
    )

    model = Model(
        name="NFW",
        function=nfw_density,
        parameters=("rho0", "rs"),
    )
    result = fit(data, model, initial_guess=[1e9, 2.0])
    print(result.summary())

    fig, _ = plot_fit(result, title="NFW dark matter density profile · Milky Way")
    output = Path("outputs/nfw_density_profile.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
