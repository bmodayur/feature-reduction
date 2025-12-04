from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

from early_markers.cribsy.common.constants import (
    PLOT_DIR,
    RAND_STATE,
)
from early_markers.cribsy.common.enums import Metric


@dataclass
class BamResult:
    """A data class to store and process the results of a BAM simulation.

    Attributes:
        width (float): The target HPD width for this simulation.
        assurances (list[float]): The calculated assurance probabilities for each sample size.
        sizes (list[int]): The sample sizes tested.
        fig_path (Path): The file path where the output plot is saved.
    """
    width: float
    assurances: list[float]
    sizes: list[int]
    fig_path: Path

    def _interpolate(self, a, range_a, range_b):
        if range_a[0] == range_a[1]:  # Handle division by zero case
            return (range_b[0] + range_b[1]) / 2

        return np.interp(a, range_a, range_b)

    def _segment_list(self, lst: list) -> list[tuple]:
        return [(lst[i], lst[i+1]) for i in range(len(lst)-1)]

    def _estimate_y(self, tgt_x: float, x_ranges: list[float | int], y_ranges: list[float | int]) -> float | int:
        rng_x = self._segment_list(x_ranges)
        rng_y = self._segment_list(y_ranges)

        for i, rng in enumerate(rng_x):
            if tgt_x < rng_x[0][0]:
                return rng_y[0][0]
            if tgt_x > rng_x[-1][1]:
                return rng_y[-1][1]
            if rng[0] <= tgt_x <= rng[1]:
                return self._interpolate(tgt_x, rng_x[i], rng_y[i])

    def estimate_size_from_assurance(self, tgt_assurance: float, assurance_ranges: list[float] | list[list[float]], size_ranges: list[int] | list[list[int]]) -> int:
        return self._estimate_y(tgt_assurance, assurance_ranges, size_ranges)

    def estimate_assurance_from_size(self, tgt_size: float, assurance_ranges: list[float] | list[list[float]], size_ranges: list[int] | list[list[int]]) -> int:
        return self._estimate_y(tgt_size, size_ranges, assurance_ranges)


@dataclass
class Bammer:
    """Orchestrates the Bayesian Assurance Method (BAM) simulation.

    This class sets up and runs a simulation to determine the sample size
    needed to achieve a desired precision (HPD width) for a given diagnostic
    metric, based on pilot data.

    Attributes:
        metric (Metric): The diagnostic metric being analyzed (e.g., SENS, SPEC).
        numerator (int): The number of correct classifications in the pilot study.
        denominator (int): The total number of relevant cases in the pilot study.
        max_sample (int): The maximum sample size to test in the simulation.
        num_cores (int): The number of cores to use for the simulation.
        widths (list[float] | None): A list of target HPD widths to analyze.
        results (dict[float, BamResult]): A dictionary to store the `BamResult`
            for each target width analyzed.
    """
    metric: Metric
    numerator: int
    denominator: int
    max_sample: int
    num_cores: int
    widths: list[float] | None = None
    results: dict[float, BamResult] = field(default_factory=dict, init=False)

    @property
    def estimate(self):
        return self.numerator / self.denominator

    # def __post_init__(self):
    #     if self.widths is not None:
    #         for width in self.widths:
    #             self._plot_assurance_curve(width)

    def _plot_assurance_curve(self, target_width):
        """Calculate and plot Bayesian assurance curve with threshold"""
        # Calculate assurance probabilities
        sample_sizes, assurances = [], []

        # Test sample sizes in increments of 50 for better visualization
        for n in tqdm(range(50, self.max_sample+1, 50)):
            logger.debug(f"Testing Sample Size: {n}")
            with pm.Model() as model:
                theta = pm.Beta("theta",
                               alpha=self.numerator + 1,
                               beta=self.denominator - self.numerator + 1)
                y_pred = pm.Binomial("y_pred", n=n, p=theta)

                prior_pred = pm.sample_prior_predictive(
                    draws=500,
                    random_seed=RAND_STATE,
                )

            hpd_widths = []
            for s in prior_pred.prior["y_pred"].data.flatten():
                with pm.Model() as post_model:
                    theta_post = pm.Beta("theta_post",
                                        alpha=self.numerator + s + 1,
                                        beta=self.denominator - self.numerator + (n - s) + 1)
                    trace = pm.sample(
                        draws=2000,
                        cores=self.num_cores,
                        # random_seed=RAND_STATE,
                        progressbar=False
                    )
                    hdi = az.hdi(trace.posterior["theta_post"]).theta_post
                    hpd_widths.append(float(hdi.max() - hdi.min()))

            assurance_prob = np.mean(np.array(hpd_widths) <= target_width)
            sample_sizes.append(n)
            assurances.append(assurance_prob)

            if assurance_prob >= 0.95:
                break  # Stop when reaching threshold

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Plot assurance curve
        plt.plot(sample_sizes, assurances,
                 marker='o', linestyle='-',
                 color='#2c7bb6', linewidth=2.5,
                 markersize=8, markerfacecolor='#fdae61')

        # Threshold line
        plt.axhline(0.95, color='#d7191c', linestyle='--', linewidth=2,
                    label='95% Assurance Threshold')

        # Optimal sample size marker
        if len(assurances) > 0 and max(assurances) >= 0.95:
            optimal_n = sample_sizes[np.argmax(np.array(assurances) >= 0.95)]
            plt.axvline(optimal_n, color='#2c7bb6', linestyle=':', linewidth=2,
                       label=f'Optimal N = {optimal_n}')

        # Formatting
        plt.title('Bayesian Assurance Curve\
'
                 f'Pilot Data - {self.metric.value}: {self.estimate:0.3f} (Numerator: {self.numerator}  | Denominator {self.denominator})\
'
                 f'Target HPD Width: {target_width}', pad=20)
        plt.xlabel('Sample Size', labelpad=15)
        plt.ylabel('Assurance Probability', labelpad=15)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.xlim(0, max(sample_sizes)*1.1)
        plt.ylim(0, 1.05)

        # Annotation
        plt.annotate(f'Pilot {self.metric.value}: {self.numerator/self.denominator:.2f}',
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    fontsize=10, color='#636363')

        plt.tight_layout()
        fig = plt.gcf()

        w = f"{target_width:0.3f}".replace(".", "_")
        file_path = Path(PLOT_DIR) / f"bac_{self.metric.value.lower()}_{w}.png"
        fig.savefig(file_path)

        result = BamResult(
            width=target_width,
            assurances=assurances,
            sizes=sample_sizes,
            fig_path=file_path,
        )
        self.results[target_width] = result

    def set_bams_for_widths(self, widths: list[float]):
        if self.widths is None:
            self.widths = widths
        else:
            self.widths.extend(widths)

        for width in widths:
            self._plot_assurance_curve(target_width=width)
