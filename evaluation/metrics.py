"""Selective Classification evaluation metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score, auc
import torch


class SelectionMetrics:
    """Selective Classification metrics: AUROC, AURC, E-AURC."""

    @staticmethod
    def compute_auroc(confidences: np.ndarray, correctness: np.ndarray) -> float:
        """
        Compute AUROC (Area Under ROC Curve).

        Measures the ability to distinguish correct from incorrect predictions.

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}

        Returns:
            AUROC value [0, 1]
        """
        if len(np.unique(correctness)) < 2:
            # If all same class, return NaN or 0.5
            return 0.5

        return roc_auc_score(correctness, confidences)

    @staticmethod
    def compute_aurc(
        confidences: np.ndarray,
        correctness: np.ndarray,
        coverage_levels: list = None,
    ) -> float:
        """
        Compute AURC (Area Under Risk-Coverage curve).

        Measures risk (error rate) at different coverage levels.
        Lower AURC is better.

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}
            coverage_levels: Coverage percentages to evaluate [0-100]

        Returns:
            AURC value [0, 1]
        """
        if coverage_levels is None:
            coverage_levels = list(range(0, 101, 5))

        # Sort by confidence (descending)
        n = len(confidences)
        sorted_idx = np.argsort(-confidences)

        risks = []
        coverages = []

        for cov_level in coverage_levels:
            # Number of samples to select
            n_select = max(1, int(n * cov_level / 100))
            coverage = cov_level / 100.0
            coverages.append(coverage)

            # Get selected samples
            selected_correct = correctness[sorted_idx[:n_select]]

            # Risk = error rate
            if n_select > 0:
                risk = 1.0 - np.mean(selected_correct)
            else:
                risk = 1.0

            risks.append(risk)

        # Compute area under risk-coverage curve
        # Risk at coverage=0 is undefined, we interpolate
        aurc = auc(coverages, risks)
        return aurc

    @staticmethod
    def compute_eaurc(
        confidences: np.ndarray,
        correctness: np.ndarray,
        coverage_levels: list = None,
    ) -> float:
        """
        Compute E-AURC (Excess AURC).

        Normalized AURC that compares to optimal selective classifier.
        E-AURC = AURC - AURC_optimal
        Lower E-AURC is better.

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}
            coverage_levels: Coverage percentages to evaluate [0-100]

        Returns:
            E-AURC value [0, 1]
        """
        if coverage_levels is None:
            coverage_levels = list(range(0, 101, 5))

        n = len(confidences)
        sorted_idx = np.argsort(-confidences)

        risks = []
        coverages = []

        for cov_level in coverage_levels:
            n_select = max(1, int(n * cov_level / 100))
            coverage = cov_level / 100.0
            coverages.append(coverage)

            selected_correct = correctness[sorted_idx[:n_select]]
            if n_select > 0:
                risk = 1.0 - np.mean(selected_correct)
            else:
                risk = 1.0

            risks.append(risk)

        # AURC
        aurc = auc(coverages, risks)

        # Optimal AURC: oracle that perfectly selects correct predictions
        # Order by correctness (descending)
        optimal_idx = np.argsort(-correctness)
        optimal_risks = []

        for cov_level in coverage_levels:
            n_select = max(1, int(n * cov_level / 100))
            selected_correct = correctness[optimal_idx[:n_select]]
            if n_select > 0:
                risk = 1.0 - np.mean(selected_correct)
            else:
                risk = 1.0
            optimal_risks.append(risk)

        aurc_optimal = auc(coverages, optimal_risks)

        # E-AURC = AURC - AURC_optimal
        eaurc = aurc - aurc_optimal
        return max(0, eaurc)  # Clamp to [0, inf)

    @staticmethod
    def compute_all_metrics(
        confidences: np.ndarray,
        correctness: np.ndarray,
        coverage_levels: list = None,
    ) -> dict:
        """
        Compute all SC metrics.

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}
            coverage_levels: Coverage percentages to evaluate [0-100]

        Returns:
            Dictionary with AUROC, AURC, and E-AURC
        """
        if coverage_levels is None:
            coverage_levels = list(range(0, 101, 5))

        return {
            "auroc": SelectionMetrics.compute_auroc(confidences, correctness),
            "aurc": SelectionMetrics.compute_aurc(confidences, correctness, coverage_levels),
            "eaurc": SelectionMetrics.compute_eaurc(confidences, correctness, coverage_levels),
        }

    @staticmethod
    def compute_confidence_distribution(confidences: np.ndarray) -> dict:
        """
        Compute statistics about confidence distribution.

        Args:
            confidences: Model confidence/max probability [0,1]

        Returns:
            Dictionary with confidence statistics
        """
        return {
            "conf_mean": float(np.mean(confidences)),
            "conf_std": float(np.std(confidences)),
            "conf_min": float(np.min(confidences)),
            "conf_max": float(np.max(confidences)),
            "conf_median": float(np.median(confidences)),
        }


class CalibrationMetrics:
    """Calibration metrics to measure confidence-accuracy alignment."""

    @staticmethod
    def compute_ece(
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures average difference between confidence and accuracy across bins.
        Lower is better (0 = perfectly calibrated).

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}
            n_bins: Number of confidence bins

        Returns:
            ECE value [0, 1]
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(confidences)

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # Get samples in this confidence bin
            in_bin = (confidences >= lower) & (confidences < upper)
            if i == n_bins - 1:  # Include rightmost boundary
                in_bin = (confidences >= lower) & (confidences <= upper)

            n_in_bin = np.sum(in_bin)

            if n_in_bin == 0:
                continue

            # Accuracy in this bin
            accuracy_in_bin = np.mean(correctness[in_bin])

            # Average confidence in this bin
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            # Contribution to ECE
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (n_in_bin / total_samples)

        return ece

    @staticmethod
    def compute_mce(
        confidences: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).

        Maximum difference between confidence and accuracy across all bins.
        Lower is better (0 = perfectly calibrated).

        Args:
            confidences: Model confidence/max probability [0,1]
            correctness: Binary correctness labels {0, 1}
            n_bins: Number of confidence bins

        Returns:
            MCE value [0, 1]
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            in_bin = (confidences >= lower) & (confidences < upper)
            if i == n_bins - 1:
                in_bin = (confidences >= lower) & (confidences <= upper)

            n_in_bin = np.sum(in_bin)

            if n_in_bin == 0:
                continue

            accuracy_in_bin = np.mean(correctness[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])

            mce = max(mce, np.abs(accuracy_in_bin - avg_confidence_in_bin))

        return mce

    @staticmethod
    def compute_brier_score(
        probs: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute Brier Score.

        Mean squared difference between predicted probabilities and true labels.
        Lower is better (0 = perfect predictions).

        Args:
            probs: Probability predictions [N, C]
            targets: Ground truth labels [N] (integers)

        Returns:
            Brier score [0, 1]
        """
        n_classes = probs.shape[1]
        n_samples = len(targets)

        # Convert targets to one-hot
        targets_onehot = np.eye(n_classes)[targets]

        # Brier score = mean((pred - true)^2)
        brier = np.mean((probs - targets_onehot) ** 2)

        return brier

    @staticmethod
    def compute_nll(
        probs: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """
        Compute Negative Log-Likelihood (NLL).

        Cross-entropy loss (lower is better, measures surprise).

        Args:
            probs: Probability predictions [N, C]
            targets: Ground truth labels [N] (integers)

        Returns:
            NLL value (typically 0-5 for well-trained models)
        """
        n_samples = len(targets)

        # Clamp probabilities to avoid log(0)
        probs = np.clip(probs, 1e-7, 1.0)

        # NLL = -log(P(correct_class))
        nll = 0.0
        for i in range(n_samples):
            nll -= np.log(probs[i, targets[i]])

        nll = nll / n_samples

        return nll

    @staticmethod
    def compute_all_calibration_metrics(
        confidences: np.ndarray,
        correctness: np.ndarray,
        probs: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Compute all calibration metrics at once.

        Args:
            confidences: Model confidence [0,1]
            correctness: Binary correctness labels {0, 1}
            probs: Probability predictions [N, C]
            targets: Ground truth labels [N]
            n_bins: Number of bins for ECE/MCE

        Returns:
            Dictionary with all calibration metrics
        """
        return {
            "ece": CalibrationMetrics.compute_ece(confidences, correctness, n_bins),
            "mce": CalibrationMetrics.compute_mce(confidences, correctness, n_bins),
            "brier": CalibrationMetrics.compute_brier_score(probs, targets),
            "nll": CalibrationMetrics.compute_nll(probs, targets),
        }


def compute_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple:
    """
    Compute confidence and correctness from logits.

    Args:
        logits: Model logits [N, C]
        targets: Ground truth labels [N] or [N, 1]

    Returns:
        tuple: (confidences, correctness) as 1D numpy arrays
    """
    logits = logits.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Flatten targets in case they have shape [N, 1]
    targets = np.atleast_1d(targets).flatten()

    # Confidence = max softmax probability
    probs = torch.nn.functional.softmax(
        torch.tensor(logits), dim=1
    ).numpy()
    confidences = np.max(probs, axis=1)

    # Predictions
    predictions = np.argmax(logits, axis=1)

    # Correctness
    correctness = (predictions == targets).astype(int)

    return confidences, correctness


def compute_probs_from_logits(logits: torch.Tensor) -> np.ndarray:
    """
    Compute softmax probabilities from logits.

    Args:
        logits: Model logits [N, C]

    Returns:
        Probability predictions [N, C] as numpy array
    """
    logits = logits.detach().cpu().numpy()
    probs = torch.nn.functional.softmax(
        torch.tensor(logits), dim=1
    ).numpy()
    return probs
