from inspect_evals.agentic_misalignment.classifiers.blackmail_classifier import (
    BlackmailClassifier,
)
from inspect_evals.agentic_misalignment.classifiers.classifier import Classifier
from inspect_evals.agentic_misalignment.classifiers.deliberation_classifier import (
    DeliberationClassifier,
)
from inspect_evals.agentic_misalignment.classifiers.leak_classifier import (
    LeakClassifier,
)
from inspect_evals.agentic_misalignment.classifiers.murder_classifier import (
    MurderClassifier,
)

__all__ = [
    "Classifier",
    "BlackmailClassifier",
    "MurderClassifier",
    "LeakClassifier",
    "DeliberationClassifier",
    "CLASSIFIER_VERSION",
]

# Bump when classifier prompts, gates, or verdict logic change so runs remain
# comparable.
CLASSIFIER_VERSION = "1.0.0"
