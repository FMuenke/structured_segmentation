import os
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve


def plot_calibration_curve(est_group, x_test, y_test, path, pos_label=0):
    fig_index = 1
    """Plot calibration curve for est w/o and with calibration. """

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((5, 1), (2, 0))
    ax3 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in est_group:
        y_pred = clf.predict(x_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(x_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(x_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = f1_score(y_test, y_pred, average="macro")

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
        fpr, tpr, thresholds = roc_curve(y_test, prob_pos)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)
        ax3.plot(fpr, tpr, "s-", label="%s (%1.3f)" % (name, clf_score))

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    ax3.set_ylabel("TP Rate")
    ax3.set_xlabel("FP Rate")
    ax3.legend(loc="lower right")
    ax3.set_title('ROC-Curve')

    plt.tight_layout()
    fig.savefig(os.path.join(path, "calibration_results.png"))
    plt.close(fig)
