How to read the table:

- **AUROC (area under the receiver operating characteristic curve):** the chance that a clip showing the behaviour scores higher than one that does not. 0.5 is guessing and 1.0 is perfect. It needs no threshold.
- **Macro-F1:** the F1 score balances missed detections against false alarms, from 0 to 1. "Macro" averages it over the activity classes, so rare classes count as much as common ones.
- **MAE (mean absolute error):** the average error on the output's 0 to 1 scale. 0 is perfect.
- **Baseline:** the score of a constant guess from the training labels, in brackets. For AUROC it is always 0.50.
- **Test people:** the people whose clips were scored. Where an output is scored on several datasets, its score is the mean of the per-dataset scores, and their test people are added up. A group is a dataset's test side that does not identify each person.

No confidence intervals are shown: Bulwark's evaluation gives them only with at least 10 test people. Head pose, mouth open and head nod are not scored, because no training dataset gives ground truth for them.
