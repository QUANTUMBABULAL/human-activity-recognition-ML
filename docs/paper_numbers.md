# Paper numbers (task P1-0)

Source: Athens, Blum, Singh, "Human Activity Classification", CS229 report (2018), and the
poster. **Figure 2 of the report is an image; the "Figure 2" columns below were read off the
rendered figure by hand. P2 must verify every number against the PDF in PR #1.** Write
"~0.97" style values if a figure value cannot be read exactly.

Accuracy in %. "limited" = our `reduced` (hand IMU + HR); "full" = three IMUs + HR.

| Model | Set | Fig. 2 train | Fig. 2 test | Report-text test | Poster train | Poster test | Note |
|---|---|---|---|---|---|---|---|
| Logistic Regression | reduced | 64.2 | 63.9 | 63.9 | 64.26 | 64.25 | text = figure |
| Logistic Regression | full | 81.7 | 81.5 | 81.5 | 82.06 | 82.06 | text = figure |
| SVM (RBF) | reduced | 98.7 | 95.0 | 95.0 | 94.04 | 91.94 | poster differs |
| SVM (RBF) | full | 99.9 | 98.9 | 98.9 | 99.83 | 99.09 | |
| Decision Tree | reduced | 97.8 | 87.3 | 87.3 | 99.65 | 87.32 | |
| Decision Tree | full | 98.7 | 92.7 | 92.7 | 99.77 | 91.93 | |
| AdaBoost | reduced | 99.9 | 94.0 | 94.0 | 99.93 | 93.43 | |
| AdaBoost | full | 99.97 | 98.5 | 98.5 | 100.00 | 98.71 | |
| Random Forest | reduced | 99.9 | 93.7 | 93.7 | 100.00 | 93.32 | |
| Random Forest | full | 99.97 | 98.0 | 98.0 | 100.00 | 97.82 | |
| MLP | reduced | 86.1 | **84.0** | **81.4** | 84.78 | 81.4 | **figure vs text disagree** |
| MLP | full | 99.8 | 98.1 | 98.1 (best run) | 93.33 | 95.63 | poster differs (text: "consistently > 95%") |

Notes
- The report is the primary reference. The poster numbers differ in several cells; they look
  like a different/earlier run. Use the poster column only for context, never as the target.
- MLP reduced: Figure 2 appears to show 84.0 test but the text and poster say 81.4.
  Report both ("figure vs. text") and use the text value (81.4) as the target.

## Hyperparameters stated in the report

| Model | Settings |
|---|---|
| Logistic Regression | L2, SAG solver, C ~ 0.01 for both sets (5-fold CV) |
| SVM | RBF beat linear (~15 pts) and poly (~10 pts) on validation accuracy; C = 100 (full), C = 1000 (reduced) |
| Decision Tree | Gini, max depth 15 (one-standard-deviation rule), both sets |
| AdaBoost | default learning rate; chosen 500 trees / depth 10 (reduced), 250 trees / depth 9 (full) |
| Random Forest | 100 trees, sqrt(features) per split, max depth 20, both sets |
| MLP | 512-512 ReLU, dropout 0.5, softmax, categorical cross-entropy, SGD; 100 epochs shown for reduced set |
| Protocol | pooled subjects, random 85/15 split, 5-fold CV on training rows, accuracy as main metric |
