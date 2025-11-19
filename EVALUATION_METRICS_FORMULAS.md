# 📐 Complete Mathematical Formulas for All Evaluation Metrics

**Comprehensive Guide for Research Paper**

---

## 📊 Table of Contents

1. [Basic Definitions](#1-basic-definitions)
2. [Confusion Matrix Elements](#2-confusion-matrix-elements)
3. [Standard Classification Metrics](#3-standard-classification-metrics)
4. [Averaging Methods](#4-averaging-methods)
5. [Advanced Metrics](#5-advanced-metrics)
6. [ROC and AUC](#6-roc-and-auc)
7. [Statistical Significance Tests](#7-statistical-significance-tests)
8. [Confidence Intervals](#8-confidence-intervals)
9. [Complete Example Calculation](#9-complete-example-calculation)

---

## 1. Basic Definitions

### Binary Classification Terms

For a **single class** in multi-class classification:

- **TP (True Positive)**: Correctly predicted as positive
- **TN (True Negative)**: Correctly predicted as negative  
- **FP (False Positive)**: Incorrectly predicted as positive (Type I Error)
- **FN (False Negative)**: Incorrectly predicted as negative (Type II Error)

### Total Predictions

$$
\text{Total} = TP + TN + FP + FN
$$

---

## 2. Confusion Matrix Elements

### For Multi-Class Classification

Given a confusion matrix $C$ where $C_{i,j}$ represents the number of samples with true label $i$ predicted as label $j$:

**For class $i$:**

$$
TP_i = C_{i,i}
$$

$$
FN_i = \sum_{j \neq i} C_{i,j}
$$

$$
FP_i = \sum_{j \neq i} C_{j,i}
$$

$$
TN_i = \sum_{j \neq i} \sum_{k \neq i} C_{j,k}
$$

**Alternative notation:**

$$
TP_i = C_{i,i}
$$

$$
FN_i = \text{row}_i - TP_i
$$

$$
FP_i = \text{col}_i - TP_i
$$

$$
TN_i = \text{Total} - TP_i - FN_i - FP_i
$$

---

## 3. Standard Classification Metrics

### 3.1 Accuracy

**Definition**: Overall correctness of the model

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}
$$

**Multi-class:**

$$
\text{Accuracy} = \frac{\sum_{i=1}^{K} C_{i,i}}{\sum_{i=1}^{K} \sum_{j=1}^{K} C_{i,j}}
$$

where $K$ is the number of classes.

**Range**: [0, 1], where 1 is perfect

---

### 3.2 Precision (Positive Predictive Value)

**Definition**: Of all positive predictions, how many are actually correct?

$$
\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{Total Predicted Positive}}
$$

**Interpretation**: 
- High precision = Few false positives
- Answers: "When the model predicts positive, how often is it right?"

**Range**: [0, 1], where 1 means no false positives

---

### 3.3 Recall (Sensitivity, True Positive Rate, Hit Rate)

**Definition**: Of all actual positives, how many did we correctly identify?

$$
\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{Total Actual Positive}}
$$

**Interpretation**:
- High recall = Few false negatives
- Answers: "Of all positive cases, how many did we find?"

**Range**: [0, 1], where 1 means no false negatives

---

### 3.4 F1-Score (F-Measure)

**Definition**: Harmonic mean of Precision and Recall

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times TP}{2 \times TP + FP + FN}
$$

**Alternative form:**

$$
F_1 = \frac{2}{\frac{1}{\text{Precision}} + \frac{1}{\text{Recall}}}
$$

**Why harmonic mean?** Punishes extreme values (e.g., high precision but low recall)

**Range**: [0, 1], where 1 is perfect

---

### 3.5 F-Beta Score (Generalized)

**Definition**: Weighted harmonic mean, where $\beta$ controls importance of recall

$$
F_\beta = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{\beta^2 \times \text{Precision} + \text{Recall}}
$$

**Special cases:**
- $\beta = 1$: F1-Score (equal weight)
- $\beta = 2$: F2-Score (recall weighted 2x more)
- $\beta = 0.5$: F0.5-Score (precision weighted 2x more)

---

### 3.6 Specificity (True Negative Rate)

**Definition**: Of all actual negatives, how many did we correctly identify?

$$
\text{Specificity} = \frac{TN}{TN + FP} = \frac{\text{True Negatives}}{\text{Total Actual Negative}}
$$

**Range**: [0, 1]

---

### 3.7 False Positive Rate (Fall-out)

**Definition**: Of all actual negatives, how many did we incorrectly classify as positive?

$$
\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}
$$

**Range**: [0, 1], where 0 is ideal

---

### 3.8 False Negative Rate (Miss Rate)

**Definition**: Of all actual positives, how many did we miss?

$$
\text{FNR} = \frac{FN}{FN + TP} = 1 - \text{Recall}
$$

**Range**: [0, 1], where 0 is ideal

---

### 3.9 Balanced Accuracy

**Definition**: Average of sensitivity and specificity

$$
\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{\text{Recall} + \text{TNR}}{2}
$$

**Use case**: When classes are imbalanced

---

## 4. Averaging Methods (Multi-Class)

For multi-class classification with $K$ classes:

### 4.1 Macro Average

**Definition**: Average of per-class metrics (treats all classes equally)

$$
\text{Metric}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{Metric}_i
$$

**Example (Precision):**

$$
\text{Precision}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \frac{TP_i}{TP_i + FP_i}
$$

**Characteristics**:
- Treats all classes equally (regardless of size)
- Good for balanced datasets
- Can be influenced by poor performance on small classes

---

### 4.2 Micro Average

**Definition**: Aggregate counts across all classes, then calculate metric

$$
\text{Precision}_{\text{micro}} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FP_i)}
$$

$$
\text{Recall}_{\text{micro}} = \frac{\sum_{i=1}^{K} TP_i}{\sum_{i=1}^{K} (TP_i + FN_i)}
$$

**Note**: For multi-class classification:

$$
\text{Precision}_{\text{micro}} = \text{Recall}_{\text{micro}} = \text{F1}_{\text{micro}} = \text{Accuracy}
$$

**Characteristics**:
- Favors larger classes
- Good for imbalanced datasets
- Equivalent to accuracy for multi-class

---

### 4.3 Weighted Average

**Definition**: Weighted average by class support (number of true instances)

$$
\text{Metric}_{\text{weighted}} = \frac{1}{N} \sum_{i=1}^{K} n_i \times \text{Metric}_i
$$

where:
- $n_i$ = number of true instances of class $i$
- $N$ = total number of samples = $\sum_{i=1}^{K} n_i$

**Example (Precision):**

$$
\text{Precision}_{\text{weighted}} = \frac{1}{N} \sum_{i=1}^{K} n_i \times \frac{TP_i}{TP_i + FP_i}
$$

**Characteristics**:
- Accounts for class imbalance
- Larger classes have more influence
- Most commonly reported in papers

---

## 5. Advanced Metrics

### 5.1 Cohen's Kappa (κ)

**Definition**: Agreement between predictions and true labels, adjusted for chance

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where:
- $p_o$ = observed agreement (accuracy)
- $p_e$ = expected agreement by chance

**Detailed calculation:**

$$
p_o = \frac{1}{N} \sum_{i=1}^{K} C_{i,i}
$$

$$
p_e = \frac{1}{N^2} \sum_{i=1}^{K} \left( \sum_{j=1}^{K} C_{i,j} \right) \times \left( \sum_{j=1}^{K} C_{j,i} \right)
$$

**Alternative form:**

$$
\kappa = \frac{N \sum_{i=1}^{K} C_{i,i} - \sum_{i=1}^{K} \left( \sum_{j=1}^{K} C_{i,j} \right) \times \left( \sum_{j=1}^{K} C_{j,i} \right)}{N^2 - \sum_{i=1}^{K} \left( \sum_{j=1}^{K} C_{i,j} \right) \times \left( \sum_{j=1}^{K} C_{j,i} \right)}
$$

**Interpretation**:
- $\kappa < 0$: Worse than random
- $\kappa = 0$: Agreement by chance
- $0 < \kappa < 0.20$: Slight agreement
- $0.20 \leq \kappa < 0.40$: Fair agreement
- $0.40 \leq \kappa < 0.60$: Moderate agreement
- $0.60 \leq \kappa < 0.80$: Substantial agreement
- $0.80 \leq \kappa < 1.00$: Almost perfect agreement
- $\kappa = 1$: Perfect agreement

**Range**: [-1, 1]

---

### 5.2 Matthews Correlation Coefficient (MCC)

**Definition**: Correlation between observed and predicted classifications

**Binary case:**

$$
\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

**Multi-class case (generalized):**

$$
\text{MCC} = \frac{\sum_{k} \sum_{l} \sum_{m} C_{kk} C_{lm} - C_{kl} C_{mk}}{\sqrt{\sum_{k} \left( \sum_{l} C_{kl} \right) \left( \sum_{l'} \sum_{l'' \neq k} C_{l'l''} \right)} \times \sqrt{\sum_{k} \left( \sum_{l} C_{lk} \right) \left( \sum_{l'} \sum_{l'' \neq k} C_{l''l'} \right)}}
$$

**Simplified multi-class (using covariance):**

$$
\text{MCC} = \frac{\text{cov}(y_{\text{true}}, y_{\text{pred}})}{\sqrt{\text{var}(y_{\text{true}}) \times \text{var}(y_{\text{pred}})}}
$$

**Interpretation**:
- MCC = -1: Perfect disagreement
- MCC = 0: Random prediction
- MCC = +1: Perfect prediction
- Considered the most balanced metric, especially for imbalanced datasets

**Range**: [-1, 1]

---

### 5.3 Jaccard Index (Intersection over Union)

**Definition**: Size of intersection divided by size of union

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}
$$

**Relationship to F1:**

$$
J = \frac{F_1}{2 - F_1}
$$

**Range**: [0, 1]

---

### 5.4 Error Rate

**Definition**: Proportion of incorrect predictions

$$
\text{Error Rate} = 1 - \text{Accuracy} = \frac{FP + FN}{TP + TN + FP + FN}
$$

**Range**: [0, 1], where 0 is ideal

---

## 6. ROC and AUC

### 6.1 ROC Curve (Receiver Operating Characteristic)

**Definition**: Plot of True Positive Rate vs False Positive Rate at various thresholds

**Axes:**
- **X-axis**: False Positive Rate (FPR)
  $$
  \text{FPR} = \frac{FP}{FP + TN}
  $$

- **Y-axis**: True Positive Rate (TPR) = Recall
  $$
  \text{TPR} = \frac{TP}{TP + FN}
  $$

**For threshold $t$:**

$$
\text{ROC}(t) = \left( \text{FPR}(t), \text{TPR}(t) \right)
$$

---

### 6.2 AUC (Area Under the ROC Curve)

**Definition**: Area under the ROC curve

**Numerical integration:**

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(x)) \, dx
$$

**Trapezoidal approximation:**

$$
\text{AUC} \approx \sum_{i=1}^{n-1} \frac{1}{2} (\text{FPR}_{i+1} - \text{FPR}_i)(\text{TPR}_{i+1} + \text{TPR}_i)
$$

**Probabilistic interpretation:**

$$
\text{AUC} = P(\text{score}(\text{positive}) > \text{score}(\text{negative}))
$$

**Interpretation**:
- AUC = 0.5: Random classifier (diagonal line)
- 0.5 < AUC < 0.7: Poor
- 0.7 ≤ AUC < 0.8: Acceptable
- 0.8 ≤ AUC < 0.9: Excellent
- AUC ≥ 0.9: Outstanding
- AUC = 1.0: Perfect classifier

**Range**: [0, 1]

---

### 6.3 Multi-Class AUC

**One-vs-Rest (OvR) Macro:**

$$
\text{AUC}_{\text{macro}} = \frac{1}{K} \sum_{i=1}^{K} \text{AUC}_i
$$

where $\text{AUC}_i$ is calculated treating class $i$ as positive and all others as negative.

**One-vs-Rest Weighted:**

$$
\text{AUC}_{\text{weighted}} = \frac{1}{N} \sum_{i=1}^{K} n_i \times \text{AUC}_i
$$

---

### 6.4 Precision-Recall AUC

**For imbalanced datasets**, use PR-AUC:

$$
\text{PR-AUC} = \int_0^1 \text{Precision}(\text{Recall}^{-1}(x)) \, dx
$$

---

## 7. Statistical Significance Tests

### 7.1 McNemar's Test

**Purpose**: Test if two models have significantly different error rates on the same test set

**Contingency Table:**

|                    | Model B Correct | Model B Wrong |
|--------------------|-----------------|---------------|
| **Model A Correct** | $n_{00}$       | $n_{01}$      |
| **Model A Wrong**   | $n_{10}$       | $n_{11}$      |

**Test Statistic:**

$$
\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}
$$

where:
- $n_{01}$ = A correct, B wrong
- $n_{10}$ = A wrong, B correct
- The $-1$ is **continuity correction** (Yates's correction)

**Without continuity correction:**

$$
\chi^2 = \frac{(n_{01} - n_{10})^2}{n_{01} + n_{10}}
$$

**Distribution**: $\chi^2$ with 1 degree of freedom

**Null Hypothesis**: Models have same error rate ($n_{01} = n_{10}$)

**Decision Rule**: 
- If $p < 0.05$: Reject null → Models are significantly different
- If $p \geq 0.05$: Fail to reject → No significant difference

---

### 7.2 Paired t-test

**Purpose**: Compare means of two related groups (e.g., cross-validation folds)

**Test Statistic:**

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}}
$$

where:
- $\bar{d}$ = mean of differences
- $s_d$ = standard deviation of differences
- $n$ = number of pairs

**Detailed:**

$$
\bar{d} = \frac{1}{n} \sum_{i=1}^{n} (x_i - y_i)
$$

$$
s_d = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (d_i - \bar{d})^2}
$$

**Distribution**: t-distribution with $n-1$ degrees of freedom

---

### 7.3 5x2 Cross-Validated Paired t-test

**Purpose**: More robust version for comparing models

**Test Statistic:**

$$
t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5} \sum_{j=1}^{5} s_j^2}}
$$

where:
- $p_i^{(j)}$ = difference in error rate between models in fold $i$ of repetition $j$
- $s_j^2$ = variance of differences in repetition $j$

---

## 8. Confidence Intervals

### 8.1 Normal Approximation (for Accuracy)

**95% Confidence Interval:**

$$
\text{CI} = \hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

where:
- $\hat{p}$ = observed accuracy
- $z_{\alpha/2}$ = 1.96 for 95% CI
- $n$ = sample size

---

### 8.2 Wilson Score Interval (Better for small samples)

**95% CI:**

$$
\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

---

### 8.3 Bootstrap Confidence Interval

**Procedure:**
1. Resample test set with replacement $B$ times (e.g., $B = 1000$)
2. Calculate metric for each bootstrap sample
3. Sort bootstrap metrics: $\theta_1, \theta_2, \ldots, \theta_B$
4. CI = $[\theta_{(\alpha/2)B}, \theta_{(1-\alpha/2)B}]$

**For 95% CI with $B = 1000$:**

$$
\text{CI} = [\theta_{25}, \theta_{975}]
$$

---

## 9. Complete Example Calculation

### Example: 3-Class Classification

**Confusion Matrix:**

|             | Pred A | Pred B | Pred C |
|-------------|--------|--------|--------|
| **True A**  | 45     | 3      | 2      |
| **True B**  | 4      | 38     | 3      |
| **True C**  | 1      | 2      | 52     |

**Total samples**: $N = 150$

---

### Step 1: Calculate per-class metrics

**Class A:**
- $TP_A = 45$
- $FN_A = 3 + 2 = 5$
- $FP_A = 4 + 1 = 5$
- $TN_A = 38 + 3 + 2 + 52 = 95$

$$
\text{Precision}_A = \frac{45}{45 + 5} = \frac{45}{50} = 0.900
$$

$$
\text{Recall}_A = \frac{45}{45 + 5} = \frac{45}{50} = 0.900
$$

$$
F1_A = \frac{2 \times 0.900 \times 0.900}{0.900 + 0.900} = 0.900
$$

**Class B:**
- $TP_B = 38$
- $FN_B = 4 + 3 = 7$
- $FP_B = 3 + 2 = 5$
- $TN_B = 45 + 2 + 1 + 52 = 100$

$$
\text{Precision}_B = \frac{38}{38 + 5} = \frac{38}{43} = 0.884
$$

$$
\text{Recall}_B = \frac{38}{38 + 7} = \frac{38}{45} = 0.844
$$

$$
F1_B = \frac{2 \times 0.884 \times 0.844}{0.884 + 0.844} = 0.864
$$

**Class C:**
- $TP_C = 52$
- $FN_C = 1 + 2 = 3$
- $FP_C = 2 + 3 = 5$
- $TN_C = 45 + 3 + 4 + 38 = 90$

$$
\text{Precision}_C = \frac{52}{52 + 5} = \frac{52}{57} = 0.912
$$

$$
\text{Recall}_C = \frac{52}{52 + 3} = \frac{52}{55} = 0.945
$$

$$
F1_C = \frac{2 \times 0.912 \times 0.945}{0.912 + 0.945} = 0.928
$$

---

### Step 2: Calculate macro averages

$$
\text{Precision}_{\text{macro}} = \frac{0.900 + 0.884 + 0.912}{3} = 0.899
$$

$$
\text{Recall}_{\text{macro}} = \frac{0.900 + 0.844 + 0.945}{3} = 0.896
$$

$$
F1_{\text{macro}} = \frac{0.900 + 0.864 + 0.928}{3} = 0.897
$$

---

### Step 3: Calculate overall accuracy

$$
\text{Accuracy} = \frac{45 + 38 + 52}{150} = \frac{135}{150} = 0.900
$$

---

### Step 4: Calculate Cohen's Kappa

**Observed agreement:**

$$
p_o = \frac{135}{150} = 0.900
$$

**Expected agreement:**

$$
p_e = \frac{1}{150^2} [(50 \times 50) + (45 \times 43) + (55 \times 57)]
$$

$$
p_e = \frac{2500 + 1935 + 3135}{22500} = \frac{7570}{22500} = 0.3364
$$

$$
\kappa = \frac{0.900 - 0.3364}{1 - 0.3364} = \frac{0.5636}{0.6636} = 0.849
$$

---

### Step 5: Calculate MCC (simplified)

**For multi-class, use correlation:**

$$
\text{MCC} = \text{Pearson correlation}(y_{\text{true}}, y_{\text{pred}})
$$

For this example: $\text{MCC} \approx 0.845$

---

## 📖 References

### Standard Metrics
1. Powers, D. M. (2011). "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation"
2. Sokolova, M., & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks"

### Advanced Metrics
3. Cohen, J. (1960). "A coefficient of agreement for nominal scales"
4. Matthews, B. W. (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme"
5. Chicco, D., & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy"

### Statistical Tests
6. McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions"
7. Dietterich, T. G. (1998). "Approximate statistical tests for comparing supervised classification learning algorithms"

### Multi-Class Metrics
8. Grandini, M., Bagli, E., & Visani, G. (2020). "Metrics for multi-class classification: an overview"
9. Hand, D. J., & Till, R. J. (2001). "A simple generalisation of the area under the ROC curve for multiple class classification problems"

---

## 💡 Quick Reference Table

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|----------------|
| Accuracy | $\frac{TP+TN}{Total}$ | [0,1] | Overall correctness |
| Precision | $\frac{TP}{TP+FP}$ | [0,1] | Positive prediction accuracy |
| Recall | $\frac{TP}{TP+FN}$ | [0,1] | Positive detection rate |
| F1-Score | $\frac{2TP}{2TP+FP+FN}$ | [0,1] | Harmonic mean of P&R |
| Specificity | $\frac{TN}{TN+FP}$ | [0,1] | Negative detection rate |
| Cohen's κ | $\frac{p_o-p_e}{1-p_e}$ | [-1,1] | Agreement beyond chance |
| MCC | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | [-1,1] | Balanced correlation |
| ROC-AUC | Area under ROC | [0,1] | Discrimination ability |

---

## 🎯 Which Metrics to Report in Your Paper?

### Must Report (Main Results Table):
1. ✅ **Accuracy**
2. ✅ **Precision (Macro)**
3. ✅ **Recall (Macro)**
4. ✅ **F1-Score (Macro & Weighted)**

### Should Report (Advanced Table):
5. ✅ **Cohen's Kappa**
6. ✅ **Matthews Correlation Coefficient (MCC)**
7. ✅ **ROC-AUC (if probabilities available)**

### Optional (Discussion/Appendix):
8. Per-class metrics
9. Confusion matrix statistics (TPR, FPR)
10. Statistical significance tests (McNemar)
11. Confidence intervals

---

**Document Version**: 1.0  
**Last Updated**: November 19, 2025  
**For**: Darknet Product Classification Project
