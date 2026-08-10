# LendingClub Credit Risk — Survival, Causal & Policy Optimization

Credit risk analysis on **2,250,076 loans** from LendingClub (2007–2018), combining Competing Risks Survival Analysis, Predictive Modeling, and Causal Inference to answer a business question: **is the current income/source verification policy wasting money, and how should it be applied selectively to optimize cost?**

**Live Dashboard:** [https://lendingclub-credit-risk-kkujezq9pbydbufk9frqbw.streamlit.app/](https://lendingclub-credit-risk-kkujezq9pbydbufk9frqbw.streamlit.app/)

---

## 1. Business Problem

LendingClub issues loans based on self-reported credit profiles, with 3 verification levels: `Verified`, `Source Verified`, `Not Verified`. Verification costs operational money (assumed $20/loan in this analysis) but is theoretically expected to reduce default risk. This analysis tests that with real data: **does verification actually reduce risk, or is it merely a reaction to already-elevated risk (reverse causation)?**

Three linked layers of analysis:
1. **Survival Analysis** — how long a loan "survives" before default or early payoff (Competing Risks)
2. **Predictive Modeling** — predicting default probability
3. **Causal Inference (X-learner)** — isolating the true causal effect of verification from mere correlation → used to optimize policy

---

## 2. Data

- Source: [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — `accepted_2007_to_2018Q4.csv.gz` (392.6MB, 151 columns)
- After cleaning: **2,250,076 loans** (10,592 rows dropped due to date-integrity errors, 0.47%)
- Outcome distribution: Fully Paid 1,068,731 · Charged Off 268,880 · Censored (still active) 912,465

| event_type | Meaning |
|---|---|
| 0 | Censored — Current/Late/Grace Period |
| 1 | Charged Off (default) — primary risk of interest |
| 2 | Fully Paid — competing risk |

---

## 3. Key Findings — Business Insights

### 3.1 Using the wrong method distorts the risk estimate by nearly 40%

A standard Kaplan-Meier estimate (treating Fully Paid as censored — theoretically incorrect) puts default probability at month 36 at **20.88%**. The correct Aalen-Johansen (Competing Risks) estimate shows the real figure is **15.48%** — a gap of **5.40 percentage points**, meaning the wrong method overstates risk by nearly **35%**. This is concrete evidence of why choosing the correct statistical method matters more than simply "getting a model to run."

### 3.2 Default risk rises steeply with Grade — but non-linearly

| Grade | Default Risk (36 months) | Count |
|---|---|---|
| A | 5.02% | 431,473 |
| B | 10.63% | 660,649 |
| C | 17.05% | 646,907 |
| D | 23.75% | 322,643 |
| E | 30.29% | 134,839 |
| F | 36.84% | 41,530 |
| G | 41.54% | 12,035 |

Grade is the strongest risk-separating factor in the entire pipeline (confirmed by Cox PH: HR Grade G = 5.54 vs. Grade A, p<0.005).

### 3.3 Most important finding: Verification correlates with HIGHER risk, not lower

- Cox PH (Concordance 0.6823): `verification_status_Verified` has HR=1.20, `Source Verified` HR=1.16 — both **increase** default hazard (p<0.005)
- Raw comparison (unadjusted): the verified group has a **22.39%** default rate vs. **14.82%** for the unverified group — a **+7.57 percentage point** gap in the direction of "verification is worse"

→ This is **not** evidence that verification causes harm — it is a textbook sign of **reverse causation**: LendingClub tends to require verification more often for loans that *already* show risk signals (unusually high reported income, ambiguous profiles, etc.), not the other way around. This is precisely why Causal Inference is required instead of reading raw numbers at face value.

### 3.4 Causal Inference (X-learner): separating true causation from correlation

- Propensity overlap is acceptable (treatment mean 0.7255 vs. control 0.6298, good overlapping range) → sufficient to estimate reliable CATE
- **ATE across the full dataset: -1.45 percentage points** (negative = verification does not reduce risk on average, contrary to initial expectation) — after adjusting for confounding, the true causal effect is **much smaller** than the raw +7.57 point gap → most of the raw difference is driven by reverse causation, not by verification causing harm
- Only **18.44%** of loans have a positive CATE (verification genuinely helps reduce risk for this subgroup)

**Average CATE by Grade — the core insight for policy:**

| Grade | Average CATE | Interpretation |
|---|---|---|
| A | -0.46% | Verification has essentially no effect |
| B | -0.93% | No effect |
| C | -2.05% | No effect |
| D | -2.66% | No effect |
| E | -1.45% | No effect |
| F | -1.30% | No effect |
| **G** | **+2.27%** | **Verification genuinely reduces risk** |

→ Verification only has a real causal effect for the highest-risk group (Grade G). For Grades A-F, blanket verification is **wasted operational cost with no risk improvement**.

**CATE by DTI (Debt-to-Income) quartile — checking for another segmentation factor:**

| DTI Quartile | Average CATE |
|---|---|
| Q1 (lowest) | -1.37% |
| Q2 | -1.35% |
| Q3 | -1.42% |
| Q4 (highest) | -1.65% |

The spread across quartiles is very small (~0.3 percentage points) — DTI is **not an effective segmentation factor** for verification, unlike Grade (which spans -2.66% to +2.27%). Operational takeaway: a selective policy only needs to key off Grade, no need for an additional DTI condition, which keeps the rule simpler to implement.

### 3.5 LGD (Loss Given Default) — fairly consistent, Grade G is heaviest

| Grade | Average LGD |
|---|---|
| A | 40.35% |
| B | 39.58% |
| C | 40.71% |
| D | 40.88% |
| E | 40.16% |
| F | 40.48% |
| **G** | **43.46%** |

### 3.6 Policy optimization conclusion — concrete financial numbers

On a sample of 300,000 loans:

| Policy | Cost | Net Value |
|---|---|---|
| **Blanket** verification (current) | $6,000,000 | **-$20,044,319** (net loss) |
| **Selective** verification (only when Expected Value > 0) | $917,640 (**84.7% reduction**) | **+$9,384,462** |

**Improvement: +$29,428,781** on the 300K-loan sample — only **15.29%** of loans (45,882 loans) are genuinely worth verifying, concentrated heavily in the higher-risk grades:

| Grade | % of loans worth verifying |
|---|---|
| G | 61.4% |
| F | 33.5% |
| E | 26.1% |
| A | 15.3% |
| D | 14.1% |
| B | 13.4% |
| C | 12.5% |

**Sensitivity Analysis** — the conclusion holds even when confidence in the CATE estimate is sharply discounted:

| CATE Confidence Level | Loans Worth Verifying | % of Total | Net Value |
|---|---|---|---|
| 100% | 45,882 | 15.3% | $9,384,462 |
| 75% | 43,881 | 14.6% | $6,814,006 |
| 50% | 40,549 | 13.5% | $4,261,631 |
| 25% | 32,573 | 10.9% | $1,766,499 |
| 10% | 17,727 | 5.9% | $416,484 |

Even trusting the CATE estimate at only **10%**, the selective policy still nets positive — this conclusion is **highly robust**, not a fluke of the model.

**Simplest implementation scenario (easiest to explain to non-technical stakeholders):** verify only Grade F & G loans — 3,662 loans, $73,240 in cost, net value **+$1,463,976**. No CATE model required — a single simple Grade-based rule is already profitable.

### 3.7 Real limitations of the predictive model — should not be fully automated

The XGBoost model catches **67% of loans that actually default** (recall) but is only **32% accurate on loans it flags as risky** (precision) — for every 3 loans flagged as high-risk, only 1 actually defaults.

**Operational implication:** this model should **not** be used to **automatically reject** loans — it would wrongly reject a large number of good customers (high false-positive rate). It should instead serve as a **triage/prioritization layer**, surfacing high-risk-scored loans for manual underwriter review, combined with the Grade-based selective rule from section 3.6 to decide on verification.

### 3.8 Other notable underwriting factors (from Cox PH coefficients)

Beyond Grade, several other variables show meaningful correlational-causal effects on default risk and repayment speed, useful for refining underwriting/pricing policy:

| Variable | Hazard Ratio (Charged Off) | Business Meaning |
|---|---|---|
| `home_ownership_MORTGAGE` | 0.92 (lower risk, p<0.005) | Borrowers with a mortgage show more financial stability — could be considered for preferential rates |
| `application_type_Joint App` | 0.78 (substantially lower risk, p<0.005) | Joint applications (2 co-borrowers) are notably safer — worth encouraging Joint applications where eligible |
| `purpose_small_business` | 1.19 (higher risk, p<0.005) | Small-business loans carry clearly elevated risk — consider distinct rate/terms for this segment |
| `emp_length_Missing` | 1.15 (higher risk, p<0.005) | A missing employment-length field is itself a risk signal — consider requiring this info before approval |

**On early payoff speed (`Fully Paid` model):** `term_60_months` has HR=0.32 (very low) — 60-month loans are **far less likely to be paid off early** than 36-month loans. For LendingClub, this is good news for interest cash flow — longer-term loans generate a more stable, predictable interest income stream with lower prepayment risk.

### Business Recommendation

**Shift from blanket verification to selective verification, prioritizing Grades F/G.** The current policy (verifying nearly 70% of loans) is running a net loss of ~$20M per 300K loans because it applies verification indiscriminately to lower-risk grades (A-C), where verification has no real causal effect. Simply restricting verification to Grade F-G (simplest option) or deploying the full CATE model (optimal option, $9.4M net value) both deliver significant financial improvement without meaningful additional technology cost. The default-prediction model should serve as a prioritization aid, not a replacement for manual decisions, given its still-low precision (32%).

---

## 4. Technical Methodology

- **Competing Risks**: Kaplan-Meier (incorrect baseline) vs. Aalen-Johansen CIF (correct), Cause-Specific Cox PH for both outcomes (`int_rate` dropped due to multicollinearity with `grade`)
- **Predictive**: XGBoost trained on 1,337,611 "mature" loans (20.10% default rate), AUC 0.7166, PR-AUC 0.3877; top features: `int_rate` (23.8%), `term_60_months` (8.8%), `grade_B` (6.5%)
- **Causal**: Treatment = `verification_status != 'Not Verified'` (69.79% of loans), X-learner (2 outcome models + 2 tau models weighted by propensity), overlap checked before estimation
- **Optimization**: LGD computed from real Charged-Off loan data, Expected Value = CATE × Potential Loss − Verification Cost ($20/loan)

---

## 5. Project Structure

```text
├── 01_data_prep_survival.ipynb      # Load, clean, Competing Risks, Cox PH
├── 02_predictive_causal.ipynb       # XGBoost, X-Learner CATE, Heterogeneity
├── 03_optimization.ipynb            # LGD, Policy Optimization, Sensitivity
├── data/
│   ├── lendingclub_with_cate.csv
│   ├── lendingclub_final_optimization.csv
│   ├── dashboard_data/
│   ├── lgd_by_grade.csv
│   ├── model_features.csv
│   └── xgb_default_model.json
└── app.py
```
## 6. Reproducing

```bash
pip install pandas numpy lifelines xgboost scikit-learn matplotlib streamlit
```

Raw data (392.6MB) can be downloaded from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club); it is not included in this repo due to GitHub's file size limits.

## 7. Limitations

- Cox and XGBoost models were trained on a 200K–300K sample (not the full 2.25M) due to local RAM constraints — results are stable, but this should be kept in mind when interpreting them
- Treatment is not randomized — despite propensity adjustment, there remains a risk of unobserved confounding (e.g., the specific reasons a loan gets flagged for verification are not present in the data)
- Cox concordance for the Fully Paid model is only 0.5942 — notably weaker than the Charged Off model (0.6823), so hazard ratios for this outcome should be interpreted more cautiously
- XGBoost precision is only 32% — not suitable for fully automating loan-rejection decisions
- The $20/loan verification cost is an assumption, not confirmed against LendingClub's actual operational figures

