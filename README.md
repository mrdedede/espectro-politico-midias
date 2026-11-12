# Media Bias Survey: Political Perception of Portuguese-Language News Outlets

This project analyses an online survey in which respondents placed 26 popular Portuguese-language news outlets (Brazilian and Portuguese) on a left / centre / right political spectrum. It asks whether respondents' own political self-identification and demographics (age and region) are associated with how they perceive each outlet, and then derives a single "verdict" for each outlet.

The analysis lives in a single Jupyter notebook, [enquete-interpretation.ipynb](enquete-interpretation.ipynb), which reads the raw survey export [enquete.csv](enquete.csv).

> **Written in English for:** developers and researchers who want to understand, rerun, or critique the analysis. The notebook itself is in Portuguese.

## Contents

- [Data](#data)
- [Pipeline overview](#pipeline-overview)
- [1. Data cleaning](#1-data-cleaning)
- [2. Reclassifying respondents with unclear political affiliation](#2-reclassifying-respondents-with-unclear-political-affiliation)
- [3. Exploratory analysis](#3-exploratory-analysis)
- [4. Outlet verdicts](#4-outlet-verdicts)
- [5. Robustness and confidence intervals](#5-robustness-and-confidence-intervals)
- [Limitations](#limitations)
- [Running the analysis](#running-the-analysis)

## Data

[enquete.csv](enquete.csv) is the raw export of an online survey form (136 rows). Columns:

| Column (original, Portuguese) | Renamed to | Content |
|---|---|---|
| Carimbo de data/hora | `Timestamp` | Submission time |
| Como você se considera politicamente? | `Politica` | Self-declared political position (free text) |
| Qual a sua idade? | `Idade` | Age (free text, often "X anos") |
| Qual região e qual país você mora? | `Regiao` | Region and country (free text) |
| 26 outlet columns | `G1`, `Folha`, `Estadao`, `Oglobo`, `UOL`, `BBC`, `CNN BR`, `EBC`, `Poder360`, `Intercept`, `Metropoles`, `JP`, `Gazeta`, `Valor`, `Exame`, `Bloomberg`, `Apublica`, `DW`, `Publico`, `RTP`, `Expresso`, `Observador`, `CNN PT`, `Correio`, `RR`, `Aominuto` | Each respondent's rating of that outlet: "Esquerda", "Centro", "Direita", or "Não sei" |

The outlets are 17 Brazilian and 9 Portuguese (see the `RENAME` dictionary in the notebook for the full list with URLs).

`enquete.csv` is listed in [.gitignore](.gitignore), so the survey data is not meant to be committed.

## Pipeline overview

```
enquete.csv
   │
   ├─ 1. Clean: normalise region, age, political self-label; drop "Não sei"
   │
   ├─ 2. Reclassify unclear self-labels with a KNN on outlet ratings
   │       (random reference samples, 100 repetitions per configuration)
   │       → manual final choice for 10 respondents
   │
   ├─ 3. Explore: outlet ratings split by political group, age band, and region
   │
   ├─ 4. Verdict per outlet: threshold rule, with mean-of-means fallback
   │
   └─ 5. Robustness: permutation chi-square tests, Wilson CIs, Holm correction
```

## 1. Data cleaning

All text normalisation follows the same pattern: Unicode NFKD decomposition, ASCII encoding (which strips accents), lowercasing, and stripping whitespace. Then each field is mapped to a fixed label set.

- **Region (`Regiao`)**: free text is matched against ordered regular expressions. Brazilian states, cities, and region names become `Brasil (Norte|Nordeste|Centro-Oeste|Sudeste|Sul)`; Portugal and Angola become `Portugal` and `Angola`; other South American, European, and Asian locations become their continent. Anything unmatched becomes `None`. The regexes were drafted with an AI assistant and then used as written.
- **Age (`Idade`)**: the suffix " anos" is removed, then the column is converted to numbers with `errors="coerce"`, so unparseable entries become `NaN`.
- **Political self-label (`Politica`)**: normalised to `Esquerda`, `Centro`, or `Direita`. Labels that map clearly to one side are grouped:
  - `Esquerda`: Esquerda Radical, Comunista, Socialista, Social-Democrata, Marxista, and any other label containing "esquerda" without "centro" or "direita".
  - `Direita`: Conservador, Libertário, Bolsonarista, and any other label containing "direita" without "centro" or "esquerda".
  - `Centro`: Apartidário, Terceira Via, and any other label containing "centro" without "esquerda" or "direita".
  - Everything else (for example, "Centro-esquerda" or free-form answers) becomes `None`. These are the "unclear" respondents handled in step 2.
- **Outlet ratings**: the literal string `"Não sei"` is replaced with `None` across the table.

## 2. Reclassifying respondents with unclear political affiliation

Respondents whose self-label could not be mapped (`Politica` is `None`) still have outlet ratings, which are informative about their position. The notebook uses a KNN classifier to estimate their group from those ratings.

**Encoding.** Each rating is numeric: `Esquerda = -1`, `Centro = 0`, `Direita = 1`. Missing ratings are filled with the column mean, so the feature matrix has no gaps.

**Reference samples.** For each configuration, the algorithm draws `sample_size` random respondents from each of the three clearly labelled groups. Those rows are the training set for `KNeighborsClassifier`.

**Grid.** The notebook sweeps:

- `k` (neighbours) ∈ {5, 7, 9, 11, 13}
- `sample_size` per group ∈ {5, 7, 9, 11, 13}
- 100 random repetitions for each `(k, sample_size)` pair

That is 2,500 KNN fits, each predicting the unclear respondents.

**Stability check.** For each unclear respondent, the notebook computes the share of runs that predicted each class, then plots these as heatmaps per respondent (`dist` and the charts in cell 12). Each chart shows the respondent's original free-text self-label in its title.

**Final decision.** Predictions were not taken automatically from the algorithm. For 10 respondents (row indices 22, 27, 41, 44, 62, 63, 95, 100, 114, 127), the author picked the class that looked most consistent across the grid. The classifier is therefore used as a decision aid, and these 10 assignments are a manual judgement informed by the KNN output.

## 3. Exploratory analysis

Three cuts of the outlet ratings, each shown as one pie chart per outlet:

- **By political group** (`Esquerda`, `Centro`, `Direita`): the distribution of ratings within each group.
- **By age band**: respondents are split into 8 quantile bands of roughly equal size (`pd.qcut` over ranks, so repeated ages are tie-broken). A bar chart shows the number of responses per band and outlet.
- **By region group**: `Nordeste`, `Outras regiões do Brasil`, `Portugal`, and `Outros países`. The notebook notes that most respondents are from the Northeast, so equal-sized groups were not possible.

## 4. Outlet verdicts

Each outlet gets one of three labels, `Esquerda`, `Centro`, or `Direita`, using two steps.

**Step 1: 50% rule.** For each outlet, the notebook checks which rating class is chosen by at least 50% of respondents in each of the three political groups. A class counts as "strong" if it reaches 50% in at least two of the three groups. If exactly one class is strong, it becomes the verdict (origin: "regra 50%").

**Step 2: fallback on mean-of-means.** If the rule is inconclusive, the notebook averages the per-group mean ratings (on the -1 / 0 / 1 scale) and thresholds the result:

- `≤ -1/3` → `Esquerda`
- `< 1/3` → `Centro`
- otherwise → `Direita`

Both the rule-based and fallback outcomes are reported, along with the mean score.

**Results**

| Outlet | Verdict | Origin | Mean score |
|---|---|---|---|
| Portal G1 | Centro | fallback | -0.29 |
| Folha de São Paulo | Centro | fallback | -0.04 |
| Estadão | Direita | 50% rule | 0.27 |
| Jornal O Globo | Centro | fallback | -0.14 |
| UOL | Esquerda | 50% rule | -0.38 |
| BBC News | Centro | 50% rule | -0.18 |
| CNN Brasil | Centro | fallback | -0.00 |
| Agência Brasil (EBC) | Esquerda | 50% rule | -0.59 |
| Poder 360 | Centro | fallback | 0.18 |
| The Intercept | Esquerda | 50% rule | -0.70 |
| Metrópoles | Centro | fallback | 0.04 |
| Jovem Pan News | Direita | 50% rule | 0.79 |
| Gazeta do Povo | Direita | 50% rule | 0.45 |
| Valor Econômico | Direita | 50% rule | 0.32 |
| Revista Exame | Centro | fallback | 0.09 |
| Bloomberg Brasil | Direita | 50% rule | 0.21 |
| Agência Pública | Esquerda | 50% rule | -0.60 |
| Deutsche Welle | Centro | 50% rule | -0.16 |
| Público | Esquerda | 50% rule | -0.80 |
| RTP Notícias | Centro | 50% rule | -0.03 |
| Expresso | Centro | fallback | -0.19 |
| Observador | Centro | 50% rule | -0.17 |
| CNN Portugal | Centro | fallback | 0.03 |
| Correio da Manhã | Centro | fallback | 0.06 |
| Jornal Renascença | Direita | 50% rule | 0.29 |
| Notícias ao Minuto | Centro | 50% rule | -0.18 |

## 5. Robustness and confidence intervals

For each outlet, the notebook asks whether the distribution of ratings differs across the three political groups.

- **Test.** A 3×3 chi-square statistic (political group × rating) is compared against a **permutation null**: the group labels are shuffled 5,000 times, and the p-value is the proportion of shuffles with a statistic at least as large as the observed one (with the usual `+1` correction).
- **Confidence intervals.** For each group, the share of "Direita" ratings is reported with a **Wilson 95% interval**, which behaves better than the normal approximation for small groups.
- **Multiple comparisons.** The 26 p-values are adjusted with the **Holm** step-down procedure.

The analysis is run twice:

- **A**: with the 10 manual reclassifications (the main analysis).
- **B**: using only respondents who declared a clear label, excluding those 10 rows.

**Outlets significant at Holm p < 0.05**

- Run A: CNN Brasil, Estadão, Folha, G1, O Globo, UOL, Valor
- Run B: CNN Brasil, Folha, G1, O Globo, UOL, Valor

Estadão is the only outlet whose significance changes between A and B (p_holm ≈ 0.16 in B). Its verdict is therefore the least stable of the significant outlets, which is worth keeping in mind when reading the table above.

## Limitations

- **Not a representative sample.** Respondents were recruited through social media. The sample is small (136 responses, of which 8 are from Portugal), concentrated in the Northeast of Brazil, and skewed toward self-declared left-leaning respondents. The results describe these respondents' perceptions, not public opinion.
- **Manual decisions.** The final labels for 10 respondents were chosen by the author after looking at the KNN output, so they are not purely algorithmic.
- **Mean imputation before KNN.** Missing ratings are filled with the column mean. This affects both the reference samples and the unclear respondents, and it pulls values toward the middle.
- **Verdicts are descriptive.** The verdict is a summary of respondents' perception, not a measure of an outlet's actual editorial line.
- **Sample size per cell.** Splitting by age band, region, and political group leaves small cells (the Portuguese group alone has 8 respondents). Percentages in those cells are unstable, and the confidence intervals in section 5 show how wide they are.
- **Plan vs. code.** The notebook's planning notes mention "N=5 N=7", but the code sweeps `k` and `sample_size` over {5, 7, 9, 11, 13}. The code is the reference here.

## Running the analysis

Requirements (Python 3):

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `jupyter` (or any environment that runs `.ipynb` files)

Steps:

1. Place `enquete.csv` in the project root, next to the notebook. The notebook reads it with a relative path.
2. Open [enquete-interpretation.ipynb](enquete-interpretation.ipynb) in Jupyter and run the cells top to bottom.

Note that the KNN step draws random samples. The notebook seeds only the robustness test (`seed=0`), so the reclassification charts and the manual decisions in step 2 may vary slightly between runs if the seed is not fixed.
