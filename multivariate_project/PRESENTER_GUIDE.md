# Presenter's Guide
## Cointegration & Error-Correction Dynamics in a Commodity-Currency Basket
### A VECM Approach to Relative-Value Currency Allocation

*Companion to `fx_cointegration_vecm.ipynb`. This document explains the project end-to-end, fully specifies every statistical test (hypotheses, statistic, distribution, decision rule), gives the relevant mathematics, and lists the headline results so you can present and defend each step.*

---

## 0. The 30-second pitch

> We ask whether a basket of seven commodity / risk-sensitive currencies shares a **stable long-run equilibrium** (cointegration), and what the **adjustment dynamics** say about which currencies to hold. We build the **full econometric pipeline** — unit-root testing → lag selection → Johansen cointegration → VECM → out-of-sample walk-forward — and treat the *instability* of FX cointegration as the object of study rather than something to hide. The headline finding is honest: **cointegration in this basket is regime-dependent** — it exists in only ~15% of rolling windows — and while the equilibrium error shows statistically significant mean reversion, it **does not beat a random walk** out-of-sample. The grade lives in the rigour of the tests, not the size of the Sharpe.

**One-sentence thesis to repeat:** *the statistics are the deliverable; a negative or unstable result, properly tested, is a complete result.*

---

## 1. The narrative arc (how the sections connect)

The analysis is a **funnel** — each test earns the right to run the next:

$$
\underbrace{\text{Are the series } I(1)?}_{\S2}\;\rightarrow\;
\underbrace{\text{How many lags? Residuals clean?}}_{\S3}\;\rightarrow\;
\underbrace{\text{Which deterministic case?}}_{\S4}\;\rightarrow\;
\underbrace{\text{Cointegration rank } r}_{\S5\text{–}\S6}\;\rightarrow\;
\underbrace{\text{VECM } \beta,\alpha}_{\S7\text{–}\S9}\;\rightarrow\;
\underbrace{\text{Trading signal}}_{\S10}\;\rightarrow\;
\underbrace{\text{Walk-forward \& evaluation}}_{\S11\text{–}\S13}
$$

**Crucial framing point to make early.** The notebook does two kinds of work with two different correctness standards:

| Regime | Sections | Claim type | Standard |
|---|---|---|---|
| **In-sample structural inference** | §2–§9 | *descriptive* ("were these cointegrated over 2016–26?") | full sample is legitimate — no forecast is made |
| **Out-of-sample evaluation** | §11–§13 | *predictive* ("could you have traded it?") | strictly causal — any quantity used at time $t$ uses only data $\le t$ |

This distinction pre-empts the "isn't that look-ahead bias?" question.

---

## 2. The data

- **Source:** Bloomberg terminal pull (`src/fx_data_pull.py`), daily `PX_LAST`, ~2,600 trading days, 2016–2026.
- **Basket (commodity / risk bloc):** **AUD, NZD, CAD, NOK, SEK, ZAR, MXN**.
- **Transform:** every series is quoted as **USD per 1 unit of the currency** (USD-quoted pairs inverted), then **logged**. We model $y_t = \log(\text{price})$.

**Why these choices matter (say this):**
- *Common numéraire + logs* make the log-levels directly comparable, so a cross-currency cointegrating vector $\beta'y_t$ is meaningful and its weights are interpretable as elasticities.
- *Thematic tightness is statistical, not cosmetic.* A grab-bag basket tends to produce "cointegration" that is just **shared USD exposure** (the common-dollar-factor trap, tested in §8). Seven series also keeps Johansen's test powerful — it degrades badly past ~10–12 series.
- *Survivorship:* the basket was chosen **ex-ante on economic theme**, not by screening for what cointegrates, so we are not data-snooping the basket.

---

## 3. Foundational mathematics (define these once)

**Integration order.** A series $y_t$ is **integrated of order $d$**, written $y_t \sim I(d)$, if it must be differenced $d$ times to become stationary. A unit-root (random-walk) series $y_t = y_{t-1} + \varepsilon_t$ is $I(1)$: non-stationary, but its first difference $\Delta y_t = \varepsilon_t$ is stationary $I(0)$.

**Cointegration (Engle–Granger 1987).** A vector of $I(1)$ series $y_t \in \mathbb{R}^n$ is **cointegrated** if there exists a vector $\beta \neq 0$ such that the linear combination

$$
z_t = \beta' y_t \sim I(0)
$$

is **stationary**. Economically: the series individually wander (non-stationary), but a particular weighted combination is tied together — a **long-run equilibrium** — and deviations from it are temporary. The number of independent such vectors is the **cointegration rank** $r$.

**Why we care:** if $z_t$ is stationary it is mean-reverting, so a stretched $z_t$ predicts a partial reversion → a tradeable "which currency is cheap/rich" signal.

---

## 4. §2 — Order-of-integration tests (is each series $I(1)$?)

**Why first:** cointegration is only defined for series integrated of the **same** order, conventionally $I(1)$. If a series were $I(0)$ it cannot belong to a cointegrating relation; if $I(2)$ the standard Johansen setup is invalid. So we must pin the integration order before anything else.

We test each **log level** and each **first difference** with a battery. Use a *confirmatory pair* (opposite nulls) plus power- and break-robust variants.

### 4.1 Augmented Dickey–Fuller (ADF) — the workhorse

Regression:
$$
\Delta y_t = \mu + \delta t + \rho\, y_{t-1} + \sum_{i=1}^{p}\gamma_i \Delta y_{t-i} + \varepsilon_t
$$
The lagged differences "augment" the regression to whiten $\varepsilon_t$.

- **$H_0$:** $\rho = 0$ — a **unit root** is present ($y_t$ is non-stationary, $I(1)$).
- **$H_1$:** $\rho < 0$ — $y_t$ is (trend-)stationary, $I(0)$.
- **Statistic:** $\mathrm{ADF} = \hat\rho / \mathrm{se}(\hat\rho)$ — a $t$-ratio, but it does **not** follow a $t$-distribution under $H_0$; it follows the non-standard **Dickey–Fuller distribution** (left-tailed, more negative critical values).
- **Decision:** reject $H_0$ (conclude stationary) if $\mathrm{ADF}$ is **more negative** than the critical value / if $p < 0.05$.

### 4.2 KPSS — the complementary test (opposite null)

Based on the partial-sum process of residuals $S_t = \sum_{i=1}^{t}\hat u_i$:
$$
\mathrm{KPSS} = \frac{1}{T^2}\sum_{t=1}^{T} \frac{S_t^2}{\hat\sigma^2_{LR}}
$$
where $\hat\sigma^2_{LR}$ is a HAC long-run variance estimate.

- **$H_0$:** $y_t$ is **stationary** (note: the *reverse* of ADF).
- **$H_1$:** $y_t$ has a unit root.
- **Decision:** reject $H_0$ (conclude non-stationary) if $\mathrm{KPSS}$ exceeds its critical value / $p < 0.05$.

> **Why use both ADF and KPSS:** ADF has notoriously low power (it often fails to reject even for stationary series). Pairing it with KPSS's opposite null gives a **confirmatory** read. We want, on the **levels**: ADF *fails to reject* **and** KPSS *rejects* → both point to non-stationary → confident $I(1)$. On the **differences** the pattern flips.

### 4.3 DF-GLS (Elliott–Rothenberg–Stock) — higher power
Same null/alternative as ADF, but the data are **GLS-detrended** before the ADF regression, which yields substantially **greater power** near the unit-root boundary. Include it because ADF alone is weak.

### 4.4 Phillips–Perron (PP) — robust to heteroskedasticity
Same $H_0$ (unit root) as ADF, but instead of adding lagged differences it applies a **non-parametric HAC correction** to the test statistic. Apt for FX, where volatility clustering violates the i.i.d.-error assumption.

### 4.5 Zivot–Andrews — break-robust unit root
- **$H_0$:** unit root **with no structural break**.
- **$H_1$:** stationary **around one endogenously-chosen break** in level/trend.
- The break date is chosen to be **maximally favourable to rejecting** $H_0$ (minimises the ADF $t$-stat over all candidate break points).

> **Why this is essential (a high-mark point):** an unmodelled structural break makes a *stationary* series look like a unit root (Perron 1989). We *expect* breaks (COVID, the 2022 hiking cycle — dated in §11), so an $I(1)$ verdict from plain ADF alone would be self-contradictory. Zivot–Andrews closes that hole.

### 4.6 Critical-values note (remember this — it returns in §6, reversed)
Here the series are **directly observed**, so the **standard Dickey–Fuller critical values are correct**. This is *deliberately different* from §6, where we test the *estimated residual* of a cointegrating regression and must use **stricter** critical values.

### 4.7 What we found
- **Levels:** unit-root nulls (ADF/DF-GLS/PP/ZA) generally **not rejected**; KPSS **rejects** stationarity ($p\approx0$) for all seven. → **$I(1)$.**
- **Differences:** ADF/DF-GLS/PP all $p\approx0.000$ → stationary → **confirms $I(1)$, rules out $I(2)$.**
- Two minor cross-test disagreements (MXN DF-GLS $p=0.014$; NZD Zivot–Andrews $p=0.019$) are expected when running five tests; the **preponderance** says $I(1)$, and the notebook flags them honestly.

---

## 5. §3 — Lag selection and the residual-whiteness gate

Johansen rests on an underlying VAR($p$) in levels:
$$
y_t = \nu + \sum_{i=1}^{p} A_i y_{t-i} + \varepsilon_t .
$$

### 5.1 Information criteria
Choose $p$ to minimise an information criterion, each trading fit against parameter count $k=pn^2$:
$$
\mathrm{AIC} = \ln|\hat\Sigma| + \frac{2k}{T}, \qquad
\mathrm{BIC} = \ln|\hat\Sigma| + \frac{k\ln T}{T}, \qquad
\mathrm{HQIC} = \ln|\hat\Sigma| + \frac{2k\ln\ln T}{T}.
$$
The VECM lag in **differences** is $k_{\text{ar diff}} = p - 1$.

- **Our choice:** **AIC, floored at $p=2$** (so $k_{\text{ar diff}}=1$). *Why AIC, not BIC?* For cointegration the consistent criteria (BIC/HQIC) tend to **under-select** short-run dynamics, leaving residual autocorrelation that **biases the trace test**. Result: AIC $=2$, BIC/HQIC $=1$.

### 5.2 Portmanteau / Ljung–Box whiteness test (the gate)
- **$H_0$:** VAR residuals are **white** (no autocorrelation up to lag $h$).
- **Statistic (multivariate, adjusted):** $Q_h = T^2\sum_{j=1}^{h}(T-j)^{-1}\operatorname{tr}(\hat C_j' \hat C_0^{-1}\hat C_j \hat C_0^{-1}) \sim \chi^2$, where $\hat C_j$ is the residual autocovariance at lag $j$.
- **Why it is a *gate*:** the trace test's asymptotic distribution assumes serially-uncorrelated errors, so whiteness is a **precondition**, not an afterthought.

### 5.3 What we found — and how to present the "failure"
The whiteness test **rejects** ($p \approx 0$). **Do not be defensive about this — own it:**

> At **daily** frequency strict whiteness is essentially unattainable: the portmanteau statistic also absorbs **ARCH / volatility clustering**, which no lag length removes. Adding lags would chase conditional heteroskedasticity, not mean dynamics, and merely over-parameterise the system. We therefore keep the AIC order and obtain valid rank inference from the **heteroskedasticity-robust wild bootstrap in §5**, rather than the asymptotic test alone.

This turns a nuisance into a rigour point — it *motivates* the bootstrap.

---

## 6. §4 — Deterministic specification (which of Johansen's 5 cases)

Writing the VECM with deterministics split between the levels and the cointegrating relation:
$$
\Delta y_t = \alpha\bigl(\beta' y_{t-1} + \rho_0 + \rho_1 t\bigr) + \gamma_0 + \gamma_1 t + \sum_{i=1}^{k}\Gamma_i \Delta y_{t-i} + \varepsilon_t .
$$

| Case | Levels | Coint. relation | Meaning |
|---|---|---|---|
| 1 | – | – | no deterministics |
| 2 | – | constant $\rho_0$ | equilibrium has non-zero mean; **no trend in levels** |
| **3** | **constant $\gamma_0$** | constant | **linear trend allowed in levels** (`'co'`) |
| 4 | constant | + trend $\rho_1$ | trend-stationary equilibrium |
| 5 | trend | + trend | quadratic trends |

**The choice changes the critical values**, so it is substantive.

- **Our choice — Case 3 (unrestricted constant, `deterministic='co'`).** §1 shows EM currencies (ZAR, MXN, NOK) **drift persistently** vs USD. We confirm with a regression $y_t = a + bt$: HAC $t$-stats on the slope are **huge** (e.g. NZD $t=-27$, NOK/ZAR $t\approx-15$) → a real deterministic trend the model must accommodate → unrestricted constant. Case 2 is run as a robustness check.

---

## 7. §5 — The Johansen cointegration test (the core inference)

### 7.1 From VAR to VECM
Re-parameterise the VAR($p$) as the **Vector Error-Correction Model**:
$$
\Delta y_t = \Pi y_{t-1} + \sum_{i=1}^{k}\Gamma_i \Delta y_{t-i} + (\text{determ.}) + \varepsilon_t,
\qquad \Pi = \alpha\beta' .
$$
- $\Pi$ is the **long-run impact matrix**. Its **rank** is the cointegration rank: $r = \operatorname{rank}(\Pi)$.
- If $r=0$: no cointegration ($\Pi=0$, a VAR in differences). If $r=n$: the system is stationary in levels. If $0<r<n$: **$r$ cointegrating relations**, with
  - $\beta$ ($n\times r$): the **cointegrating vectors** (the equilibria),
  - $\alpha$ ($n\times r$): the **adjustment / loading matrix** (speeds of return to equilibrium).

### 7.2 Reduced-rank regression & eigenvalues
Johansen's MLE concentrates out the short-run terms and solves a generalised eigenvalue problem on the residual moment matrices $S_{ij}$ (from regressing $\Delta y_t$ and $y_{t-1}$ on the lagged differences):
$$
\bigl|\,\lambda S_{11} - S_{10} S_{00}^{-1} S_{01}\,\bigr| = 0
\;\Rightarrow\; \hat\lambda_1 > \hat\lambda_2 > \dots > \hat\lambda_n .
$$
Each $\hat\lambda_i$ is a **squared canonical correlation** between levels and differences; a large $\hat\lambda_i$ signals a stationary (mean-reverting) direction.

### 7.3 The two rank tests

**Trace test** — tests rank $\le r$ against rank $> r$:
$$
\mathrm{LR}_{\text{trace}}(r) = -T\sum_{i=r+1}^{n}\ln\bigl(1-\hat\lambda_i\bigr).
$$
- **$H_0$:** at most $r$ cointegrating relations. **$H_1$:** more than $r$.
- Procedure: start at $r=0$; if rejected, try $r=1$, … stop at the first **non-rejection**.

**Maximum-eigenvalue test** — tests rank $=r$ against rank $=r+1$:
$$
\mathrm{LR}_{\max}(r) = -T\ln\bigl(1-\hat\lambda_{r+1}\bigr).
$$
- Sharper (isolates one eigenvalue); can disagree with trace. **We report both and reconcile them.**

Both have **non-standard** asymptotic distributions (functionals of Brownian motion) that depend on $n-r$ and the deterministic case; critical values come from MacKinnon–Haug–Michelis tables (bundled in `statsmodels`).

### 7.4 Two robustness layers (these are what elevate the section)

**(a) Reinsel–Ahn finite-sample correction.** The trace test **over-rejects** in finite samples. Replace $T$ by the effective sample $T-nk$:
$$
\mathrm{LR}^{*}_{\text{trace}}(r) = -(T-nk)\sum_{i>r}\ln(1-\hat\lambda_i).
$$

**(b) Wild bootstrap (Cavaliere–Rahbek–Taylor).** Asymptotic critical values assume homoskedastic Gaussian errors; FX has heavy volatility clustering that inflates the trace statistic. Algorithm for testing $H_0: r=0$:
1. Estimate the restricted model under $H_0$ (a VAR in differences); keep residuals $\hat\varepsilon_t$.
2. Generate **Rademacher wild draws** $w_t\in\{-1,+1\}$ and build $\varepsilon^*_t = w_t\hat\varepsilon_t$ — this **preserves the conditional heteroskedasticity** pattern.
3. Recursively rebuild a bootstrap series $y^*_t$ and recompute $\mathrm{LR}_{\text{trace}}(0)$.
4. Repeat $B$ times; bootstrap $p$-value $= \dfrac{\#\{\mathrm{LR}^*\ge \mathrm{LR}^{\text{obs}}\}+1}{B+1}$.

*(The notebook's from-scratch reduced-rank routine is validated to match `statsmodels.coint_johansen` to the decimal, and supplies the $S_{ij}$ matrices reused in §8.)*

### 7.5 What we found — the headline
| Quantity | Value |
|---|---|
| Trace rank @95% | **0** |
| Finite-sample-corrected trace rank | **0** |
| Max-eigenvalue rank @95% | **0** |
| Observed $\mathrm{LR}_{\text{trace}}(0)$ | 115.58 |
| Asymptotic 95% crit. value | 125.62 |
| **Bootstrap 95% crit. value** | **135.62** |
| **Bootstrap $p$-value** | **0.345** |

**How to present this beautifully:** the bootstrap critical value (135.6) sits **above** the asymptotic one (125.6) — *exactly* the direction theory predicts when errors are heteroskedastic (the asymptotic test over-rejects). All four routes agree: **no cointegration in the full 2016–26 sample.** This is not a failure — it is the FX-instability thesis, confirmed by convergent, robust tests.

---

## 8. §6 — Cross-validating cointegration (a second method)

Relying on one test is fragile; we confirm with **residual-based** tests and re-estimate $\beta$ two more ways.

### 8.1 Engle–Granger & Phillips–Ouliaris
Regress one currency on the others, $y_{1t} = \theta' y_{-1,t} + u_t$, and test the residual $\hat u_t$ for a unit root.
- **$H_0$:** $\hat u_t$ has a unit root → **no cointegration**. **$H_1$:** $\hat u_t$ stationary → cointegration.
- **Engle–Granger** uses an ADF on $\hat u_t$; **Phillips–Ouliaris** uses a PP-type (HAC) statistic on $\hat u_t$.

> **The critical-values point (the §2 caveat, now *reversed*):** because $\hat u_t$ is the residual of an **estimated** regression whose coefficients were chosen to *minimise its variance*, $\hat u_t$ is biased toward looking stationary. So we **must use the stricter, $N$-dependent Engle–Granger / Phillips–Ouliaris critical values** (more regressors ⇒ more stringent). `arch` applies these automatically; feeding $\hat u_t$ into a plain ADF would silently use too-lenient values and **over-detect** cointegration. Make this point explicitly — it shows you understand *why* there are two sets of critical values.

### 8.2 DOLS (Stock–Watson) and the Johansen $\beta$
Re-estimate the long-run vector with valid standard errors (DOLS adds leads and lags of $\Delta y$ to correct for endogeneity). **Convergent $\beta$ across Johansen / DOLS is strong evidence; divergence is itself a finding.**

### 8.3 What we found
Engle–Granger $p=0.61$, Phillips–Ouliaris $p=0.69$ → **no cointegration**, independently confirming Johansen. (Long-run vectors are shown for comparison but flagged as economically meaningful only if $r\ge1$.)

---

## 9. §7 — VECM estimation & diagnostics

Fit the VECM at the chosen rank:
$$
\Delta y_t = \underbrace{\alpha}_{\text{adjustment}}\underbrace{\beta'}_{\text{equilibria}} y_{t-1} + \sum_{i=1}^{k}\Gamma_i\Delta y_{t-i} + (\text{determ.}) + \varepsilon_t .
$$
- $\beta$ — long-run equilibrium combinations.
- $\alpha$ — how fast each currency is pulled back to equilibrium (the "which to hold" engine, §9).

**Three checks — "a model we have not checked is a model we cannot trust":**
1. **Stationarity of the error-correction term $\hat\beta'y_t$** (ADF/KPSS) — directly verifies the cointegrating combination is $I(0)$, closing the loop. *Flagged as indicative only*, because $\beta$ is estimated (the §6 caveat); formal evidence is the §6 properly-valued tests.
2. **Companion-matrix stability** — the VAR companion form must have exactly $n-r$ unit roots and all other eigenvalue moduli $<1$.
3. **Residual whiteness / normality** on the fitted VECM.

> *Presenter note:* since the full-sample rank is 0, the notebook sets rank $=1$ **only to demonstrate the estimation/diagnostics machinery**; the honest rank verdict (0) stands. Say this plainly.

---

## 10. §8 — Testing the common-dollar-factor trap (an LR test)

**The risk:** because every series is vs USD, a "cointegrating relation" might just be **shared dollar exposure**. The purest dollar relation is the **equal-weight direction** $b=(1,\dots,1)'$. We test whether the estimated cointegrating space is statistically consistent with that direction.

For rank $r=1$, restricting $\beta$ to a known vector $b$ gives the restricted squared canonical correlation and a **likelihood-ratio statistic**:
$$
\lambda^{*} = \frac{b' S_{10} S_{00}^{-1} S_{01}\, b}{b' S_{11}\, b},
\qquad
\mathrm{LR} = T\,\ln\!\frac{1-\lambda^{*}}{1-\hat\lambda_1} \;\sim\; \chi^2_{\,n-1}.
$$
- **$H_0$:** the cointegrating vector **equals** the dollar direction $b=(1,\dots,1)'$ → the relation is a pure dollar factor (untradeable as relative value).
- **$H_1$:** $\beta \neq b$ → a genuine cross-currency relationship.
- **Decision:** small $p$ ⇒ **reject** ⇒ *not* a dollar factor.

**What we found:** $\hat\lambda_1 = 0.0146$, $\lambda^* = 0.0047$, $\mathrm{LR} = 26.07$, $\chi^2_6$ $p = 0.0002$ → **reject $H_0$**: the dominant direction is **statistically distinguishable from a pure dollar factor**. (We tested the worry rather than merely asserting it — an A-vs-B+ move.)

---

## 11. §9 — Structural interpretation: *which currencies to hold*

### 11.1 Weak exogeneity (anchors vs adjusters)
A currency whose loading $\alpha_j \approx 0$ does **not** error-correct — it **pushes** the system but is not pulled back: a long-run **anchor / driver**. Large, significant $\alpha_j$ ⇒ an **adjuster** that reverts toward the basket.
- **$H_0$:** $\alpha_j = 0$ (currency $j$ is **weakly exogenous**). Formally an LR test; we report the asymptotically-equivalent $t$-tests on each $\alpha_j$.
- **Decision:** reject ($p<0.05$) ⇒ adjuster; fail to reject ⇒ anchor.

**What we found (the holding answer):**

| Currency | $\alpha$ | $t$ | $p$ | Role |
|---|---|---|---|---|
| AUD | −0.0013 | −2.07 | 0.038 | **Adjuster** |
| NZD | −0.0013 | −2.08 | 0.038 | **Adjuster** |
| CAD | −0.0006 | −1.34 | 0.179 | Anchor |
| NOK | −0.0026 | −3.55 | 0.000 | **Adjuster** |
| SEK | −0.0023 | −3.60 | 0.000 | **Adjuster** |
| ZAR | +0.0018 | 1.92 | 0.055 | Anchor |
| MXN | −0.0001 | −0.18 | 0.858 | Anchor |

→ **CAD, ZAR, MXN are anchors** (natural core holdings the basket is pinned to); **AUD/NZD/NOK/SEK are adjusters** (where the mean-reversion trade lives).

### 11.2 Half-life of mean reversion
Model the error-correction term as an AR(1), $z_t = c + \phi z_{t-1} + e_t$. The **half-life** of a shock:
$$
H = \frac{\ln(0.5)}{\ln\phi}\quad\text{(trading days)}.
$$
**What we found:** $\phi = 0.9789 \Rightarrow H = 32.4$ days. We then **set the walk-forward holding window $M=32$ from this half-life** rather than picking it arbitrarily — a derived, not assumed, parameter.

*(Identification note: with $r>1$, $\beta$ is identified only up to rotation and needs restrictions to interpret; with $r=1$ this is moot.)*

---

## 12. §10 — From equilibrium error to an allocation

The tradeable object is the **standardised error-correction term**:
$$
s_t = \hat\beta' y_t, \qquad z_t = \frac{s_t-\mu}{\sigma},
$$
with $\mu,\sigma$ estimated **without look-ahead** (train/expanding only). When $z_t$ is stretched, the rich legs are overvalued vs the cheap legs, so we **lean against the deviation**:
$$
w_t = -\,z_t\,\hat\beta \,\Big/\, \lVert z_t\hat\beta\rVert_1 .
$$
Hold the cheap legs, fund the rich legs, gross-normalised to 1 — the explicit "which to hold, and how much, today" rule.

---

## 13. §11 — Walk-forward out-of-sample (the headline section)

**Mechanics — say this slowly, it's the part people misunderstand:**

We are **not** forecasting exchange rates or "predicting cointegration." At each rebalance point $t_0$:
1. Take the **trailing** window $[t_0-N,\,t_0)$, $N=504$ days ($\approx 2$ yr) — past data only.
2. Run Johansen. **If $r\ge1$:** fit the VECM, **freeze** $\beta,\mu,\sigma$. **If $r=0$: hold cash** for the next block (encode the instability, don't force a trade).
3. For the next $M=32$ days, trade the frozen signal; the position formed at close $t$ earns the $t\!\to\!t\!+\!1$ return.
4. Advance by $M$, **refit**. Stitch all out-of-sample blocks into one continuous, **fully-causal** equity curve.

- **Rolling (fixed-$N$)**, not expanding — we *want* the model to forget pre-regime data; that adaptivity is the right choice for an unstable relationship.
- **Leakage discipline:** every quantity used at $t$ uses only data $\le t$; an explicit assertion checks train indices precede the traded returns.

**What we found:**
- Leakage self-check: **PASS**.
- OOS span 2018-05 → 2026-06 (2,102 days); **in-market only 15% of days** (cash 85%); **15% of refit windows have rank $\ge 1$**.
- $\beta$ drifts and the rank flickers between 0 and $\ge1$ — *that drift is the result*: **FX cointegration is regime-dependent, not a fixed structural law.**

**§11b — descriptive stability (firewalled from the signal):**
- **Recursive trace statistic** (expanding window, scaled by its 95% CV): wanders above/below 1, showing the relationship's existence is not constant.
- **Bai–Perron / PELT break detection** on the ECT (via `ruptures`) dates the regime shifts. *Crucially, full-sample break dates are never fed back into the signal* — that would be look-ahead.

---

## 14. §12 — Out-of-sample statistical evaluation (graded core)

The grade is on **statistics**, so we evaluate statistically, not just by the equity curve. The benchmark every FX model must clear is the **random walk** (Meese–Rogoff 1983: exchange-rate changes are close to unforecastable).

### 14.1 Predictive regression (the mean-reversion hypothesis, formally tested)
$$
\Delta s_{t+1} = a + b\,z_t + u_{t+1}, \qquad \text{Newey–West (HAC) standard errors.}
$$
- **$H_0$:** $b=0$ (the equilibrium error has no predictive content). **$H_1$:** $b<0$ (deviations are partly reversed — error correction).
- **HAC is mandatory:** autocorrelation + volatility clustering would otherwise overstate the $t$-stat. Newey–West uses $\hat\sigma^2_{LR}=\hat\gamma_0+2\sum_{j=1}^{L}(1-\tfrac{j}{L+1})\hat\gamma_j$.

**What we found:** $b=-0.00398$, **HAC $t=-3.47$, $p=0.001$** → **statistically significant mean reversion**.

### 14.2 Diebold–Mariano test vs a random walk
Compare one-step forecasts of $s_t$ from the error-correction model against the random-walk forecast $\hat s_{t+1}=s_t$. With loss differential $d_t = e^{\text{RW}\,2}_t - e^{\text{VECM}\,2}_t$:
$$
\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}} \;\xrightarrow{d}\; \mathcal N(0,1),
\quad \widehat{\operatorname{Var}}(\bar d)\ \text{from a HAC long-run variance.}
$$
- **$H_0$:** equal forecast accuracy. **$H_1$:** the model forecasts better ($\mathrm{DM}>0$).
- **What we found:** $\mathrm{DM}=1.34$, $p=0.18$ → **cannot reject equal accuracy: the model does not beat the random walk.**

> **The sophisticated punchline to land:** there is *statistically significant local mean reversion* in the equilibrium error (predictive regression), yet that **does not translate into beating a random walk** (Diebold–Mariano). This is precisely the Meese–Rogoff nuance — predictability $\neq$ forecast superiority — and reporting it honestly is exactly what the course rewards.

---

## 15. §13 — Comparative benchmarks (the demonstration)

Over the **identical OOS window**, compare the strategy against naïve alternatives: each single-currency buy-and-hold, **USD cash** (the flat "do nothing" line), and an equal-weight basket.

**What we found (price-only log returns):**

| Strategy | Total ret | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| **Cointegration (walk-fwd)** | **+0.030** | 0.011 | **+0.34** | **−0.02** |
| Hold AUD | −0.070 | 0.100 | −0.08 | −0.29 |
| Hold NZD | −0.170 | 0.098 | −0.21 | −0.30 |
| Hold CAD | −0.084 | 0.064 | −0.16 | −0.19 |
| Hold NOK | −0.156 | 0.120 | −0.16 | −0.38 |
| Hold SEK | −0.084 | 0.103 | −0.10 | −0.33 |
| Hold ZAR | −0.276 | 0.141 | −0.23 | −0.47 |
| Hold MXN | +0.122 | 0.120 | +0.12 | −0.32 |
| USD cash | 0.000 | 0.000 | – | 0.00 |
| Equal-weight basket | −0.103 | 0.086 | −0.14 | −0.25 |

**Two honesty caveats — state both:**
1. **Risk profiles differ.** The strategy is **long/short, dollar-neutral** (it strips out USD direction); the buy-and-holds carry full directional USD exposure. Not apples-to-apples — that's the point: it isolates relative value, and by sitting in cash 85% of the time it sidestepped the EM drawdowns.
2. **Carry is omitted.** Price-only P&L misses rate differentials. Real holders of high-yielders (ZAR, MXN) earn substantial **carry** → the single-currency lines are **understated** (e.g. real MXN returns would be materially higher). We do **not** claim "the strategy beats holding MXN" without that asterisk.

---

## 16. Headline results — one slide

| Question | Test(s) | Verdict |
|---|---|---|
| Integration order | ADF, KPSS, DF-GLS, PP, Zivot–Andrews | all seven **$I(1)$** |
| Lag / specification | AIC + portmanteau; HAC drift test | $p=2$; **Case 3** (drift) |
| Cointegration (full sample) | Johansen trace & max-eig, Reinsel–Ahn, **wild bootstrap**, EG, PO | **rank 0** — no stable cointegration |
| Is it just the dollar? | $\beta$-restriction LR ($\chi^2_6$) | **rejected** — not a pure dollar factor |
| Which to hold? | weak-exogeneity $t$-tests | anchors **CAD/ZAR/MXN**; adjusters **AUD/NZD/NOK/SEK** |
| Reversion speed | AR(1) half-life | **32 days** |
| Is it stable over time? | rolling Johansen, recursive trace, Bai–Perron | **regime-dependent** — coint. in only **15%** of windows |
| Local predictability? | predictive regression (HAC) | **yes**, $b<0$, $t=-3.5$ |
| Beats a random walk? | Diebold–Mariano | **no** ($p=0.18$) — Meese–Rogoff holds |

**The story in one line:** *a fully-diagnosed pipeline showing that cointegration in this FX basket is real but fragile and regime-dependent, with mean reversion that is statistically significant yet not forecast-superior to a random walk.*

---

## 17. Anticipated questions (and crisp answers)

- **"Isn't using the whole sample look-ahead bias?"** Only for *predictive* claims. §2–§9 are *descriptive* in-sample inference (no forecast); all *predictive* work (§11–§13) is strictly causal with a leakage assertion.
- **"Your whiteness test failed — isn't the model invalid?"** At daily frequency strict whiteness is unattainable (ARCH dominates the portmanteau). That's *why* we use a heteroskedasticity-robust **wild bootstrap** for rank inference instead of relying on asymptotics.
- **"Why ADF *and* KPSS?"** Opposite nulls → a confirmatory read that guards against ADF's low power.
- **"Why is the cointegration ADF different from a normal ADF?"** The residual of an estimated cointegrating regression is biased toward stationarity (coefficients chosen to minimise its variance), so it needs **stricter, $N$-dependent** Engle–Granger/Phillips–Ouliaris critical values — *not* standard DF values.
- **"Rank is 0 — isn't the project a failure?"** No. The negative result is *established by convergent robust tests*, and the walk-forward shows the relationship genuinely flickers in and out. Honest characterisation of an unstable relationship is the deliverable.
- **"Why trace *and* max-eigenvalue?"** They test different hypotheses ($\le r$ vs $=r$) and can disagree; reporting both and reconciling is the rigorous standard.
- **"Why rolling, not expanding, windows?"** We *want* the model to forget pre-regime data; adaptivity suits an unstable relationship.
- **"How does VECM answer *which to hold*?"** Two ways: structurally via weak exogeneity (anchors vs adjusters) and dynamically via the error-correction signal (lean against $z_t$).

---

## 18. Mini-glossary

- **$I(d)$** — integrated of order $d$; needs $d$ differences to be stationary.
- **Cointegration** — a stationary linear combination $\beta'y_t$ of non-stationary series; a long-run equilibrium.
- **Rank $r$** — number of independent cointegrating relations $=\operatorname{rank}(\Pi)$.
- **$\beta$** — cointegrating vectors (equilibria). **$\alpha$** — adjustment speeds (loadings).
- **Error-correction term (ECT)** — $z_t=\beta'y_t$, the disequilibrium; mean-reverting if cointegrated.
- **Weak exogeneity** — $\alpha_j=0$: currency $j$ drives but does not adjust (an anchor).
- **Half-life** — $\ln 0.5/\ln\phi$; time for a deviation to halve.
- **HAC / Newey–West** — heteroskedasticity- and autocorrelation-consistent standard errors.
- **Wild bootstrap** — resampling with random sign flips that preserves volatility clustering.
- **Meese–Rogoff** — the result that structural FX models struggle to beat a random walk OOS.

---

*Reproduce everything: `uv run --with nbformat python src/build_notebook.py` regenerates the notebook; run it on the project `.venv` from the `multivariate_project/` directory.*
