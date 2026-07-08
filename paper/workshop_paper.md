# Visible but Idle: A Gradient-Only Detector of Concept Use Validated by Symmetric Causal Interchange

## Abstract

Methods that translate hidden activations into words can tell us that a concept is represented inside a language model. They do not, by themselves, tell us whether that concept contributes to the model's next behavior. We study this distinction using Anthropic's Jacobian Lens (J-Lens) and J-Space as the representational foundation. Our contribution is not a new lens. It is a candidate *READ* signal and an intervention-based test of what that signal can and cannot support.

We construct matched prompts in which the same explicit concept is either needed to answer a question (an *engine*) or present but irrelevant to the answer (a *dashboard*). On Qwen2.5-7B-Instruct at layer 16, we first verify that the concept is written in both cases. We then establish expensive causal ground truth by interchanging the complete residual state at the concept token in both directions. Finally, we evaluate `READ_IG`, a donor-free score that integrates output gradients along a path between paired J-Lens concept directions. The cheap path is firewalled from the causal results.

Of 118 candidate pairs, 25 are reserved for calibration and 93 for evaluation; 77 evaluation pairs in 24 dependency groups pass all frozen verification gates. `READ_IG` separates the causally relevant engines from both the original idle controls and answer-type-matched idle controls with ROC AUC 1.000 (95% group-bootstrap CI [1.000, 1.000]). A local-gradient score reaches 0.915 [0.864, 0.967], while a static capacity baseline is 0.500. The stronger interpretation fails: within engines, `READ_IG` does not rank causal magnitude (Spearman's rho = -0.179, 95% CI [-0.431, 0.126]). We therefore find evidence for a binary relevant-versus-idle detector in this setting, not for a graded meter of causal use. We label the broader graded claim **ARTIFACT (partial)** and state the scope narrowly: one model, explicit concepts, one selected layer, and a roster whose positive examples already have strong causal effects.

## 1. Introduction

A model can represent a concept without using it for the behavior we are examining. A prompt about aluminum may leave a clear aluminum-related state in the residual stream even when the requested answer is simply `2 + 2`. For an interpretability display, both a chemistry question and the arithmetic continuation may visibly contain *aluminum*. For a behavioral audit, however, they are different. In the chemistry question, changing the concept should change the answer. In the arithmetic continuation, it should not.

Anthropic's work on [verbalizable representations and the global workspace](https://transformer-circuits.pub/2026/workspace/index.html) provides the starting point for this study. Its [J-Lens reference implementation](https://github.com/anthropics/jacobian-lens) transports an internal residual vector through an average Jacobian and decodes it in the model's output vocabulary. This makes otherwise hidden representations legible in word-like form. We reproduce and use that machinery; we do not claim it as our contribution.

The question here begins one step later: once J-Lens has surfaced a concept, can we tell whether downstream computation is using it? The distinction matters for proposed monitoring applications. A detector that reports every visible concept as behaviorally active can create false alarms. A detector that mistakes an idle trace for a driver can also produce false mechanistic explanations. In external commentary accompanying Anthropic's report, Neel Nanda specifically asks for more evidence about reliability and false-positive behavior ([commentary, pp. 41--42](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf)). The present experiment supplies one small, controlled piece of that evidence.

We call concept visibility **WRITTEN** and downstream behavioral sensitivity **READ**. The names are operational rather than ontological. WRITTEN means that the selected activation has a sufficiently large projection onto a J-Lens concept direction. READ means that a separately computed gradient score predicts membership in a causally validated relevant or idle task class. Neither label establishes that we have found a unique semantic variable or a complete circuit.

Our main result has two parts. First, the gradient-integrated score `READ_IG` perfectly separates the relevant and idle classes in the frozen evaluation roster, including an answer-type-matched stress control. Second, that score does not order the already-relevant examples by causal-effect strength. The first result is encouraging, but the second changes its interpretation. The experiment supports a binary detector under these conditions. It does not support a graded “how much was this concept used?” ruler.

This paper makes three practical contributions:

1. It defines a symmetric full-residual interchange score as causal ground truth for matched concept pairs.
2. It evaluates a gradient-only, donor-free `READ_IG` estimator behind an explicit anti-circularity firewall.
3. It reports both the successful binary test and the failed graded test, together with the instrument failures that shaped the final protocol.

We use “gradient-only” to distinguish the estimator from donor-state interchange, not to claim that it is free to compute. `READ_IG` requires 16 gradient-bearing path evaluations. We did not benchmark wall-clock time or cost, so we do not make an empirical speedup claim.

## 2. Background: from J-Lens visibility to behavioral use

### 2.1 Anthropic's J-Lens and J-Space

J-Lens is an Anthropic-developed method for reading verbalizable content from residual-stream activations. In simplified notation, a residual vector $h_\ell$ at layer $\ell$ is transported to the final residual basis using an average input-output Jacobian $J_\ell$, then decoded using the model's unembedding $U$:

$$
\operatorname{lens}_\ell(h_\ell) = U J_\ell h_\ell.
$$

The resulting vocabulary scores let a researcher inspect which words an activation is disposed to make the model say. Anthropic's broader J-Space analysis argues that verbalizable representations can act like a global workspace shared across downstream computations. The accompanying code and lenses are the foundation we rely on here. The J-Lens repository is released under Apache-2.0; this artifact keeps its attribution and license boundary explicit.

For a concept direction $v$ and residual state $h$, we use the scalar projection $h^\top v$ as the WRITTEN quantity. The final protocol selects layer 16 and a WRITTEN threshold of 2.482431 using calibration data only. Passing this gate says that the concept is detectably represented at the chosen token and layer. It says nothing yet about whether changing that state will affect the requested answer.

### 2.2 WRITTEN is not READ

Our motivating contrast is between an *engine* and a *dashboard*. An engine prompt uses a concept to determine the completion. For example, a prompt can explicitly state that atomic number 13 corresponds to aluminum and ask for the chemical symbol. A dashboard prompt retains the same factual context and the same explicit aluminum token but asks for an unrelated answer. The dashboard may faithfully display the concept even though it does not drive the selected behavior.

We use causal interchange to decide whether this contrast actually holds, then ask whether a cheaper score can predict the validated class. This ordering is important. A high WRITTEN score is a dataset inclusion criterion, not a positive label. Likewise, a high `READ_IG` score is evaluated against independently constructed task labels whose causal behavior is checked after the fact. We do not define “used” by thresholding the same score that we evaluate.

### 2.3 Integrated gradients as a path attribution

`READ_IG` adapts the path-integral idea behind [Integrated Gradients](https://arxiv.org/abs/1703.01365). Standard integrated gradients accumulate local sensitivities between a baseline and an input. Here the path is in activation space and is defined by a matched pair of J-Lens concept directions. The estimator measures how the behavior metric changes as the clean activation is moved away from its own concept and toward the paired concept. This is a local, behavior-conditioned quantity. It does not require a clean donor activation from another prompt, and it never reads the causal interchange output.

## 3. Method

### 3.1 Model, data, and frozen split

All reported experiments use `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, in bfloat16. The model family is documented in the [Qwen2.5 technical report](https://arxiv.org/abs/2412.15115). We fix random seed 1729, use five evaluation folds, and use 10,000 bootstrap draws. A 20-prompt wrapper-agreement check gives a maximum mean KL divergence of $1.66\times10^{-8}$, well below the predeclared $10^{-3}$ tolerance.

The dataset contains 118 deterministic, reciprocal prompt pairs covering element-symbol, country-capital, and US-state-capital relations. Contexts that reuse an unordered concept pair belong to the same dependency group. We allocate whole groups to calibration until 25 prompt pairs are reserved; the remaining 93 pairs are evaluation candidates. Calibration is used to choose the layer and WRITTEN threshold. It is not included in the reported trust check.

An evaluation pair must satisfy all of the following frozen checks in both directions:

- the clean engine's top completion is the declared answer;
- the J-Lens own-concept projection at the explicit concept token exceeds the frozen WRITTEN threshold;
- the corresponding idle control produces its declared answer; and
- the same explicit concept remains WRITTEN in that control.

Seventy-seven evaluation pairs pass, and 16 remain `UNVERIFIED`; no failed row is relabeled. The 77 verified pairs span 24 evaluation dependency groups. The selected intervention location is layer 16 at the explicit concept token in a byte-identical shared fact context. This design deliberately narrows the result to explicit concepts.

### 3.2 Expensive causal ground truth

For a matched engine pair $A,B$, let the answer metric be the final-token logit difference

$$
M = \operatorname{logit}(y_A)-\operatorname{logit}(y_B),
$$

and let the clean normalization be $T=M_A-M_B$. We capture the complete post-block residual vector at the explicit concept token for each clean prompt. We then install $B$'s residual vector into $A$ at that one location, and vice versa. If $M_{A\leftarrow B}$ and $M_{B\leftarrow A}$ are the two edited metrics, the directional recoveries are

$$
R_{A\leftarrow B}=\frac{M_A-M_{A\leftarrow B}}{T}, \qquad
R_{B\leftarrow A}=\frac{M_{B\leftarrow A}-M_B}{T}.
$$

Our primary causal score is

$$
C=\frac{1}{2}\left(R_{A\leftarrow B}+R_{B\leftarrow A}\right).
$$

We preserve $C$ as signed and unclipped. Values above one or below zero therefore remain visible rather than being silently truncated. A directional discrepancy greater than 0.50 is pre-flagged; none of the 77 final engine, original-dashboard, or hard-dashboard measurements crosses that threshold. Controls use their own answer metric but retain the matched engine's $T$ as the response scale, so a dashboard's near-zero effect is expressed relative to the engine behavior it is meant to contrast.

This operation is deliberately expensive and strong: it requires clean residual donors and two edited forwards per pair. We treat it as the causal truth for this experiment, not as the proposed monitoring signal.

### 3.3 Gradient-only READ estimators

Let $h_A,h_B$ be clean layer-16 states and $v_A,v_B$ the corresponding unit J-Lens concept directions. The two activation paths are defined by

$$
\Delta_A=(h_A^\top v_A)(v_B-v_A), \qquad
\Delta_B=(h_B^\top v_B)(v_A-v_B).
$$

For direction $i\in\{A,B\}$ and $K=16$ midpoint steps, we compute

$$
I_i=\frac{1}{K}\sum_{k=1}^{K}
\nabla_h M\!\left(h_i+\frac{k-1/2}{K}\Delta_i\right)^\top\Delta_i,
$$

then define

$$
\operatorname{READ}_{IG}=\frac{|I_A|+|I_B|}{2}.
$$

The absolute value makes the score a magnitude of behavior-conditioned sensitivity; it is not a claim about a universal semantic sign. The paired, symmetric construction avoids choosing whichever concept direction happens to look more favorable.

We report two comparisons. `READ_local` evaluates only the clean-point derivative, combining concept amount with the gradient projected onto its own concept direction. The capacity baseline passes the concept directions through a downstream MLP and measures static response norm without using the behavior metric. It tests a simpler hypothesis: perhaps directions associated with engines merely occupy higher-gain downstream subspaces.

We use the word “cheap” operationally: the READ path avoids storing and installing donor activations and avoids causal interchange outputs. It still performs multiple forward/backward evaluations. Since this study contains no runtime benchmark, the defensible description is *gradient-only and donor-free*, not *measured faster*.

### 3.4 Anti-circularity firewall

The causal and cheap computations are separated in code and artifacts:

1. Dataset construction freezes the model revision, layer, threshold, folds, verified rows, clean prompts, and direction cache.
2. The causal stage reads that manifest and writes $C$ without exposing it to the cheap estimator.
3. The cheap stage reads only the sanitized clean manifest and direction cache. Its module imports no causal, patching, or interchange code and has no argument through which $C$ can enter.
4. For the hard controls, `READ_IG` is computed and frozen before hard-control $C$ is calculated.
5. Only the model-free evaluation stage joins the frozen READ and causal artifacts by pair ID.

Thus the causal scores cannot select a per-example sign, transformation, threshold, or exclusion in the cheap path. The selected layer and WRITTEN threshold come from calibration groups, not from the 24 evaluation groups. This firewall is the main reason the final binary result is interpretable despite the extensive failed work that preceded it.

### 3.5 Evaluation and the hard control

The primary test is pooled out-of-fold ROC AUC over five deterministic folds assigned by complete dependency group. Confidence intervals resample all rows belonging to an unordered concept group together for 10,000 draws. This preserves repeated contexts and their matched engine/control dependence.

The ROC labels are the constructed task classes: relevant engine versus deliberately idle dashboard. They are **not** obtained by thresholding $C$. The causal score instead validates the intended meaning of those classes: engines should have large interchange effects and dashboards should not. This distinction prevents AUC from becoming a tautological prediction of a discretized version of the target score.

The original dashboards ask an arithmetic question and therefore differ from engines in answer type. We add a harder control that retains the original natural context and explicit source concept but asks the same relation and semantic answer class using a fixed calibration-only anchor: platinum $\rightarrow$ `Pt`, Netherlands $\rightarrow$ `Amsterdam`, or Alabama $\rightarrow$ `Montgomery`. The source concept cannot determine that fixed answer. All 77 hard controls pass their frozen clean-answer and WRITTEN checks.

Finally, we test gradedness without making a post-hoc weak/strong cutoff. Among the 77 engines alone, we compute Spearman correlation between the frozen READ score and $|C|$, again with dependency-group bootstrap confidence intervals. This is the decisive test of whether READ ranks causal magnitude after class membership is held fixed.

## 4. Results

### 4.1 The causal instrument separates relevant and idle concepts

The engine median signed $C$ is 0.912714, and its median $|C|$ is also 0.912714. The original dashboards have median signed $C=-0.002043$ and median $|C|=0.005083$. The answer-type-matched hard dashboards have median $|C|=0.006466$. None of the three families has a sharp directional-disagreement flag. The full-state interchange therefore behaves as required for the final roster: changing the explicit concept state changes the answer in engines, but barely affects either idle control family.

![Causal effects for engines and idle controls](figures/f1_causal_sanity.png)

**Figure 1.** Symmetric full-residual causal sanity. Engines have large effects; original and hard dashboards remain close to zero.

The agreement between the two swap directions is shown separately because a mean can otherwise conceal an unstable intervention.

![Directional agreement for symmetric interchange](figures/f6_directional_agreement.png)

**Figure 2.** Directional interchange agreement. The final rows do not rely on one anomalous swap direction.

### 4.2 READ detects relevant versus idle use

The primary `READ_IG` score separates engines from original dashboards with AUC 1.000 and a 95% group-bootstrap interval of [1.000, 1.000]. `READ_local` is weaker but still informative at 0.914825 [0.863661, 0.967161]. The behavior-independent capacity baseline is exactly chance at 0.500 [0.500, 0.500].

| Estimator | Engine vs. original dashboard AUC | 95% group-bootstrap CI |
| --- | ---: | ---: |
| `READ_IG` | 1.000000 | [1.000000, 1.000000] |
| `READ_local` | 0.914825 | [0.863661, 0.967161] |
| static capacity baseline | 0.500000 | [0.500000, 0.500000] |

![Binary AUC and baseline](figures/f2_binary_auc_and_baseline.png)

**Figure 3.** Held-out binary detection. The AUC labels are relevant/idle task classes validated by $C$, not labels created by thresholding $C$.

The hard-control comparison produces the same primary result: engine versus hard dashboard AUC is 1.000 [1.000, 1.000]. This rules out the narrow explanation that `READ_IG` succeeds only because the original dashboards end in arithmetic rather than a chemical symbol or capital name.

![Original and answer-type-matched control AUC](figures/f4_hard_dashboard_auc.png)

**Figure 4.** `READ_IG` maintains perfect separation when the idle control uses the same semantic answer type as the engine.

The raw distributions make the operating behavior clearer. Engine scores range from 0.034970 to 0.992770, with median 0.210247. Original-dashboard scores range from 0.000881 to 0.023411, with median 0.005553. Hard-dashboard scores range from 0.001650 to 0.021032, with median 0.006319. The two dashboard ranges substantially overlap each other and remain disjoint from the engine range in this roster.

![Raw READ_IG distributions](figures/f5_read_ig_distributions.png)

**Figure 5.** The final score behaves as a relevant-versus-idle separator: both idle families occupy a compressed low band.

Perfect empirical separation should be read literally and locally. It says that no verified dashboard outranks a verified engine under the frozen score on this dataset. The bootstrap interval conditions on the sampled dependency groups; it does not establish a universal false-positive rate for other concepts, layers, models, or prompt distributions.

### 4.3 READ does not measure graded causal strength

Within engines, $|C|$ ranges only from 0.785789 to 1.012025. Over these 77 already-strong cases, `READ_IG` has Spearman $\rho=-0.179110$ with a 95% group-bootstrap interval of [-0.431377, 0.126014]. The point estimate is negative and the interval spans zero. `READ_local` and the capacity baseline also have intervals spanning zero. We therefore find no positive graded-use signal within engines.

![Engine-only graded check](figures/f3_engine_only_graded_check.png)

**Figure 6.** Engine-only `READ_IG` versus $|C|$. The frozen score does not order causal strength among the positive class.

Across engines and dashboards pooled together, `READ_IG` has $\rho=0.707412$ with $|C|$. That number is not evidence for graded measurement: it is largely the same between-class separation already captured by AUC. Once class membership is held fixed, the positive association disappears. We did not create a weak/strong engine cutoff after seeing the values, because the protocol specified no such cutoff and the observed range contains no defensible natural split.

The formal verdict is therefore asymmetric:

- **binary relevant-versus-idle detector:** supported;
- **graded causal-use meter:** not supported; and
- **broader stress-test label:** **ARTIFACT (partial)**.

“Partial” does not erase the binary result. It records that a narrower interpretation survives while the stronger one does not.

## 5. Research arc and limitations

### 5.1 Failed instruments before a usable ground truth

The clean result was reached only after several invalid or unsuccessful designs. We preserve these experiments because they change how much confidence the final number deserves.

The first apparent negative used a broken causal instrument. In a canonical check, swapping the internal representation of spider toward ant should move a leg-count answer from 8 toward 6. The run produced 4 instead. We discarded the scientific conclusion: a failed calibration cannot adjudicate the hypothesis. After correcting token surfaces, the Jacobian convention, and the intervention setup, three canonical swaps reproduced in all three cases.

The repaired edit was then too broad. An all-position, layer-13--24 intervention at strength $\alpha=2$ changed negative log-likelihood on 24 unrelated texts by a mean of 0.623323 and a mean absolute amount of 0.669081, above the predeclared 0.25 damage guard. Apparent causal effects under that edit could reflect general model damage. A masked variant avoided the measured damage on the capability bank only because all 24 masks were empty, so that null was a no-op rather than affirmative safety evidence.

Later, the first final-position version of the matched protocol made the dashboard WRITTEN check void. Moving to a shared-context boundary restored comparability, but layer 26 had engine median $|C|\approx0.0022$: with only one downstream block left, the instrument had almost no leverage. A calibration sweep over latent boundary states did not rescue it; the best engine median was 0.0076 versus 0.0039 for dashboards. These failures motivated the explicit natural fact and the exact concept-token intervention at layer 16. That change also narrows the claim: the final result is about explicitly written concepts, not arbitrary latent thoughts.

### 5.2 READ definitions that failed

Several plausible READ scores did not validate. A local attribution attempt had Pearson correlation 0.061845 with the intervention endpoint (95% CI [-0.406415, 0.482404], $N=20$). A global static capacity score marked none of eight causally quiet narration examples as low-read. A behavior-specific path score reached Spearman $\rho=-0.076623$ (95% CI [-0.517658, 0.408929], $N=21$), while its path finder returned an empty path for all eight narration controls. An absent measurement was not counted as evidence of low use.

An intermediate held-out validation also failed on its own terms. It contained 163 cases but only four independent dashboard concepts; 49 of the 163 rows failed declared-label verification. Two causal proxies agreed only moderately ($\rho=0.410167$). The only complete global READ score ranked the intended classes backwards, with AUC 0.078333 [0.021429, 0.152032], and the more targeted scores were undefined under the predeclared complete-coverage rule. We did not flip the sign or fill missing values after observing the result.

Those failures led to the final matched design: clean labels, reciprocal concepts, explicit tokens, symmetric full-state truth, group-held-out evaluation, and a cheap score frozen without access to causal outcomes. The failures are not independent evidence that `READ_IG` works, but they explain why the successful protocol includes unusually explicit gates.

### 5.3 What the final experiment still does not establish

The evidence remains limited in several ways.

First, this is one pinned 7B model, one selected layer, and three structured relation families. It does not establish transfer to other model sizes, architectures, layers, multilingual concepts, free-form reasoning, or concepts that are only implicit. Selection of layer 16 and the WRITTEN threshold was protected by calibration groups, but it remains a choice tailored to this model and dataset.

Second, the 77 engines occupy a narrow and already-strong causal range. Range restriction may make a real graded relationship difficult to detect. It cannot, however, justify claiming such a relationship from the present data. The needed follow-up is a dataset designed in advance to span weak, medium, and strong causal effects while holding prompt family and answer type fixed.

Third, full-residual interchange is a behavioral intervention, not proof that a single J-Lens direction is the unique mechanism. Replacing the full state at one token can transport other information correlated with the concept. The matched construction, WRITTEN gate, two directions, and idle controls reduce this ambiguity but do not eliminate it.

Fourth, the hard control addresses answer-type confounding but is not exhaustive. It uses three fixed calibration anchors and the same relation families as the engines. Other idle controls could preserve more lexical, syntactic, or distributional detail. AUC 1.000 on this roster should motivate adversarial controls, not end the search for false positives.

Fifth, group bootstrap quantifies sampling variability over the 24 observed dependency groups. Perfect separation yields a degenerate-looking interval because every resample preserves separation. This interval does not account for model selection, new task families, prompt paraphrases, or distribution shift.

Finally, `READ_IG` has not been timed against causal interchange. It is donor-free and does not require causal artifacts, but 16 midpoint gradient evaluations are not free. A deployment-oriented claim would require wall-clock, memory, throughput, and calibration-cost measurements on the same hardware.

An exploratory post-success localization test also failed to identify a strongly faithful compact top-eight circuit: faithfulness fractions for three cases were 0.3649, 0.2387, and 0.3623. We make no compact-circuit claim.

## 6. Conclusion and future work

This experiment separates two questions that are often collapsed. Anthropic's J-Lens can expose a verbalizable concept in a residual state. Our results ask whether a gradient-only signal can predict whether that visible concept matters for a particular answer.

Within the frozen Qwen2.5-7B setting, the answer is yes for a binary distinction. `READ_IG` separates causally validated relevant concepts from two families of causally idle concepts, including controls matched on semantic answer type. The anti-circularity firewall matters here: the estimator never sees $C$, donor activations, edited outputs, or per-example causal labels. The chance-level capacity baseline also suggests that the result is not explained simply by a concept direction lying in a high-gain downstream subspace.

The answer is no for the stronger graded claim. `READ_IG` does not rank causal magnitude within engines, and the pooled correlation is dominated by class separation. We therefore regard the main result as a validated use-versus-idle detector in a narrow setting, not as a general meter of how strongly a model used a thought.

The next experiment should be designed around this failure. It should preregister a broad causal range; include weak, medium, and strong engines rather than deriving a cutoff after the fact; use multiple models and layers; add adversarial idle controls with tighter lexical matching; test implicit as well as explicit concepts; and report runtime alongside causal and statistical reliability. The most informative outcome may again be mixed: a detector that is useful for screening can still be unsuitable for causal ranking. Keeping those claims separate is a feature, not a concession.

## Reproducibility and ethics statement

The artifact pins the model ID and revision, bfloat16 dtype, seed 1729, layer 16, WRITTEN threshold 2.482431, 16 midpoint steps, five group folds, and 10,000 group-bootstrap draws. The repository separates dataset construction, causal truth, cheap READ, evaluation, and plotting into explicit stages. Reported headline values are recorded in `results/PROVENANCE_pre_refactor.json`; the cleaned release reruns the full pipeline and compares its post-refactor provenance against that frozen snapshot with a $10^{-3}$ numerical tolerance. Failed and superseded experiments are retained in `experiments/` rather than deleted.

The work uses synthetic factual prompt templates and model outputs; it does not involve human participants or private personal data. The relevant ethical risk is overconfidence. A concept-use detector may be attractive for model monitoring, but this experiment does not establish safety performance, deployment thresholds, or false-positive rates under distribution shift. Reporting the failed graded test, the failed instruments, and the limited scope is therefore part of the safety case for the artifact rather than an incidental caveat.

We attribute J-Lens/J-Space and their reference implementation to Anthropic. Our contribution begins with the READ estimators, the firewall, and their causal validation. Model weights, upstream code, and downloaded artifacts remain subject to their respective licenses; the J-Lens reference code is Apache-2.0.

## References

1. Wes Gurnee, Nicholas Sofroniew, Adam Pearce, et al. [*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). Anthropic, Transformer Circuits Thread, 2026.
2. Anthropic. [*Jacobian Lens reference implementation*](https://github.com/anthropics/jacobian-lens), commit `581d398`, 2026.
3. Mukund Sundararajan, Ankur Taly, and Qiqi Yan. [*Axiomatic Attribution for Deep Networks*](https://arxiv.org/abs/1703.01365), 2017.
4. An Yang, Baosong Yang, Beichen Zhang, et al. [*Qwen2.5 Technical Report*](https://arxiv.org/abs/2412.15115), 2024.
5. Anthropic. [*External commentary accompanying the global-workspace report*](https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf), especially Neel Nanda's comments on reliability and false-positive evidence, pp. 41--42, 2026.
