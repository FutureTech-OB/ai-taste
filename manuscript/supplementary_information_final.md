# Supplementary Information

**Fine-tuned AI learns tacit scientific judgment from institutional traces**

---

## Table of Contents

**Supplementary Methods**

- SM1: Supervised Fine-Tuning Hyperparameters and Training Corpus
- SM2: Reinforcement Learning Objective and Reward
- SM3: RL Infrastructure and Resource Consumption
- SM4: Research-Idea Extraction
- SM5: Journal-to-Tier Mapping
- SM6: Evaluation Prompts and Zero-Shot Design Rationale
- SM7: Human Study Design
- SM8: Label-Noise Ceiling Analysis

**Supplementary Tables**

- ST1: Prompt-Sensitivity Summary (Including Gemini 3.1)
- ST2: Benchmark Tier Balance
- ST3: Benchmark Domain Coverage
- ST4: Pairwise Discrimination by Tier Distance
- ST5: Cost and Inference Regime Comparison
- ST6: Core Per-Class Metrics
- ST7: All Pairwise SFT Ensemble Combinations
- ST8: Human Panel Composition and Descriptives
- ST9: Label Normalization
- ST10: Filtering Sensitivity
- ST11: Pairwise McNemar Test Compendium
- ST12: Individual Expert Accuracy Distribution
- ST13: Monte Carlo Matched-N Analysis
- ST14: Prior-Exposure Descriptive Summary
- ST15: Agreement and Consensus Diagnostics
- ST16: Model Inventory and Access Window

**Supplementary Figures**

- SF1: Prediction Distribution Comparison
- SF2: Expert Individual Accuracy Distribution
- SF3: Junior Monte Carlo Subsampling Curve
- SF4: Flagship Collapse-Metric Landscape
- SF5: AI-Human Error Complementarity
- SF6: Human Confusion Matrices

---

## Supplementary Methods

### Supplementary Methods 1 (SM1): Supervised Fine-Tuning Hyperparameters and Training Corpus

**Table SM1. SFT training configuration.**

| Parameter | Qwen3-4B-Instruct | Qwen3-30B-A3B-Instruct | GPT-4.1-nano | GPT-4.1 |
|-----------|-------------------|------------------------|--------------|---------|
| Architecture | Dense transformer | Mixture-of-experts (30B total, 3B active) | Proprietary transformer (undisclosed) | Proprietary transformer (undisclosed) |
| Training framework | TRL (Hugging Face) | TRL (Hugging Face) | OpenAI fine-tuning API | OpenAI fine-tuning API |
| Training location | Local GPU cluster | Local GPU cluster | OpenAI cloud | OpenAI cloud |
| Learning rate | 1e-4 | 2e-5 | API-managed-default | API-managed-default |
| Batch size | 64 | 64 | 32 | 32 |
| Epochs | 3 | 3 | 3 | 3 |
| Optimizer | AdamW | AdamW | API-managed | API-managed |
| Hardware | 1 x A100 | 8 x A100 | Provider-managed | Provider-managed |
| Training duration | ~1 hour | ~1 hour | ~2 hours | ~2 hours |

All four models used the same curated training corpus and frozen instruction scaffold.

**Training corpus.** The corpus comprised 3,994 processed research-idea instances paired with tier labels, derived from organisational behaviour and management source articles drawn from the predefined 19-journal source universe described in SM5, published prior to mid-2025. Articles were assigned tier labels via the deterministic journal-to-tier mapping described in SM5. The distribution was approximately balanced across tiers; articles with ambiguous venue assignment or unclear publication status were excluded during curation to reduce avoidable label noise. Each training example consisted of the frozen evaluation prompt (SM6, Prompt 1) wrapping a research-question-with-context pitch extracted via the SM4 pipeline, with a single tier-label token (one of four: exceptional, strong, fair, limited) as the completion target. Loss was computed on label tokens only; input tokens were masked from gradient updates, forcing the model to learn exclusively the mapping from research content to quality tier rather than memorising prompt structure. The 120 benchmark idea pitches, derived from held-out source articles, were fully disjoint from the training corpus.

---

### Supplementary Methods 2 (SM2): Reinforcement Learning Objective and Reward

RL checkpoints were trained for Qwen3-4B and Qwen3-32B using a modified GRPO-style objective with asymmetric clipping and token-level normalization. The design tests whether explicit chain-of-thought policy optimization can recover the same evaluative signal captured by direct supervised alignment (SM1).

#### Training objective

$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left(r_{i,t}(\theta)\hat{A}_i,\ \text{clip}\left(r_{i,t}(\theta),\ 1-\varepsilon,\ 1+\varepsilon+\varepsilon_{\text{higher}}\right)\hat{A}_i\right)
$$

where $G$ is group size, $o_i$ is sampled output $i$, and

$$
r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\text{ref}}(o_{i,t}\mid q,o_{i,<t})}
$$

is the token-level importance ratio.

#### Reward design

We used a consistency-gated ordinal reward:

$$
R(o_i, y)=\mathbf{1}\left[\hat{y}_i^{\text{label}}=\hat{y}_i^{\text{reasoning}}\right]\cdot r(\hat{y}_i^{\text{label}}, y)
$$

with

$$
r(\hat{y}, y)=
\begin{cases}
1, & |\hat{y}-y|=0 \\
0.3, & |\hat{y}-y|=1 \\
0, & |\hat{y}-y|\ge 2
\end{cases}
$$

The gate suppresses reward for reasoning-label mismatch; partial credit preserves ordinal structure.

#### Advantage normalization

$$
\hat{A}_i=\frac{R_i-\mu_G}{\sigma_G+\epsilon}
$$

with per-group mean $\mu_G$ and standard deviation $\sigma_G$.

#### Privileged GRPO sampling strategy

To address advantage vanishing (where all sampled outputs for a given prompt receive the same reward, yielding zero advantage and no policy gradient signal), we developed a sample-wise adaptive sampling strategy. Prior to each rollout, the current model performs $K$ diagnostic rollouts on each prompt $x$ to estimate per-sample accuracy. Training mode is then assigned per-sample: samples with accuracy below threshold $\tau$ are routed to Privileged GRPO mode, where a hindsight hint grounded against the ground-truth label is prepended to the prompt. This mechanism corrects distributional bias in the base model's rollouts, ensuring that the training group contains sufficient reward contrast across the full label space for stable policy gradient updates. The exact run-level values of $\tau$ and $K$ were fixed in trainer-side configuration files for each RL experiment; those internal configuration files are outside this manuscript package.

---

### Supplementary Methods 3 (SM3): RL Infrastructure and Resource Consumption

RL training was conducted on a cluster of 8 x A100 GPUs. Both Qwen3-4B-Thinking and Qwen3-32B were trained for over a week, with sustained GPU utilization exceeding 95% throughout. Training was built on and extended the fully asynchronous AgentRL framework open-sourced by Tsinghua University, with customizations to support our reward design and data pipeline; the full training infrastructure will be released alongside the model weights and code.

**Table SM3. RL training hyperparameters.**

| Parameter | Qwen3-4B-Thinking | Qwen3-32B |
|-----------|-------------------|------------------------|
| Learning rate | 5×10⁻⁵ | 1×10⁻⁵ |
| Batch size | 32 | 32 |
| Adam betas | 0.9, 0.99 | 0.9, 0.99 |
| Weight decay | 0.05 | 0.05 |
| Optimizer | AdamW | AdamW |
| Clipping $\varepsilon$ | 0.2 | 0.2 |
| Asymmetric bonus $\varepsilon_{\text{higher}}$ | 0.1 | 0.1 |
| Max gradient norm | 1.0 | 1.0 |
| Attention implementation | FlashAttention-2 | FlashAttention-2 |
| Parallelism | DDP | FSDP |
| Precision | bf16 mixed precision | bf16 mixed precision |
| Inference backend | SGLang | SGLang |
| Hardware | 8×A100 | 8×A100 |
| Training duration | <1 week | ~1 week |

---

### Supplementary Methods 4 (SM4): Research-Idea Extraction

To standardise inputs across all training and evaluation conditions, we used Qwen3-235B-A22B-Instruct (Alibaba) to extract structured research idea descriptions from each article. The extraction prompt instructed the model to produce a structured description including: (1) the core research question, (2) theoretical motivation, (3) methodological approach, and (4) expected contribution, while omitting methods details, empirical results, publication venue, and author identities.

**Model selection.** The extraction pipeline was validated by comparing outputs across multiple large language models, including Claude Sonnet 3.5 (Anthropic), Qwen3-235B-A22B-Instruct, Qwen3-32B-Instruct (Alibaba), and others. No substantive differences were observed across models. Qwen3-235B-A22B-Instruct was selected as the production extraction model on the basis of human quality judgments of output coherence and completeness.

**Extraction prompt.** The extraction prompt instructed the model to act as an objective research paper analyser, extracting research questions and core elements without interpretation or embellishment. The prompt specified five output versions in JSON format:

1. **CORE_RQ_SHORT** (40--60 words): Distilled essential research question(s).
2. **RQ_WITH_CONTEXT** (120--150 words): Research question with enough context for expert evaluation, including the phenomenon, gap, question, approach, and claimed contribution.
3. **GAP_FOCUSED** (100--130 words): What is known, what remains unknown, and how the study addresses it.
4. **THEORY_AND_MODEL** (100--130 words): Theoretical framework, key variables and relationships, and theoretical contribution.
5. **CONTRIBUTION_FOCUSED** (80--100 words): Theoretical, empirical/methodological, and practical contributions as claimed by the authors.

The main benchmark uses the RQ_WITH_CONTEXT format. Critical extraction rules required focusing on the abstract, introduction, and theoretical development sections; using the authors' exact terminology for key constructs; preserving the level of theoretical sophistication in the original; and avoiding any addition of theoretical connections, persuasive hooks, or inferred contributions not explicitly stated.

#### Verbatim extraction prompt (exact text)

````text
# ROLE
You are an objective research paper analyzer. Your task is to extract and present research questions and core elements from academic papers WITHOUT interpretation, embellishment, or improvement.

# CRITICAL PRINCIPLE: OBJECTIVITY OVER PERSUASIVENESS
- Present the paper EXACTLY as written by the authors
- Do NOT add theoretical sophistication if it's not there
- Do NOT create compelling hooks if the original lacks them
- Do NOT infer contributions beyond what authors explicitly state
- Do NOT improve weak framing - describe it as presented
- If the idea seems underdeveloped in the original, your summary should reflect that

Your goal: Represent the research proposal exactly as the authors present it-the way a doctoral student would pitch their idea to an advisor. Convey their thinking faithfully, including any lack of polish or theoretical sophistication, so the professor can understand and evaluate the original idea.

---

# OUTPUT STRUCTURE
Generate exactly 5 versions in JSON format:

---

## VERSION 1: CORE_RQ_SHORT
**Purpose:** Distill the essential research question(s)
**Word count:** 40-60 words (2-3 sentences maximum)
**Structure:**
- Sentence 1: The phenomenon or behavior under study
- Sentence 2: The specific question or what's being tested
- [Optional Sentence 3: The key boundary condition or mechanism if central to RQ]

---

## VERSION 2: RQ_WITH_CONTEXT
**Purpose:** Add just enough context for a professor to evaluate the idea's merit
**Word count:** 120-150 words (1 paragraph)
**Structure:**
- What phenomenon/problem (1-2 sentences)
- What's missing/unclear in existing research - the gap (2-3 sentences)
- The research question (1-2 sentences)
- The approach/framework used (1 sentence)
- Key claimed contribution (1 sentence)

---

## VERSION 3: GAP_FOCUSED
**Purpose:** Emphasize what's unknown and how this study addresses it
**Word count:** 100-130 words (1 paragraph)
**Structure:**
- What existing research has established (2 sentences)
- What remains unknown/unresolved (2-3 sentences)
- How this study addresses the gap/extends the prior research/challenges the understanding (2 sentences)
- Expected insight (1 sentence)

---

## VERSION 4: THEORY_AND_MODEL
**Purpose:** Describe the theoretical framework and research model
**Word count:** 100-130 words (1 paragraph)
**Structure:**
- Core theoretical lens/framework (1-2 sentences)
- How theory is applied to the phenomenon (2 sentences)
- Key variables and relationships (2-3 sentences)
- Theoretical contribution claimed (1 sentence)

---

## VERSION 5: CONTRIBUTION_FOCUSED
**Purpose:** Extract what the authors claim as their contributions
**Word count:** 80-100 words
**Structure:**
- Primary theoretical contribution (1-2 sentences)
- Empirical/methodological contribution if claimed (1 sentence)
- Practical contribution if claimed (1 sentence)
- How it advances the literature (1-2 sentences)

---

# EXTRACTION RULES

## Where to Look:
Focus on the **front-end** of the paper:
- **Abstract**
- **Introduction** (entire section - contains RQ, gap, motivation)
- **Theoretical Development** (theory and hypotheses framing)

Most information needed is in these sections. Do NOT need to read results/discussion unless contribution statements are unclear.

## What to Extract:
1. **Research Questions:** Usually in abstract's and introduction
2. **Gaps/problematization:** mostly in introduction and sometimes in theoretical development
3. **Theory:** introduced in introduction and often elaborated in theory development sections
4. **Contributions:** Abstract, introduction's end

## What to Avoid:
❌ Adding your own theoretical connections
❌ Improving vague or weak language
❌ Creating persuasive hooks not in the original
❌ Inferring contributions not explicitly stated
❌ Making gaps sound more compelling than presented

## Language Rules:
✅ Use the authors' exact terminology for key constructs
✅ Preserve the level of theoretical sophistication in the original
✅ Match the certainty level (e.g., "explores" vs. "demonstrates")
✅ If authors use simple language, you use simple language

---

# JSON OUTPUT FORMAT

Output the following JSON structure with all 5 versions:
```json
{
"core_rq_short": "string",
"rq_with_context": "string",
"gap_focused": "string",
"theory_and_model": "string",
"contribution_focused": "string"
}
```
````

#### Worked extraction examples (RQ_WITH_CONTEXT)

Example 1  
Input article title: *Game Over or Game Changer? The Impact of Applicants' Gaming Skills on Their Hirability*  
Extraction output (RQ_WITH_CONTEXT):

> The increasing digitalization of work has raised interest in nontraditional skills such as those developed through video gaming, which may include strategic thinking, teamwork, and digital fluency. However, hiring managers may hold negative stereotypes about gamers, viewing them as lazy or socially isolated, potentially disadvantaging applicants who list gaming on their resumes. While research shows gaming can enhance job-relevant cognitive abilities, there is no empirical evidence on how such skills are perceived during resume screening. Using a 2 (ECA: gaming vs. volleyball) × 2 (proficiency: neutral vs. high) experimental design, this study investigates whether listing gaming as an ECA affects applicant evaluations compared to traditional team sports. The authors apply signaling theory to understand how ECAs serve as cues about unobservable applicant traits. The primary contribution is initial evidence on the perception of gaming skills in personnel selection, highlighting a disconnect between potential skill benefits and actual hiring biases.

Example 2  
Input article title: *Good intentions, bad outcomes: how and when family motivation leads to work-family conflict*  
Extraction output (RQ_WITH_CONTEXT):

> Family motivation is widely seen as a positive driver of work performance, yet its effects on employees' family lives remain underexplored. While prior research highlights benefits in the work domain, little is known about potential downsides for family well-being. This study addresses this gap by investigating whether high family motivation, despite good intentions, can lead to work-family conflict (WFC) and negative spousal interactions due to excessive work effort depleting personal resources. Drawing on resource drain theory, the authors propose that FSSBs from supervisors serve as external resources that may mitigate this drain. Using a three-wave dyadic survey design with employee-partner data, the study tests a mediated moderation model. The key contribution lies in revealing the 'dark side' of family motivation and identifying organizational support as a boundary condition.

---

### Supplementary Methods 5 (SM5): Journal-to-Tier Mapping

Ground-truth tier labels were assigned a priori via a deterministic mapping from publication venue to one of four institutional prestige tiers. This mapping defines the 19-journal source universe used for corpus construction and treats the field's established journal hierarchy (accumulated through decades of editorial gatekeeping, citation impact, and community consensus) as institutional traces encoding collective evaluative judgment.

**Table SM5. Journal-to-tier mapping for the 19-journal source universe.**

| Tier | Journals | Rationale |
|------|----------|-----------|
| Exceptional | Academy of Management Journal, Academy of Management Review, Administrative Science Quarterly, Journal of Applied Psychology, Organization Science, Strategic Management Journal | Elite field-defining outlets in management and organizational research; highest selectivity and institutional prestige |
| Strong | Journal of Management, Organizational Behavior and Human Decision Processes, Personnel Psychology | Top-tier specialty outlets with strong citation impact and high prestige, typically just below the elite general-management tier |
| Fair | Human Resource Management, Human Relations, Journal of Management Studies, Journal of Organizational Behavior, Leadership Quarterly | Well-regarded field journals with rigorous peer review and substantial disciplinary visibility, but lower prestige than the strong tier |
| Limited | Group & Organization Management, Journal of Business and Psychology, Journal of Managerial Psychology, Journal of Organizational Behavior Management, Journal of Personnel Psychology | More specialized or lower-prestige outlets with narrower scope and lower institutional status in the field hierarchy |

The tier mapping reflects field-wide consensus as codified in institutional tenure and promotion standards. Exceptional-tier journals are universally recognised as top-tier outlets across major research universities and are consistently counted toward tenure at leading institutions. Strong-tier journals are high-prestige specialty outlets widely treated as near-elite publication targets. Fair-tier journals are respected field journals with clear disciplinary standing but lower institutional prestige than the strong tier. Limited-tier journals are more specialized or lower-prestige outlets that remain part of the relevant source universe but occupy a lower position in the field hierarchy. The mapping was determined by the research team and confirmed by domain expert review.

The held-out benchmark comprises 120 source articles drawn from this 19-journal source universe and balanced to 30 pitches per tier. Benchmark articles are therefore an article-level subset of the source universe rather than the basis for defining the tier framework. Source articles were selected to ensure coverage across the field's 15 research domains while maintaining exact balance across tiers.

---

### Supplementary Methods 6 (SM6): Evaluation Prompts and Zero-Shot Design Rationale

#### Three prompt variants

We designed three candidate evaluation prompts varying in structure, specificity, and anchoring strategy. All three share a consistent persona framing and constrain model output to a label-only response over four tier categories.

**Prompt 1 (Expert prompt; selected as primary):** A detailed prompt with structured tier definitions emphasising originality and usefulness, with behavioural anchors for each tier. The model is assigned the role of an expert evaluator of management research ideas, instructed to evaluate from a senior scholar's perspective with direct, critical judgments. Two evaluation dimensions are defined in detail:

- *Novelty*: Whether the research idea challenges existing assumptions, reveals something genuinely surprising, or provides cognitive disruption that fundamentally changes understanding of relationships or phenomena. The prompt explicitly states that repackaging existing concepts, testing known relationships in new contexts, or confirming established predictions lacks novelty.
- *Usefulness*: Whether the research idea addresses problems that matter, with broad implications for multiple stakeholders, resolving long-standing theoretical debates or providing insights that meaningfully improve organisational practices. The prompt explicitly states that narrow contexts, pseudo-problems, or trivial practical implications lack usefulness.

Four tiers are defined with behavioural anchors: Exceptional (strong novelty + strong usefulness; field-reshaping potential; most prestigious journals), Strong (clear strength in one dimension with the other reasonably developed; meaningful contributions; near-top-tier journals), Fair (incremental contributions with modest novelty or usefulness; mid-level journals), Limited (lacks both novelty and usefulness; lower-tier journals). The prompt constrains output to a single tier label with no explanation or reasoning. An explicit instruction prohibits the use of search capabilities. Full prompt text is available in the code repository.

**Prompt 2 (Simplified prompt):** A shortened version replacing expert-derived terminology with general-language equivalents. Tier definitions are compressed into single-sentence descriptions without behavioural anchors: Exceptional = "field-defining work, recognised across disciplines"; Strong = "meaningful contribution, clearly advances theory or method"; Fair = "solid but incremental, recognised mainly by specialists"; Limited = "weak contribution, obvious findings or narrow scope." No detailed criteria for novelty or usefulness are provided. Full prompt text is available in the code repository.

**Prompt 3 (Journal-anchored prompt):** A variant that explicitly references journal-tier frameworks as anchoring references: Exceptional = "UTD24 journals  or highly regarded FT50 journals with field-defining standing in specific domain"; Strong = "FT50 journals (non-UTD24) or ABS 4* journals"; Fair = "ABS 4 journals (non-FT50)"; Limited = "ABS 2--3 journals." This prompt anchors quality levels to institutional prestige indicators rather than abstract quality dimensions. Full prompt text is available in the code repository.

#### Verbatim prompt texts (exact strings)

**Prompt 1 (expert rubric)**

```text
# ROLE
You are an expert evaluator of management research ideas. Your task is to evaluate from a senior scholar's perspective: be direct and critical, give clear judgments based on novelty and usefulness to classify research ideas into appropriate publication potential tiers.

---
# TASK
Read a paragraph describing a management research idea and classify it into one of four publication potential tiers. Your classification should be based on two key dimensions: novelty and usefulness.

Output ONLY the tier notation with NO explanation or reasoning.

---
# EVALUATION CRITERIA

## Novelty
Novelty reflects whether the research idea challenges existing assumptions or reveals something genuinely surprising. Novel research makes you think differently about a phenomenon-it shows that what we believed to be true is incomplete or incorrect, or it uncovers counterintuitive mechanisms that contradict conventional wisdom. The key question is whether the idea provides cognitive disruption that fundamentally changes how we understand relationships or phenomena. Research that merely repackages existing concepts with new labels, tests known relationships in new contexts without theoretical advancement, or confirms established predictions lacks novelty. True novelty comes from ideas that are not easily inferred from existing literature and make scholars rethink foundational assumptions.

## Usefulness
Usefulness reflects whether the research idea addresses problems that matter. Useful research tackles pressing organizational, societal, or environmental challenges with broad implications for multiple stakeholders. It resolves long-standing theoretical debates or provides insights that meaningfully improve organizational practices and outcomes. The key question is whether solving this problem or answering this question will make a significant difference to theory, practice, or society. Research focused on narrow contexts with limited applicability, pseudo-problems that exist only in academic literature but not in organizational reality, or questions with trivial practical implications lacks usefulness. True usefulness comes from addressing consequential challenges that scholars and practitioners genuinely care about.

---
# CLASSIFICATION TIERS

## Tier 4: Exceptional (Publication Potential)
Research that demonstrates both strong novelty and strong usefulness. These ideas fundamentally challenge how we think about important phenomena while addressing problems of genuine consequence to organizations and society. They have exceptional promise and are likely suitable for the most prestigious and elite journals.

## Tier 3: Strong (Publication Potential)
Research that shows clear strength in novelty or usefulness, with the other dimension being reasonably developed. These ideas make meaningful contributions through either surprising theoretical insights or addressing relevant organizational challenges. They have strong potential to be published in near-top-tier journals.

## Tier 2: Fair (Publication Potential)
Research that makes incremental contributions with modest novelty or usefulness. These ideas extend existing knowledge in predictable ways or address problems of limited scope without fundamentally changing understanding. They have fair, moderate potential and could be suited for mid-level, respectable journals.

## Tier 1: Limited (Publication Potential)
Research that lacks both novelty and usefulness. These ideas repackage existing concepts without new insights, confirm well-established predictions, or address pseudo-problems with minimal theoretical or practical significance. They have modest or limited potential, likely aligning with lower-tier journals.

# OUTPUT FORMAT

---
# IMPORTANT
- Do not use search capabilities to look up information about this idea

---
Respond with EXACTLY ONE of these four notations:

- Exceptional
- Strong
- Fair
- Limited

Output only the tier notation in your final answer.
```

**Prompt 2 (simplified rubric)**

```text
You are an expert in management research. Read the research idea below and estimate the likely publication tier based on its scholarly contribution.

- Exceptional: Field-defining work. Would be recognized across disciplines as a major advance. Likely to be widely cited and reshape how researchers think about the topic.
- Strong: Meaningful contribution within the field. Clearly advances theory or method in a non-trivial way. Would be well-regarded by domain experts.
- Fair: Solid but incremental. Competent execution with limited novelty. Recognized mainly by specialists in the same narrow area.
- Limited: Weak contribution. Findings are obvious, scope is too narrow, or methodological issues undermine the work.

---
# IMPORTANT
- Do not use search capabilities to look up information about this idea

---
# OUTPUT FORMAT

Respond with EXACTLY ONE of these four notations:

- Exceptional
- Strong
- Fair
- Limited

Output only the tier notation in your final answer.
```

**Prompt 3 (journal-anchored rubric)**

```text
You are an expert in management research with deep knowledge of academic publishing standards across top-tier journals.

---
# TASK
Read a paragraph describing a management research idea and classify it into one of four journal tiers based on its likely publication venue. Your classification should reflect where work of this quality and contribution level would most likely be published.

- Exceptional: UTD24 journals or highly regarded FT50 journals with field-defining standing in their domain - paradigm-shifting work, highest selectivity, field-redefining impact
- Strong: FT50 journals (non-UTD24) or ABS 4* journals - substantial contribution, A-level quality, high methodological rigor
- Fair: ABS 4 journals (non-FT50) - solid contribution with clear theoretical grounding, competent execution but limited novelty
- Limited: ABS 2-3 journals - incremental findings, narrower scope, or moderate methodological rigor

---
# IMPORTANT
- Do not use search capabilities to look up information about this idea

---
# OUTPUT FORMAT

Respond with EXACTLY ONE of these four notations:

- Exceptional
- Strong
- Fair
- Limited

Output only the tier notation in your final answer.
```

#### Prompt selection rationale

We selected Prompt 1 as the fixed evaluation instruction on the basis that it yielded the highest frontier model accuracy among the three variants. This choice is deliberately conservative with respect to frontier model performance: because the prompt was optimised for frontier models, any accuracy advantage of SFT models over frontier models under this prompt represents a lower bound on the true effect. The same prompt was used identically for SFT training (as the instruction template wrapping each training example) and for all evaluation conditions.

#### Zero-shot design rationale

All evaluations used zero-shot prompting (no few-shot exemplars) for three reasons.

First, frontier models have already encountered extensive academic literature during pre-training, including papers and editorial commentary that implicitly encode tier-level evaluative norms; few-shot exemplars would redundantly re-introduce information these models have in some form already processed, making the marginal contribution of exemplars uninterpretable.

Second, because ground-truth labels are derived from publication outcomes rather than direct assessments of idea quality, any selected exemplar carries noise from confounding factors such as execution quality, writing craft, and reviewer fit. Providing such exemplars risks anchoring models to misleading features, teaching them to recognise correlates of publication success that are orthogonal to the research idea dimension we aim to evaluate.

Third, zero-shot prompting ensures that all models (frontier, base, SFT, and human raters) are evaluated under identical conditions. SFT models have already seen labelled examples during training; providing additional labelled exemplars at inference would give frontier models a compensatory signal unavailable to human raters in our experiment, undermining the fairness of the comparison.

#### Sensitivity analysis

Extended Data Fig. 1 reports cross-model prompt-sensitivity results for models with complete three-prompt coverage (Simple, Journal, Expert). SFT models do not currently have complete three-prompt coverage in the prompt-sensitivity files, so they are not included in this strict three-prompt comparison panel.

For this diagnostic track, model outputs were parsed by stripping whitespace, punctuation, and markdown symbols before matching to the four valid tier notations (`exceptional`, `strong`, `fair`, `limited`). Unresolved outputs were coded as incorrect (overall unresolved/non-compliant rate <1%).

#### Output compliance and cohort cleaning

Gemini 3.1 Pro showed anomalous output behaviour in the evaluation pipeline, including occasional full-paper-text outputs and a higher formatting non-compliance rate (~2.3% vs <0.5% for all other evaluated models). Because these outputs could reflect built-in search and retrieval behaviour rather than zero-shot evaluation, Gemini 3.1 is retained in the conservative frontier cohort with explicit contamination-risk caveat in interpretation.

---

### Supplementary Methods 7 (SM7): Human Study Design

This section provides procedural details that supplement the Methods description of the human evaluation protocol. For panel composition, recruitment, package design, cohort structure, and survey administration overview, see Methods ("Human evaluation protocol").

#### Institutional review

The study was approved by the institutional review board (Project No. THU-04-2026-0034). All participants provided written informed consent prior to participation. Raters were not informed of the study's hypotheses or of the AI evaluation component. Experts participated voluntarily without financial compensation; all were promised access to the research results upon publication. Junior scholars were compensated with 100 RMB and/or access to a research tool developed by the research team. Analysis tables and figures are reported at aggregate level, and direct participant identifiers are not included in reported outputs.

#### Full survey instrument

For each benchmark pitch, raters were shown the research-question pitch alongside the evaluation criteria and responded to four items with the following exact wording and scales:

1. **Prior exposure**: "Had you encountered this research idea or its source paper before?" Response options: Yes / No.
2. **Quality rating**: "Based on the evaluation criteria, how would you rate the quality of this research idea?" Response options: Top / Top- / Good / Fair (mapped to exceptional / strong / fair / limited in all analyses).
3. **Confidence**: "How confident are you in your rating?" Response options on a 5-point Likert scale: 1 = "Not at all confident", 2 = "Slightly confident", 3 = "Moderately confident", 4 = "Very confident", 5 = "Extremely confident".
4. **Domain familiarity**: "How familiar are you with this research area?" Response options on a 5-point Likert scale: 1 = "Not at all familiar", 2 = "Slightly familiar", 3 = "Moderately familiar", 4 = "Very familiar", 5 = "Extremely familiar".

#### Completion duration

Median expert completion time was 923 seconds (~15.4 minutes) for 8 pitches. Median junior completion time was 2,534 seconds (~42.2 minutes) for approximately 14.5 pitches. Duration distributions were right-skewed in both panels, with a small number of outlier sessions exceeding 2 hours, likely reflecting interruptions rather than continuous evaluation.

#### Background data collection

For junior scholars, demographic and academic background information was collected:
- Gender
- University and department
- Research direction/area
- Doctoral year (PhD1 through PhD5+, or postdoc)
- Number of published papers
- Peer-review experience (yes/no, number of reviews)
- AI tool familiarity (1--5 scale)

Background data was matched to ratings for 104 of 108 old-cohort juniors (96.3%) and 52 of 67 new-cohort juniors (77.6%). Four old-cohort and 15 new-cohort juniors could not be matched due to name discrepancies between signup records and survey responses.

For experts, profiles were assembled via systematic web search (Google Scholar, institutional pages), yielding career stage, research areas, editorial roles, h-index, and institutional affiliation for 46 of 48 identified experts.

#### Filtering criteria

Experts were recruited through personal and professional networks via one-on-one direct contact. Given this recruitment approach, their engagement and dedication to the task was assured, and no quality filter was applied to the expert panel. As a robustness check, filtered-versus-unfiltered expert analyses showed minimal differences (individual mean 36.2% vs. 36.2%; majority vote 41.6% vs. 39.7%; Supplementary Table ST10). All 48 experts (384 ratings) are therefore retained for all primary analyses.

Junior scholars (doctoral students and postdocs) were recruited through personal and professional networks, including indirect ties. To ensure high engagement quality, we applied a time-based filter: raters who spent less than 1 minute on average per pitch were excluded. This filter showed a marginally significant effect on accuracy (25.3% vs. 31.7%, $P$ = 0.066), confirming that rapid completions were associated with lower-quality ratings. The filtered panel (174 raters, 2,530 ratings) is used in all primary analyses; unfiltered results (189 raters, 2,730 ratings) are reported for comparison in Supplementary Table ST10.

Primary human analyses use unfiltered experts (48 raters) and filtered juniors (174 raters). Filtered-versus-unfiltered sensitivity is reported in Supplementary Table ST10.

---

### Supplementary Methods 8 (SM8): Label-Noise Ceiling Analysis

Publication outcomes are not solely determined by research idea quality. Execution fidelity, writing quality, reviewer--manuscript fit, and editorial discretion all contribute to final publication decisions, while our standardised inputs capture only the idea dimension. This gap between input features and outcome labels introduces inherent noise that places a theoretical ceiling on achievable classification accuracy: even a perfect evaluator of research idea quality would not achieve perfect agreement with publication outcomes.

Several factors contribute to this noise floor:

1. **Execution gap.** A strong research idea may be published in a lower-tier journal due to poor execution, and a modest idea may reach a top-tier journal through exceptional methods and writing. Our inputs strip execution information, so the model cannot account for this variance.

2. **Reviewer--manuscript fit.** Publication decisions depend partly on the match between reviewer expertise and the manuscript's topic, which introduces stochastic variation unrelated to idea quality.

3. **Editorial discretion.** Editors exercise judgment that reflects strategic considerations (journal scope, topic balance, timeliness) beyond pure quality assessment.

4. **Tier boundary ambiguity.** Some journals sit at the boundary between adjacent tiers. While our mapping is deterministic, the underlying quality distribution is continuous, creating inherent disagreement for articles near tier boundaries.

Observed accuracies should therefore be interpreted relative to this ceiling rather than against a 100% standard. Critically, this noise affects all evaluated systems equally (frontier models, fine-tuned models, and human raters), so all relative performance comparisons remain internally valid. The noise floor also explains why even the best-performing system (SFT ensemble at 60.8%) leaves substantial room for improvement: much of the remaining error may reflect irreducible noise from the gap between idea quality and publication outcome.

---

## Supplementary Tables

### Supplementary Table 1 (ST1): Prompt-Sensitivity Summary (Including Gemini 3.1)

**Table ST1. Fixed-denominator prompt-variant results (`n = 120`). Unresolved outputs count as incorrect.**

| Prompt variant | Accuracy (%) | Predicted Exceptional (%) | Predicted Strong (%) | Predicted Fair (%) | Predicted Limited (%) | Unresolved (%) |
|----------------|--------------|---------------------------|----------------------|--------------------|-----------------------|----------------|
| Expert | 37.5 | 19.2 | 25.0 | 46.7 | 5.0 | 4.2 |
| Simplified | 30.0 | 0.0 | 77.5 | 21.7 | 0.0 | 0.8 |
| Journal-anchored | 38.3 | 46.7 | 16.7 | 8.3 | 18.3 | 10.0 |

---

### Supplementary Table 2 (ST2): Benchmark Tier Balance

**Table ST2. Evaluation benchmark tier distribution.**

| Tier | N pitches | Share |
|------|------------|-------|
| Exceptional | 30 | 25.0% |
| Strong | 30 | 25.0% |
| Fair | 30 | 25.0% |
| Limited | 30 | 25.0% |

The benchmark is exactly balanced by construction.

---

### Supplementary Table 3 (ST3): Benchmark Domain Coverage

**Table ST3. Domain distribution across 120 benchmark pitches.** Pitches may be assigned to multiple domains; column totals exceed 120.

| Domain | N |
|--------|---|
| Leadership/Managers | 27 |
| AI and Technology | 25 |
| Employee Behavior and Attitudes | 23 |
| Social Psychology/Interpersonal | 22 |
| Teams and Organizations | 22 |
| Diversity and Equity | 17 |
| Performance and Outcomes | 16 |
| Other Management Research | 16 |
| Innovation and Entrepreneurship | 14 |
| Knowledge and Learning | 10 |
| Human Resource Management | 10 |
| Ethics and Morality | 7 |
| Career Development | 3 |
| Work Stress and Health | 3 |
| Strategy and Decision-Making | 2 |

---

### Supplementary Table 4 (ST4): Pairwise Discrimination by Tier Distance

**Table ST4. Pairwise head-to-head accuracy (label-free task).**

| Model | Distance 1 (adjacent) | Distance 2 | Distance 3 | Weighted overall |
|-------|------------------------|------------|------------|------------------|
| GPT-4.1 (baseline) | 69.33% (104/150) | 79.00% (79/100) | 90.00% (45/50) | 76.00% (228/300) |
| GPT-5.2 High | 69.33% (104/150) | 85.00% (85/100) | **94.00% (47/50)** | 78.67% (236/300) |
| Gemini 3.1 Pro | 68.67% (103/150) | 86.00% (86/100) | 86.00% (43/50) | 77.33% (232/300) |
| SFT GPT-4.1 | **78.67% (118/150)** | **89.00% (89/100)** | 92.00% (46/50) | **84.33% (253/300)** |

All four released models produced valid predictions on all 300 pairwise items.
Fig. 5 now plots the headline SFT/Gemini 3.1 Pro/GPT-5.2 High/GPT-4.1 baseline pairwise comparison, covering overall weighted accuracy and the two hardest boundaries (`fair_strong`, `strong_exceptional`). Extended Data Fig. 2 retains the six individual pair types and the paired-discordance decomposition for the same plotted subset. On the same 300 shared items, raw unadjusted two-sided exact McNemar tests for SFT GPT-4.1 gave `p = 0.00646` versus Gemini 3.1 Pro, `p = 0.0300` versus GPT-5.2 High, and `p = 0.000621` versus GPT-4.1 baseline.

---

### Supplementary Table 5 (ST5): Cost and Inference Regime Comparison

**Table ST5. Training and inference cost bands by evaluator type.**

| Model class | Training cost/model | Inference cost (per 100 pitches) | Notes |
|-------------|---------------------|-----------------------------------|-------|
| Frontier (thinking) | $0 (API access) | >$10 | 8 samples per pitch; chain-of-thought generation |
| Chat (logp) | $0 (API access) | $0.01--$0.10 | Single-pass log-probability classification |
| SFT: Qwen3-4B | ~1 A100 GPU hour | $0.001 | Log-probability classification |
| SFT: Qwen3-30B-A3B | ~8 A100 GPU hours | $0.01 | Log-probability classification |
| SFT: GPT-4.1-nano | ~$10 (API) | $0.01 | Log-probability classification |
| SFT: GPT-4.1 | ~$200 (API) | $0.10 | Log-probability classification |
| RL checkpoints | Multi-day 8×A100 runs | Higher than log-probability pipelines | Reasoning generation + label extraction |

---

### Supplementary Table 6 (ST6): Core Per-Class Metrics (Non-overlapping with Figure Panels)

**Table ST6. Precision/recall/F1 by tier for key evaluators.**

| Evaluator | Tier | Precision | Recall | F1 |
|-----------|------|-----------|--------|----|
| Best Flagship (Gemini 3.1 Pro) | Exceptional | 0.478 | 0.379 | 0.423 |
| Best Flagship (Gemini 3.1 Pro) | Strong | 0.433 | 0.448 | 0.441 |
| Best Flagship (Gemini 3.1 Pro) | Fair | 0.304 | 0.607 | 0.405 |
| Best Flagship (Gemini 3.1 Pro) | Limited | 0.667 | 0.138 | 0.229 |
| SFT 2-Model Ensemble | Exceptional | 0.632 | 0.800 | 0.706 |
| SFT 2-Model Ensemble | Strong | 0.621 | 0.600 | 0.610 |
| SFT 2-Model Ensemble | Fair | 0.472 | 0.567 | 0.515 |
| SFT 2-Model Ensemble | Limited | 0.824 | 0.467 | 0.596 |
| Expert Majority (unfiltered) | Exceptional | 0.625 | 0.227 | 0.333 |
| Expert Majority (unfiltered) | Strong | 0.371 | 0.591 | 0.456 |
| Expert Majority (unfiltered) | Fair | 0.361 | 0.520 | 0.426 |
| Expert Majority (unfiltered) | Limited | 0.600 | 0.300 | 0.400 |
| Junior Majority (filtered) | Exceptional | 0.667 | 0.333 | 0.444 |
| Junior Majority (filtered) | Strong | 0.312 | 0.385 | 0.345 |
| Junior Majority (filtered) | Fair | 0.347 | 0.654 | 0.453 |
| Junior Majority (filtered) | Limited | 0.700 | 0.259 | 0.378 |

---

### Supplementary Table 7 (ST7): All Pairwise SFT Ensemble Combinations

**Table ST7. Accuracy of all six two-model SFT ensembles (probability averaging).**

| Model 1 | Model 2 | Accuracy (%) |
|---------|---------|--------------|
| GPT-4.1-nano (SFT) | Qwen3-4B (SFT) | **60.8** |
| GPT-4.1 (SFT) | Qwen3-30B-A3B (SFT) | 60.0 |
| GPT-4.1 (SFT) | Qwen3-4B (SFT) | 60.0 |
| GPT-4.1-nano (SFT) | Qwen3-30B-A3B (SFT) | 60.0 |
| GPT-4.1-nano (SFT) | GPT-4.1 (SFT) | 59.2 |
| Qwen3-30B-A3B (SFT) | Qwen3-4B (SFT) | 59.2 |

All six combinations exceed the frontier average benchmark. The public `best_2_model_combo` was selected by accuracy, then macro F1, then canonical model-key order to break ties, which retained GPT-4.1-nano (SFT) + Qwen3-4B (SFT) as the primary pair.

As a supporting temporal-stability check, the matched older-source temporal package compares an older training slice (`2015-2020`) against the matched recent slice (`2021-2025`), with the benchmark itself drawn from post-June-30-2025 publications. On the same benchmark, the older-source GPT-4.1-nano SFT reached 43.3% accuracy and macro F1 0.423, the older-source Qwen3-30B SFT reached 46.7% and 0.460, and the older-trace matched 2-model ensemble reached 47.5% and 0.470, versus 57.5% and 0.573, 55.8% and 0.558, and 60.0% and 0.599 for the corresponding recent-training GPT-4.1-nano, Qwen3-30B, and matched 2-model ensemble. The older-trace ensemble also remained more inflationary than the matched recent ensemble, with lower exceptional-tier precision (46.7% versus 59.0%), lower fair-tier recall (26.7% versus 60.0%), and stronger strong->exceptional confusion (46.7% versus 36.7%), indicating that the institutional signal persists across time but yields weaker tier calibration under the older-source training set.

---

### Supplementary Table 8 (ST8): Human Panel Composition and Descriptives

**Table ST8a. Expert career-stage distribution (N = 48).**

| Career stage | N |
|--------------|---|
| Assistant Professor | 5 |
| Associate Professor | 17 |
| Full Professor | 12 |
| Endowed Chair | 12 |
| Unreported | 2 |

**Table ST8b. Panel-level descriptive summary.**

| Metric | Experts | Juniors |
|--------|---------|---------|
| Number of raters | 48 | 174 |
| Total ratings | 384 | 2,530 |
| Mean ratings per pitch | 3.2 | 21.1 |
| Median completion time (seconds) | 923 | 2,534 |
| Mean confidence (1-5) | 3.50 | 3.46 |
| Mean familiarity (1-5) | 3.15 | 2.81 |

---

### Supplementary Table 9 (ST9): Label Normalization

**Table ST9. Deterministic mapping used before all analyses.**

| Unified tier | Source-article metadata label | Human survey label | Numeric code |
|--------------|------------------------|--------------------|--------------|
| Exceptional | top | Top | 1 |
| Strong | top- | Top- | 2 |
| Fair | good | Good | 3 |
| Limited | fair | Fair | 4 |

Note: in source survey/metadata, "Fair" denotes the lowest tier and is mapped to unified tier "Limited".

---

### Supplementary Table 10 (ST10): Filtering Sensitivity

**Table ST10. Filtered versus unfiltered panel outcomes.**

| Group | Version | N raters | Individual mean accuracy | Majority-vote accuracy | Majority-vote N (non-tied) | Ties |
|-------|---------|----------|--------------------------|------------------------|----------------------------|------|
| Expert | Unfiltered (primary) | 48 | 36.2% | 41.6% | 89 | 31 |
| Expert | Filtered | 39 | 36.2% | 39.7% | 68 | 52 |
| Junior | Unfiltered | 189 | 31.2% | 41.3% | 104 | 16 |
| Junior | Filtered (primary) | 174 | 31.7% | 40.8% | 103 | 17 |

---

### Supplementary Table 11 (ST11): Pairwise McNemar Test Compendium

**Table ST11. Pairwise significance tests for key evaluator comparisons.**

*Note: "Best Frontier" refers to Gemini 3.1 Pro under the conservative frontier protocol. Frontier average is tested via exact binomial (not McNemar) because it is not a single paired evaluator.*

| Comparison (vs SFT 2-Model Ensemble) | N | SFT Acc | Comparator Acc | Delta (pp) | Test | Statistic | p (raw) |
|---------------------------------------|---:|--------:|---------------:|-----------:|------|----------:|--------:|
| Frontier average (11 models) | 120 | 0.6083 | 0.3105 | +29.78 | Exact binomial | --- | 1.74×10⁻¹¹ |
| Best Frontier (Gemini 3.1 Pro) | 115 | 0.6000 | 0.3913 | +20.87 | McNemar | 10.173 | 0.001425 |
| Expert majority (excl. ties) | 89 | 0.6180 | 0.4157 | +20.22 | McNemar | 6.568 | 0.010382 |
| Junior majority (full, excl. ties) | 103 | 0.6117 | 0.4078 | +20.39 | McNemar | 8.889 | 0.002869 |

**Extended pairwise comparisons.**

| Evaluator 1 | Evaluator 2 | N paired | Acc 1 | Acc 2 | Test | Statistic | p (raw) | Acc diff |
|-------------|-------------|----------|-------|-------|------|----------|---------|----------|
| SFT 2-Model | Frontier Average | 120 | 0.6083 | 0.3105 | Exact binomial | --- | 1.74×10⁻¹¹ | +0.2978 |
| SFT 2-Model | Best Frontier (Gemini 3.1 Pro) | 115 | 0.6000 | 0.3913 | McNemar | 10.173 | 0.001425 | +0.2087 |
| SFT 2-Model | Expert Majority | 89 | 0.6180 | 0.4157 | McNemar | 6.568 | 0.010382 | +0.2022 |
| SFT 2-Model | Junior Majority (full) | 103 | 0.6117 | 0.4078 | McNemar | 8.889 | 0.002869 | +0.2039 |

---

### Supplementary Table 12 (ST12): Individual Expert Accuracy Distribution

**Table ST12. Individual expert accuracy distribution (unfiltered panel, N = 48).**

Each of 48 experts evaluated exactly 8 pitches. Individual accuracy ranges from 0/8 (0%; 2 experts) to 8/8 (100%; 1 expert). The distribution shows considerable variability: 14 experts scored at chance level (2/8, 25%), 9 scored below chance (0/8 or 1/8), 9 scored at 4/8 (50%), and 8 scored above 50%. Median accuracy was 3/8 (37.5%).

---

### Supplementary Table 13 (ST13): Monte Carlo Matched-N Analysis

**Table ST13. Junior panel subsampling to expert-sized panels.**

| Metric | Value |
|--------|-------|
| Draws | 5,000 |
| Target panel size | Expert-equivalent (~3.2 raters/pitch) |
| Mean majority-vote accuracy | 36.1% |
| 95% CI | 26.8% to 45.7% |
| Mean effective non-tied N | 83.4 pitches |

---

### Supplementary Table 14 (ST14): Prior-Exposure Descriptive Summary

Prior exposure was uncommon in the expert panel: 28 of 383 ratings with non-missing prior-exposure responses (7.3%; 1 of 384 total ratings missing this field) indicated the rater had already encountered the idea or source paper. Accuracy for prior-exposure ratings was 53.6% (15/28), compared with 34.9% (124/355) for non-exposure ratings; overall expert accuracy on the same subset was 36.3% (139/383). We report this as a descriptive check only.

---

### Supplementary Table 15 (ST15): Agreement and Consensus Diagnostics

**Table ST15a. Human inter-rater reliability.**

| Panel | Fleiss' kappa | 95% CI |
|-------|---------------|--------|
| Expert | 0.0469 | [-0.0114, 0.1068] |
| Junior | 0.0318 | [0.0194, 0.0446] |

**Table ST15b. Pairwise Cohen's kappa among 4 SFT models.**

| Pair | Family | Size | Agreement | $\kappa$ | Mean distance |
|------|--------|------|-----------|----------|---------------|
| GPT-4.1-FT $\times$ GPT-4.1-nano-FT | same | cross | 0.633 | +0.503 | 0.500 |
| GPT-4.1-FT $\times$ Qwen3-30B-FT | cross | same (large) | 0.675 | +0.557 | 0.425 |
| GPT-4.1-FT $\times$ Qwen3-4B-FT | cross | cross | 0.642 | +0.511 | 0.450 |
| GPT-4.1-nano-FT $\times$ Qwen3-30B-FT | cross | cross | 0.700 | +0.594 | 0.425 |
| GPT-4.1-nano-FT $\times$ Qwen3-4B-FT | cross | same (small) | 0.708 | +0.604 | 0.367 |
| Qwen3-30B-FT $\times$ Qwen3-4B-FT | same | cross | 0.650 | +0.524 | 0.458 |

Mean distance = mean absolute ordinal rank distance between model predictions (lower = more similar predictions).

Pairwise Cohen's $\kappa$ across the 6 model pairs ranges from +0.503 to +0.604. AI models are therefore an order of magnitude more internally consistent than human raters ($\kappa$ $\approx$ 0.03--0.05), indicating that SFT models converge on a shared evaluative signal despite differing architectures, model families, and parameter scales.

Agreement remains strongest for cross-family same-size pairs (mean $\kappa$ = +0.580), followed by cross-family cross-size pairs (+0.552), with same-family cross-size pairs lowest (+0.513). The highest agreement occurs for the cross-family small-model pair (GPT-4.1-nano-FT $\times$ Qwen3-4B-FT: $\kappa$ = +0.604), while the lowest occurs within the GPT family across sizes (GPT-4.1-FT $\times$ GPT-4.1-nano-FT: $\kappa$ = +0.503).

**Dissent frequency (which model disagrees with majority most often).**

| Model | N disagreements (of 120) |
|-------|--------------------------|
| GPT-4.1-FT | 24 |
| GPT-4.1-nano-FT | 24 |
| Qwen3-30B-FT | 22 |
| Qwen3-4B-FT | 22 |

**Table ST15c. Consensus coverage-accuracy tradeoff.**

| Policy | Coverage (N / 120) | Coverage (%) | Accuracy (%) |
|--------|---------------------|--------------|--------------|
| SFT 4/4 consensus | 50 | 41.7 | 72.0 |
| SFT >=3/4 consensus | 97 | 80.8 | 64.9 |
| SFT 2/4 split | 23 | 19.2 | 34.8 |
| Junior >=60% vote share | 3 | 2.5 | 66.7 |
| Junior >=50% vote share | 25 | 20.8 | 56.0 |
| Expert unanimous (>=2 raters) | 13 | 10.8 | 69.2 |
| Junior full-panel plurality | 120 | 100.0 | 35.0 |
| Expert full-panel plurality | 120 | 100.0 | 30.8 |

When all four SFT models agree ($N$ = 50), accuracy reaches 72.0%; when the strongest agreement is only 2/4, accuracy drops to 34.8%. Human voting shows a steeper tradeoff: junior >=60% consensus reaches comparable precision but covers only 2.5% of pitches. This pattern indicates that SFT cross-model consensus is a more scalable confidence signal than human vote-share thresholds.

**Per-class accuracy at full AI consensus (4/4).**

| Tier | Correct / N | Accuracy (%) |
|------|-------------|-------------|
| Exceptional | 14 / 14 | 100.0 |
| Strong | 6 / 12 | 50.0 |
| Fair | 6 / 9 | 66.7 |
| Limited | 10 / 15 | 66.7 |

**Per-evaluator prediction distribution.**

| Evaluator | Exceptional | Strong | Fair | Limited |
|-----------|-----------|--------|------|---------|
| GPT-4.1-FT | 0.358 | 0.192 | 0.275 | 0.175 |
| GPT-4.1-nano-FT | 0.300 | 0.200 | 0.317 | 0.183 |
| Qwen3-30B-FT | 0.342 | 0.225 | 0.258 | 0.175 |
| Qwen3-4B-FT | 0.325 | 0.250 | 0.292 | 0.133 |
| Human junior majority | 0.117 | 0.311 | 0.476 | 0.097 |
| Ground truth (uniform) | 0.250 | 0.250 | 0.250 | 0.250 |

All AI models over-predict "exceptional"; human junior majority over-predicts "fair." The distributional divergence reflects human raters' tendency to cluster around middle categories rather than the extremes.

**AI--human consistency.** AI--human pairwise $\kappa$ ranges from +0.10 to +0.21, substantially lower than AI--AI agreement ($\kappa$ = 0.50--0.72). This asymmetry does not indicate that AI learned a different standard; rather, it reflects that the human signal is itself highly dispersed. Humans individually score above random (experts 36.2%, juniors 31.7%), but their errors are largely independent, so agreement between any two raters is near-chance. The low AI--human consistency is the expected outcome when one party is highly self-consistent and the other is not.

---

### Supplementary Table 16 (ST16): Model Inventory and Access Window

**Table ST16a. Frontier reasoning models (conservative primary cohort).**

| Model | Provider | Access window | Samples/pitch |
|-------|----------|---------------|-----------------|
| Claude Opus 4.6 | Anthropic | March 2026 | 8 |
| GPT-5.2 High | OpenAI | March 2026 | 8 |
| Gemini 2.5 Pro | Google | March 2026 | 8 |
| Gemini 3.1 Pro | Google | March 2026 | 8 |
| Qwen 3.5 Plus | Alibaba | March 2026 | 8 |
| DeepSeek V3.2 | DeepSeek | March 2026 | 8 |
| Seed 2.0 | ByteDance | March 2026 | 8 |
| MiniMax M2.5 | MiniMax | March 2026 | 8 |
| Kimi K2.5 | Moonshot AI | March 2026 | 8 |
| Grok 4.1 Fast | xAI | March 2026 | 8 |
| GLM-5 | Zhipu AI | March 2026 | 8 |

**Table ST16b. Chat/log-probability evaluators.**

| Model | Provider | Access window | Log-probability extraction |
|-------|----------|---------------|-----------------------------|
| GPT-5.2 (chat) | OpenAI | March 2026 | Top-token log-probabilities |
| Kimi K2 (chat) | Moonshot AI | March 2026 | Top-token log-probabilities |
| DeepSeek Chat | DeepSeek | March 2026 | Top-token log-probabilities |

**Table ST16c. SFT/RL base architectures.**

| Model | Family | Architecture |
|-------|--------|--------------|
| GPT-4.1 | OpenAI | Proprietary transformer |
| GPT-4.1-nano | OpenAI | Proprietary transformer |
| Qwen3-4B-Instruct | Qwen | Dense transformer |
| Qwen3-30B-A3B-Instruct | Qwen | Mixture-of-experts |
| Qwen3-4B-Thinking (RL) | Qwen | Dense reasoning checkpoint |
| Qwen3-32B (RL) | Qwen | MoE reasoning checkpoint |

---

## Supplementary Figures

### Supplementary Figure 1 (SF1): Prediction Distribution Comparison

Prediction distributions across evaluator classes. **a**, 100% stacked predicted-tier shares for the frontier average (11 models), chat average (3 models), SFT single-model average (4 models), SFT 2-model ensemble, expert majority vote, and junior majority vote. This panel shows at a glance which evaluator classes collapse into the middle tiers and which use the full label space. **b**, Deviation of each predicted-tier share from the balanced 25% benchmark, showing evaluator-specific tier bias. **c**, Normalized prediction entropy of the full predicted distribution, where higher values indicate broader use of the four tiers and lower values indicate stronger collapse into a narrow subset of labels.

### Supplementary Figure 2 (SF2): Expert Individual Accuracy Distribution

**a**, Histogram of per-expert accuracy for the unfiltered expert panel (N = 48; 8 pitches each). Dashed line indicates mean accuracy (36.2%); dotted line indicates chance level (25%). The panel highlights substantial heterogeneity across experts rather than a tight cluster around a common level of skill. **b**, Empirical cumulative distribution function (CDF) of per-expert accuracy with the same chance baseline, making it easier to see how much of the expert panel lies near chance versus in the higher-performing tail.

### Supplementary Figure 3 (SF3): Junior Monte Carlo Subsampling Curve

**a**, Majority-vote accuracy versus panel size under repeated Monte Carlo subsampling (5,000 draws), with 95% confidence band. Chance baseline (25%) is shown as a dotted line, and the curve shows that larger panels help early but then flatten. **b**, Marginal accuracy gain per additional rater, showing diminishing returns as panel size increases and clarifying why aggregation alone does not keep improving linearly.

### Supplementary Figure 4 (SF4): Flagship Collapse-Metric Landscape

Collapse diagnostics for the 11 frontier flagships. **a**, Share of predictions assigned to the middle tiers (strong + fair) by model. **b**, Per-tier recall heatmap across the four quality tiers, making visible which classes are effectively never recovered. **c**, Relationship between middle-tier concentration and overall accuracy across models. **d**, Normalized prediction entropy ranking, where higher values indicate less distributional collapse and broader use of the four-tier scale.

### Supplementary Figure 5 (SF5): AI-Human Error Complementarity

**a**, Overlap decomposition of correct and incorrect outcomes between the SFT ensemble and expert majority vote on the expert-comparable subset (N = 89 non-tied pitches): both correct, AI-only correct, human-only correct, and shared error. This panel shows directly how much of the two systems' success is overlapping versus complementary. **b**, Complementarity ceiling: the oracle upper bound reached when either AI or the expert majority is correct is 75.3%, compared with SFT alone (55.1% on this subset) and expert majority vote (41.6%), quantifying the remaining room for hybrid routing strategies.

### Supplementary Figure 6 (SF6): Human Confusion Matrices

Row-normalized confusion matrices for expert and junior panels. **a**, Expert individual pooled (N = 384 ratings). **b**, Junior individual pooled (N = 2,530 ratings). **c**, Expert strict clear-majority voting (N = 89 non-tied; 31 ties excluded). **d**, Junior strict clear-majority voting (N = 103 non-tied; 17 ties excluded). Reading pooled and majority panels together clarifies how much disagreement is smoothed by voting and which off-diagonal confusions persist even after aggregation. Panel composition is summarized in Supplementary Table ST8, and majority-vote counts are summarized in Supplementary Table ST10.

---

## Supplementary References

Supplementary citations use the same numbered bibliography as the main manuscript.
