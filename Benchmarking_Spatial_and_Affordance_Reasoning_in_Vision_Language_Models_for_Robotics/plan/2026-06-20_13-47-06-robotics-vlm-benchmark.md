---
mode: paper-plan
topic: "Benchmarking spatial reasoning and affordance understanding of vision-language models for robotics using curated Robo2VLM subsets, prompt-mode ablations, and visual grounding sanity checks"
timestamp: 2026-06-20_13-47-06
slug: robotics-vlm-benchmark
created_at: "2026-06-20T13:47:06+02:00"
complexity: medium
latex_available: false
---

# Paper Plan: Benchmarking spatial reasoning and affordance understanding of vision-language models for robotics using curated Robo2VLM subsets, prompt-mode ablations, and visual grounding sanity checks

## Goal
- Produce a compact final project report in the existing 3DV/CVPR-style LaTeX template under this folder.
- Main text must not exceed 6 pages; references may occupy a 7th page.
- Ground claims in local artifacts and verified literature. Avoid unsupported novelty claims.
- Include problem statement, short related work, formulation/algorithm, implementation provenance, results, discussion, and references.

## Scope
- In: Robo2VLM-derived spatial/affordance benchmark curation; Qwen2.5-VL-3B and DeepSeek-VL2-tiny 200-example results; zero-shot vs. CoT prompt comparison; subcategory analysis; visual-grounding sanity probes; implementation provenance.
- Out unless user approves: broad claims that this is the first spatial/affordance robotics VLM benchmark overall. Poster-reported SmolVLM/Phi-4 numbers may be included as user-approved results.

## Kickoff Gate (must be confirmed before writing)
- **STOP**: Do not write prose into `main.tex` until this gate is confirmed and the issues CSV exists.
- [x] User confirmed scope + outline in chat
- Venue/template: existing 3DV/CVPR-style LaTeX template in `Benchmarking_Spatial_and_State_Reasoning_Skills_of_VLMs_for_Robotics/`
- Target length: <=6 pages main text, references-only 7th page allowed
- "Latest" definition: local artifacts as of 2026-06-20; do not claim unobserved latest benchmark results
- Scope boundaries: use local Qwen/DeepSeek artifacts plus user-approved poster-reported SmolVLM/Phi-4 numbers.

## Clarification Q&A (record answers)
| Question | Answer |
|---|---|
| What venue + page limit should we target? | Existing 3DV/CVPR template; <=6 pages excluding references; references-only page 7 allowed. |
| Which subtopics MUST be covered? | Problem statement, related work, formulation/algorithm, results, discussion, explicit contributions, implementation provenance. |
| What are the main emphasis areas? | Robotics VLM spatial reasoning and affordance understanding on Robo2VLM-derived questions. |
| Any required datasets/baselines? | Robo2VLM-1 test split; Qwen2.5-VL-3B-Instruct and DeepSeek-VL2-tiny raw JSON results. |
| Expected figures/tables (min 5 types): any must-haves? | Proposed below; final count may be reduced to fit 6 pages. |
| Author block / anonymity? | Maurice Behanzin, Celine Sonnenschein, Nina Gruteser, Mathilda Lee. Keep review mode. |

## Confirmed Outline (edit to match the user-approved outline)
1. Introduction
   - State why spatial and affordance reasoning matter for robotics VLMs.
   - Describe the gap: general VLM/VQA scores do not isolate embodied spatial/action reasoning.
   - Contributions, phrased conservatively as an evaluation layer on Robo2VLM.
2. Related Work
   - Robotics VLM/VLA context: PaLM-E, RT-2, Open X-Embodiment/OpenVLA, DROID.
   - VQA and spatial reasoning benchmarks: CLEVR, GQA, VSR, SpatialVLM/SpatialRGPT.
   - Multimodal evaluation and hallucination/grounding checks: MME, POPE.
3. Benchmark Formulation and Implementation
   - Define curated labels, deterministic subcategories, and goal-state override.
   - Define prompt modes, answer extraction, accuracy, paired prompt deltas, and sanity probes.
   - Separate implemented code from external code/assets.
4. Experimental Setup
   - Dataset: `keplerccc/Robo2VLM-1`, test split.
   - Curation artifact: 6,676 examples; 2,084 spatial, 842 affordance, 3,750 neither.
   - Main runs: Qwen2.5-VL-3B-Instruct and DeepSeek-VL2-tiny; 200 examples per subset/prompt; RGB, temperature 0.
5. Results
   - Headline accuracies and response times.
   - Prompt-mode comparison and subcategory breakdown.
   - Sanity-probe matrix.
6. Discussion and Limitations
   - Spatial results are near 5-way chance; affordance is driven by object blockage; grasp stability remains weak.
   - CoT is not reliably beneficial and strongly hurts DeepSeek affordance.
   - Accuracy alone does not imply grounded explanations.
   - Limitations: first 200 curated examples for the main raw-result matrix, heuristic sanity labels, and benchmark sensitivity to aggregate/subcategory reporting.
7. Conclusion
   - One compact paragraph if space allows; otherwise fold into Discussion.

## Plan Notes
- Keywords used for discovery: Robo2VLM, robotics VQA, vision-language-action robotics, spatial reasoning VLM, visual grounding hallucination benchmark, Qwen2.5-VL, DeepSeek-VL2.
- Candidate titles proposed:
  - Benchmarking Spatial and Affordance Reasoning in Vision-Language Models for Robotics
  - Do Vision-Language Models Ground Robotic Spatial and Affordance Questions?
  - A Robo2VLM-Based Benchmark for Spatial and Affordance Understanding in Robotics VQA
- Proposed contribution wording:
  - "This project does not introduce a new VLM or a new Robo2VLM generation method. Its contribution is an evaluation layer over Robo2VLM: a curated spatial/affordance split with deterministic subcategories, a config-driven vLLM evaluation harness with prompt and image controls, and an auditable empirical study of two open VLMs with visual-grounding sanity probes."
  - Avoid "first" unless the user explicitly approves and accepts the risk/wording.
- Evidence-backed result scope:
  - Use latest Qwen spatial reruns from 2026-05-19, Qwen affordance results from 2026-05-16/19, and DeepSeek results from 2026-05-13/19.
  - Note that Qwen spatial 2026-05-19 reruns exactly reproduce 2026-05-13 correctness.
  - Include poster-reported SmolVLM/Phi-4 numbers as user-approved results.
- Visualization plan:
  - Table: curation label/subcategory counts.
  - Table: main accuracy and response-time matrix.
  - Table or compact plot: subcategory breakdown.
  - Table: sanity-probe heuristic outcomes.
  - Small qualitative failure-case box if space permits.
  - Existing images that may be reused if space permits: `images/robo2vlm_pipeline.png`, `output/pdf/data_categorization_chart.pdf`, `output/pdf/discussion_sanity_heatmap_matplotlib.pdf`.
- Verified citation spine:
  - Robo2VLM (Chen et al., 2025), Qwen2.5-VL (Bai et al., 2025), DeepSeek-VL2 (Wu et al., 2024), PaLM-E (Driess et al., 2023), RT-2 (Brohan et al., 2023), Open X-Embodiment (O'Neill et al., 2023), DROID (Khazatsky et al., 2024), OpenVLA (Kim et al., 2024), CLEVR (Johnson et al., 2017), GQA (Hudson and Manning, 2019), VSR (Liu et al., 2022), SpatialVLM (Chen et al., 2024), SpatialRGPT (Cheng et al., 2024), MME (Fu et al., 2023), POPE (Li et al., 2023).

## Issue CSV
- Path: <paper_dir>/issues/2026-06-20_13-47-06-robotics-vlm-benchmark.csv
- Must share the same timestamp/slug as this plan
- This CSV is the execution contract: update issue status as you write and QA
- Issues may be added/split/inserted during execution; re-validate after edits and keep going until all issues are `DONE`/`SKIP` (when feasible, in the same run).
