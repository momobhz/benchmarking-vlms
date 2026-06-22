# Literature Notes: Benchmarking spatial reasoning and affordance understanding of vision-language models for robotics using curated Robo2VLM subsets, prompt-mode ablations, and visual grounding sanity checks

This file is **optional**. Use it to keep a lightweight evidence trail for the papers you cite in `ref.bib`.
It helps prevent ``citation-only'' writing where claims drift away from what the cited work actually supports.

## Rules
- Keep each entry short (aim: 3--6 bullets).
- Every entry should answer: *what is the contribution* and *why are we citing it*.
- If a key claim is important, record the exact metric/dataset/setting.
- If the paper is not fully verifiable (paywall, missing PDF), mark **TODO** and do not rely on it for strong claims.
- If helpful, keep a very short abstract excerpt (1--3 sentences) to avoid re-searching.

## Entry template (copy/paste)

### <citationKey> --- <short title> (<year>)
- URL:
- Abstract (optional, 1--3 sentences max):
- One-sentence summary:
- Why we cite it (section + claim):
- Evidence (numbers / datasets / ablations):
- Limitations / caveats:
- Notes:

## Entries

### chen2025robo2vlm --- Robo2VLM (2025)
- URL: https://arxiv.org/abs/2505.15517
- Summary: Generates robotics VQA from large-scale in-the-wild robot manipulation trajectories.
- Why we cite it: Source benchmark and dataset generator.
- Evidence: Robo2VLM-1 contains 684,710 questions from 176k trajectories.

### bai2025qwen25vl --- Qwen2.5-VL (2025)
- URL: https://arxiv.org/abs/2502.13923
- Summary: Technical report for Qwen2.5-VL family.
- Why we cite it: One audited model family in our evaluation.

### wu2024deepseekvl2 --- DeepSeek-VL2 (2024)
- URL: https://arxiv.org/abs/2412.10302
- Summary: Mixture-of-experts VLM family with tiny/small/full variants.
- Why we cite it: One audited model family in our evaluation.

### driess2023palme --- PaLM-E (2023)
- URL: https://arxiv.org/abs/2303.03378
- Summary: Embodied multimodal language model trained for robotics and VQA-style tasks.
- Why we cite it: Robotics VLM context.

### brohan2023rt2 --- RT-2 (2023)
- URL: https://arxiv.org/abs/2307.15818
- Summary: Vision-language-action model transferring web VLM knowledge to robot control.
- Why we cite it: Robotics VLA context and CoT/semantic reasoning motivation.

### oneill2023openx --- Open X-Embodiment (2023)
- URL: https://arxiv.org/abs/2310.08864
- Summary: Multi-robot dataset and RT-X models for cross-embodiment robot learning.
- Why we cite it: Large-scale robot data context.

### khazatsky2024droid --- DROID (2024)
- URL: https://arxiv.org/abs/2403.12945
- Summary: Large in-the-wild robot manipulation dataset.
- Why we cite it: Real-world manipulation data context.

### kim2024openvla --- OpenVLA (2024)
- URL: https://arxiv.org/abs/2406.09246
- Summary: Open-source VLA trained on diverse robot demonstrations.
- Why we cite it: Modern open robotics VLA context.

### johnson2017clevr --- CLEVR (2017)
- URL: https://arxiv.org/abs/1612.06890
- Summary: Diagnostic visual reasoning dataset designed to expose reasoning failures and bias.
- Why we cite it: Motivation for diagnostic benchmarks.

### hudson2019gqa --- GQA (2019)
- URL: https://arxiv.org/abs/1902.09506
- Summary: Real-world compositional visual reasoning dataset with scene-graph-generated questions.
- Why we cite it: VQA benchmark context and aggregate-score caution.

### liu2022vsr --- Visual Spatial Reasoning (2022)
- URL: https://arxiv.org/abs/2205.00363
- Summary: Dataset testing natural-language spatial relations in images.
- Why we cite it: Spatial-reasoning benchmark context.

### chen2024spatialvlm --- SpatialVLM (2024)
- URL: https://arxiv.org/abs/2401.12168
- Summary: Spatial reasoning data generation and training for VLMs.
- Why we cite it: Evidence that VLM spatial reasoning requires targeted evaluation/training.

### cheng2024spatialrgpt --- SpatialRGPT (2024)
- URL: https://arxiv.org/abs/2406.01584
- Summary: Spatially grounded VLM with region and 3D-aware reasoning focus.
- Why we cite it: Recent spatial grounding benchmark/model context.

### fu2023mme --- MME (2023)
- URL: https://arxiv.org/abs/2306.13394
- Summary: Broad multimodal evaluation benchmark for perception and cognition.
- Why we cite it: General MLLM evaluation context.

### li2023pope --- POPE (2023)
- URL: https://arxiv.org/abs/2305.10355
- Summary: Polling-based object hallucination evaluation for LVLMs.
- Why we cite it: Grounding and hallucination-check precedent.
