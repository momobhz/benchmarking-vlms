# Presenter Reference: VLM Robotics Evaluation Benchmark

Generated: 2026-06-11  
Poster reference: `poster/final.pdf`

This is a compact backup document for detailed poster questions. It summarizes
the benchmark story, data curation, exact prompts, evaluation settings, result
tables, sanity checks, and known provenance caveats.

## 1. One-Minute Summary

- Goal: evaluate whether VLMs can answer robotics VQA questions that require
  spatial reasoning and affordance/state understanding.
- Dataset: `keplerccc/Robo2VLM-1`, test split.
- Curated test split size: 6,676 questions.
- Curated labels:
  - `spatial_reasoning`: 2,084 samples.
  - `affordance_understanding`: 842 samples.
  - `neither`: 3,750 samples.
- Main scored runs: Qwen2.5-VL-3B-Instruct and DeepSeek-VL2-tiny, each on
  200 spatial and 200 affordance questions, with zero-shot and CoT prompts.
- Main result: spatial performance stays near 5-way chance, while affordance is
  stronger but mostly driven by object-blockage questions.
- Prompting result: CoT changes behavior but is not reliably beneficial. The
  clearest failure is DeepSeek affordance, which drops from 72.0% zero-shot to
  21.5% CoT.
- Grounding result: high task accuracy does not imply grounded explanations.
  Sanity probes show rationalization and weak image-mismatch detection.

## 2. Source Artifact Map

Primary local artifacts:

- Poster PDF: `poster/final.pdf`.
- Main result summary: `docs/evaluation-results-and-sanity.md`.
- Evaluation workflow docs: `docs/evaluation.md`.
- Curation workflow docs: `docs/curation.md`.
- Evaluation prompt code: `src/vlm_bench/eval/prompts.py`.
- Curation taxonomy and subset mapping: `src/vlm_bench/curation/taxonomy.py`.
- VLM evaluator: `benchmark/evaluation.py`.
- Model-family prompt wrappers: `benchmark/vision_language.py`.
- Curation script: `scripts/recategorize_robo2vlm.py`.
- Curation postprocess scripts:
  - `scripts/relabel_robo2vlm_goal_state_questions.py`.
  - `scripts/subcategorize_robo2vlm_questions.py`.
- Main evaluation configs: `configs/eval/*_200.yaml`.
- Raw main result JSONs: `runs/eval/*/results/*.json`.
- Raw image-understanding sanity JSONs:
  - `runs/diagnostics/image_understanding/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/image_understanding_sanity.json`
  - `runs/diagnostics/image_understanding/deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_20260513_163446_deepseek-vl2-tiny_20260517_141222/image_understanding_sanity.json`

External-to-repo but readable local curation artifact used by poster figure:

- `/Users/momo/Uni/MSc/3DV/robo2vlm_spatial_affordance/summary.json`
- `/Users/momo/Uni/MSc/3DV/robo2vlm_spatial_affordance/curation_results.jsonl`

Important provenance caveats:

- Current config `configs/curation/robo2vlm_spatial_affordance.yaml` specifies
  curator model `gpt-5-mini`, but the curation artifact used for the poster
  reports `gpt-4.1-mini`, created on 2026-04-12, with prompt version
  `spatial_affordance_v1`.
- The configs reference `outputs/robo2vlm_spatial_affordance/curation_results.jsonl`,
  but that path is not present in this checkout. The original curation artifact
  exists at the absolute path above.
- Poster PDF includes extra-model and human-comparison visuals. The eight
  Qwen/DeepSeek result JSONs are locally auditable. I did not find local raw
  result JSONs for the poster's SmolVLM/Phi-4 extra-model table, nor a local
  source table for the human radar chart.
- Poster labels one extra baseline as `Phi-4-multimodal`, while checked-in
  Phi-4 eval configs use `microsoft/Phi-4-mini-reasoning`. Treat that row as
  poster-reported unless the exact raw artifact is available elsewhere.

## 3. Data Curation

### Dataset

- Dataset: `keplerccc/Robo2VLM-1`.
- Split: `test`.
- Processed count: 6,676 examples.
- Images are not duplicated into curation output; records keep dataset IDs and
  `source_index` for joining labels back to the original split.

### Top-Level Taxonomy

The LLM curator assigns exactly one label:

- `spatial_reasoning`: geometric or spatial relations, relative position,
  direction, depth, distance, cross-view correspondence, or movement direction
  in 2D/3D space.
- `affordance_understanding`: whether interaction is possible, blocked,
  stable, or physically permitted, including reachability, graspability,
  blockage, obstacle interference, and stable manipulation.
- `neither`: task success, temporal order, action phase, goal-state matching,
  task identification, language-instruction matching, or simple robot-state
  questions that are not primarily spatial or affordance questions.

### Counts

From `/Users/momo/Uni/MSc/3DV/robo2vlm_spatial_affordance/summary.json`:

| Label | Count | Mean curator confidence |
| --- | ---: | ---: |
| `spatial_reasoning` | 2,084 | 0.9428 |
| `affordance_understanding` | 842 | 0.9500 |
| `neither` | 3,750 | 0.9310 |
| Total | 6,676 | - |

Deterministic subcategory counts derived from question text:

| Parent label | Subcategory | Rule phrase | Count |
| --- | --- | --- | ---: |
| spatial | distance/depth | contains `point` | 1,283 |
| spatial | direction arrows | contains `which colored arrow` | 798 |
| spatial | other/none | no spatial rule match | 3 |
| affordance | object blockage | contains `obstacle blocking` | 543 |
| affordance | grasp stability | contains `stable` | 299 |

### Postprocessing

After LLM curation:

- Goal-state matching override moves questions containing
  `Which configuration shows the goal state that the robot should achieve?`
  from `spatial_reasoning` to `neither`.
- In the artifact used by the poster, this moved 107 questions.
- Subcategories are deterministic and are recomputed from final labels plus
  question text.

### Exact Curation System Prompt

Source: `scripts/recategorize_robo2vlm.py`.

```text
You are curating a robotics VQA benchmark.

Your job is to assign exactly one label to each question:

1. spatial_reasoning
   Use this when the question primarily evaluates geometric or spatial relations:
   relative position, direction, depth, distance, correspondence across views,
   or movement direction in 2D/3D space.

2. affordance_understanding
   Use this when the question primarily evaluates whether an interaction is
   possible, blocked, stable, or physically permitted: reachability, graspability,
   blockage, obstacle interference, or stable manipulation.

3. neither
   Use this for task success, temporal order, action phase, goal-state matching,
   task identification, language instruction matching, or simple robot state
   questions that are not primarily about spatial relations or affordances.

Decision rules:
- Choose the dominant evaluation target, not every concept mentioned.
- Original template tags are optional context only. They are not authoritative.

Examples:
- A question about "where is X relative to Y" is spatial_reasoning.
- A question about "can the robot reach or stably grasp X" is affordance_understanding.
- A question about "is the gripper open" is neither.
- A question about phases, next action, success, or overall task is neither.

Return only JSON that satisfies the provided schema.
```

### Exact Curation User Prompt Template

Source: `scripts/recategorize_robo2vlm.py`.

```text
Curate the following robotics VQA item.

Question ID: {record["id"]}
Original template tag: {record["original_tag"]}  # only included if present
Question: {record["question"]}
Choices:
1. {choice_1}
2. {choice_2}
...

Classify the question according to the taxonomy.
```

The response is constrained by a strict JSON schema with fields:

- `label`: one of `spatial_reasoning`, `affordance_understanding`, `neither`.
- `confidence`: number in [0, 1].
- `rationale`: non-empty string.
- `signals`: non-empty string array.

## 4. Evaluation Setup

### Core Design

- Evaluation is config-driven through `vlm-bench eval --config ...`.
- Local backend: vLLM only (`model.backend: vllm`).
- Dataset subset selection uses `curation_results.jsonl`.
- For curated subsets, `load_subset_source_indices()` iterates the curation
  JSONL in order and selects matching source indices.
- With `max_samples: 200`, the first 200 matching source indices are used. There
  is no random sampling in the evaluator.
- Spatial and affordance tasks are 5-way multiple choice, so chance is roughly
  20%.

### Main Evaluated Models

| Poster name | Hugging Face ID in configs/results |
| --- | --- |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` |
| DeepSeek-VL2 tiny | `deepseek-ai/deepseek-vl2-tiny` |

### Main Run Settings

Shared across the eight main runs:

- Dataset: `keplerccc/Robo2VLM-1`.
- Split: `test`.
- Subsets: `spatial`, `affordance`.
- Examples per run: 200.
- Image mode: `rgb`.
- Temperature: 0.0.
- Batch size: 1.
- Tensor parallel size: 1.
- `trust_remote_code: true`.
- Sanity check during scored eval: `none`.
- Sanity seed in configs: 11.
- CoT max tokens: 10,240.
- Zero-shot max tokens: 128.

Main 200-sample source-index ranges:

| Subset | First source indices | Min | Max | Count |
| --- | --- | ---: | ---: | ---: |
| spatial | 368, 369, 370, 372, 373, ... | 368 | 944 | 200 |
| affordance | 0, 2, 4, 5, 12, ... | 0 | 1178 | 200 |

## 5. Exact Evaluation Prompts

### Dataset Question Formatting

Source: `benchmark/evaluation.py`.

The evaluator formats each dataset item as:

```text
{question_text}
Choices:
A. {choice_0}
B. {choice_1}
C. {choice_2}
D. {choice_3}
E. {choice_4}
```

### Prompt-Mode Instructions

Source: `src/vlm_bench/eval/prompts.py`.

CoT instruction:

```text
Answer the following multiple choice question by selecting the letter (A, B, C, D, or E). Reason step by step about the answer, and show your work for each step. Only after that, proceed to the final answer. Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end.
```

Zero-shot instruction:

```text
Answer the following multiple choice question by selecting the letter (A, B, C, D, or E). Only output the correct option letter, i.e., A, B, C, D, or E.
```

The benchmark-level prompt is:

```text
{prompt_mode_instruction} {formatted_multiple_choice_question}
```

### Model-Specific Chat Wrappers

Source: `benchmark/vision_language.py`.

Qwen2.5-VL wrapper:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{benchmark_prompt}<|im_end|>
<|im_start|>assistant
```

DeepSeek-VL2 wrapper:

```text
<|User|>: <image>
{benchmark_prompt}

<|Assistant|>:
```

Gemma 3 wrapper:

```text
<bos><start_of_turn>user
<start_of_image>{benchmark_prompt}<end_of_turn>
<start_of_turn>model
```

SmolVLM wrapper:

```text
<|im_start|>User:<image>{benchmark_prompt}<end_of_utterance>
Assistant:
```

Phi-4 multimodal wrapper in code:

```text
<|user|><|image_1|>{benchmark_prompt}<|end|><|assistant|>
```

### RGB-Depth Instruction

The main scored runs use RGB only. The evaluator also supports `rgb_depth`.
When enabled, it creates a side-by-side RGB plus estimated-depth image and
prepends:

```text
The provided image has two side-by-side panels: the left panel is the original RGB image, and the right panel is an estimated depth map generated from that RGB image. Use both panels when helpful.
```

Depth model default:

```text
depth-anything/Depth-Anything-V2-Small-hf
```

### Answer Extraction and Scoring

Source: `benchmark/evaluation.py`.

- The model produces free text.
- The evaluator extracts standalone option letters with regex
  `\b(A|B|C|D|E)\b` after uppercasing.
- If multiple letters are found, it uses the last one.
- Integer expected answers are converted with `chr(65 + expected_answer)`.
- A row is correct when `predicted_letter == expected_letter`.
- Overall accuracy is `correct / total * 100`.

This means CoT can include many letters in the reasoning; the last standalone
letter controls scoring.

## 6. Main Results

Auditable result JSONs under `runs/eval/*/results/*.json`.

| Model | Subset | Prompt | Timestamp | Correct | Accuracy | Avg response time |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Qwen2.5-VL-3B | spatial | zero-shot | 20260519_030705 | 40/200 | 20.0% | 0.18s |
| Qwen2.5-VL-3B | spatial | CoT | 20260519_030446 | 39/200 | 19.5% | 4.90s |
| Qwen2.5-VL-3B | affordance | zero-shot | 20260516_165054 | 131/200 | 65.5% | 0.46s |
| Qwen2.5-VL-3B | affordance | CoT | 20260519_031940 | 128/200 | 64.0% | 3.40s |
| DeepSeek-VL2 tiny | spatial | zero-shot | 20260513_163639 | 39/200 | 19.5% | 0.17s |
| DeepSeek-VL2 tiny | spatial | CoT | 20260513_163446 | 44/200 | 22.0% | 0.74s |
| DeepSeek-VL2 tiny | affordance | zero-shot | 20260519_222757 | 144/200 | 72.0% | 0.14s |
| DeepSeek-VL2 tiny | affordance | CoT | 20260519_222609 | 43/200 | 21.5% | 0.27s |

Prompt-mode comparison:

| Model | Subset | Zero-shot | CoT | Delta | Same predicted letter | Zero right, CoT wrong | Zero wrong, CoT right |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | spatial | 20.0% | 19.5% | -0.5 pp | 53/200 | 29 | 28 |
| Qwen2.5-VL-3B | affordance | 65.5% | 64.0% | -1.5 pp | 149/200 | 24 | 21 |
| DeepSeek-VL2 tiny | spatial | 19.5% | 22.0% | +2.5 pp | 63/200 | 23 | 28 |
| DeepSeek-VL2 tiny | affordance | 72.0% | 21.5% | -50.5 pp | 69/200 | 114 | 13 |

Subcategory breakdown:

| Model | Prompt | Direction arrows | Distance/depth | Cross-view correspondence |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-VL-3B | zero-shot | 8/76, 10.5% | 24/91, 26.4% | 8/33, 24.2% |
| Qwen2.5-VL-3B | CoT | 12/76, 15.8% | 21/91, 23.1% | 6/33, 18.2% |
| DeepSeek-VL2 tiny | zero-shot | 14/76, 18.4% | 17/91, 18.7% | 8/33, 24.2% |
| DeepSeek-VL2 tiny | CoT | 15/76, 19.7% | 21/91, 23.1% | 8/33, 24.2% |

| Model | Prompt | Object blockage | Grasp stability |
| --- | --- | ---: | ---: |
| Qwen2.5-VL-3B | zero-shot | 105/132, 79.5% | 26/68, 38.2% |
| Qwen2.5-VL-3B | CoT | 107/132, 81.1% | 21/68, 30.9% |
| DeepSeek-VL2 tiny | zero-shot | 112/132, 84.8% | 32/68, 47.1% |
| DeepSeek-VL2 tiny | CoT | 18/132, 13.6% | 25/68, 36.8% |

Interpretation:

- Spatial is near chance across both models and prompt modes.
- Affordance accuracy is mostly carried by object-blockage questions.
- Grasp stability remains weak; best observed run is DeepSeek zero-shot at
  47.1%.
- DeepSeek CoT collapses specifically on object blockage.
- Qwen CoT is slower and not more accurate overall.

## 7. Poster-Reported Extra Results

These numbers are visible in `poster/final.pdf`, but I did not find local raw
result JSONs in this checkout.

| Model | Spatial ZS | Affordance ZS | Spatial CoT | Affordance CoT |
| --- | ---: | ---: | ---: | ---: |
| SmolVLM-2.2B | 20% | 23% | 66% | 47% |
| Phi-4-multimodal | 12% | 19% | 40% | 44% |

The checked-in configs include SmolVLM and Phi-4 variants, but the local raw
JSON result directory only contains Qwen/DeepSeek scored runs.

## 8. Validation and Grounding Checks

### Poster-Reported Scored Validation Tests

Visible in `poster/final.pdf`.

| Condition | Spatial | Affordance |
| --- | ---: | ---: |
| Normal RGB | 20% (100 samples) | 62.6% (842 samples) |
| Without images / blank | 21-29% | 30.2% (842 samples) |
| Random question-image pairs / shuffle | 19-26% | 36% (100 samples) |
| RGB + depth | 22% | - |

Takeaway: blank/shuffled images do not fully destroy spatial performance
because spatial accuracy is already near chance. Affordance drops more clearly
under missing or mismatched images.

### Raw Image-Understanding Sanity Probes

Source: `scripts/run_image_understanding_sanity.py` and
`runs/diagnostics/image_understanding/*/image_understanding_sanity.json`.

Setup:

- Separate from scored eval.
- Starts from spatial CoT result files.
- Selects 20 examples.
- Expands into five probe types, for 100 raw-generation probes per model.
- Temperature: 0.0.
- Max tokens: 512.
- Batch size: 1.
- Wrong-answer strategy: `next`.
- Sanity seed: 11.

Probe types:

- `explain_correct`: original image plus correct answer, ask model to justify.
- `explain_wrong`: original image plus deliberately wrong answer, ask model to
  assess support.
- `explain_correct_blank`: blank image plus original correct answer.
- `explain_correct_shuffle`: mismatched shuffled image plus original correct
  answer.
- `overlay_describe`: ask model to describe overlays without answering.

Heuristic summary:

| Probe | Expected behavior | Qwen2.5-VL-3B | DeepSeek-VL2 tiny |
| --- | --- | ---: | ---: |
| `explain_correct` | support correct answer | 12/20 supports, 8 unclear | 0/20 supports, 20 unclear |
| `explain_wrong` | reject wrong answer | 0/20 rejects, 16 supports, 4 unclear | 3/20 rejects, 17 unclear |
| `explain_correct_blank` | reject blank image | 20/20 rejects | 20/20 rejects |
| `explain_correct_shuffle` | reject mismatched image | 3/20 rejects, 17 unclear | 20/20 rejects |
| `overlay_describe` | describe overlays | all 20 unclear | all 20 unclear |

The heuristic is phrase-based triage. It is useful for failure discovery but
should not be treated as a final manual grounding score.

### Exact Sanity Prompt Templates

For explanation probes:

```text
You are inspecting a robotics VQA image.

{question}

{answer_intro} {given_letter}: {given_text}.

{expectation}
Do not choose a new answer unless the image contradicts the proposed answer.
```

For `explain_correct`:

```text
answer_intro = "The known correct answer is"
expectation = "Explain why this answer is correct using only visible evidence from the image. If the image does not visibly support the answer, say that explicitly."
```

For `explain_wrong`:

```text
answer_intro = "The proposed answer is"
expectation = "Assess whether this proposed answer is supported by visible evidence. If it is correct, explain why. If it is not supported, say so explicitly and identify the visual conflict."
```

For blank and shuffled controls:

```text
answer_intro = "The known correct answer for the original question is"
expectation = "Explain whether the current image visibly supports that answer. If the current image is blank, mismatched, or insufficient, say so explicitly."
```

For direction-arrow overlay description:

```text
You are inspecting a robotics VQA image with visual overlays.

Original question: {first_question_line}

List each visible colored arrow and describe the direction it points in image coordinates. Also mention the robot gripper/current position if visible. Do not answer the multiple-choice question.
```

For depth point questions:

```text
You are inspecting a robotics VQA image with visual overlays.

Original question: {first_question_line}

List each visible colored point and describe its approximate 2D image position (for example top-left, center, lower-right). Mention any visible depth cues, but do not decide which point is closest or farthest.
```

For cross-view correspondence:

```text
You are inspecting a robotics VQA image with visual overlays.

Original question: {first_question_line}

Describe the left and right image panels. Locate the red dot in the left panel and list the labeled candidate points in the right panel with their approximate positions. Do not choose the corresponding point.
```

## 9. Representative Failure Cases

### DeepSeek Affordance: Zero-Shot Correct, CoT Wrong

- `question_id`: `fractal20220817_data_place_blue_plastic_bottle_into_middle_drawer_3973_q43`
- `source_index`: 0.
- Question: `Is there any obstacle blocking the robot from reaching bottle?`
- Expected: `A`.
- Zero-shot predicted: `A`, correct.
- CoT predicted: `D`, incorrect.
- CoT excerpt: `Reason: The presence of an object (bottle) near the end of the robotic arm suggests that it may pose a challenge for the robot's movement. Answer: D`
- Interpretation: CoT appears to treat the target bottle itself as an obstacle,
  matching the aggregate object-blockage collapse.

### Qwen Affordance: CoT Fixes One Zero-Shot Error

- `question_id`: `fractal20220817_data_pick_brown_chip_bag_64_q9`
- `source_index`: 20.
- Question: `Is there any obstacle blocking the robot from reaching chip bag?`
- Expected: `C`.
- Zero-shot predicted: `D`, incorrect.
- CoT predicted: `C`, correct.
- CoT excerpt: `The water bottle is to the right of the chip bag. There are no other objects or obstacles between the robot and the chip bag.`
- Interpretation: CoT can help individual cases, but aggregate Qwen affordance
  still slightly decreases with CoT.

### Qwen Spatial: CoT Uses Weak 2D Heuristics

- `question_id`: `droid_pick_up_the_objects_from_the_table_and_put_them_on_the_plastic_5775_q17`
- `source_index`: 374.
- Question: `In the image from ext1, which colored point is FARTHEST from the camera?`
- Expected: `B`.
- Zero-shot predicted: `B`, correct.
- CoT predicted: `D`, incorrect.
- CoT excerpt: `The distance from the camera to each point can be inferred from their positions.`
- Interpretation: the explanation relies on 2D image position as a proxy for
  depth, which is unreliable.

### DeepSeek Spatial: CoT Helps One Depth Example

- `question_id`: `droid_pick_up_the_objects_from_the_bowl_and_put_them_in_the_box_14259_q49`
- `source_index`: 370.
- Question: `In the image from ext1, which colored point is CLOSEST to the camera?`
- Expected: `C`.
- Zero-shot predicted: `A`, incorrect.
- CoT predicted: `C`, correct.
- CoT excerpt: `The green point appears to be the nearest to the camera compared to the other colored points.`
- Interpretation: DeepSeek gains some spatial examples from CoT, but the net
  gain is only +2.5 percentage points on the 200-example spatial subset.

### Qwen Sanity: Rationalizes A Wrong Answer

- Probe: `explain_wrong`.
- `question_id`: `droid_pick_up_the_objects_and_put_them_in_the_box_1174_q15`.
- Correct answer: `A: Red`.
- Deliberately supplied wrong answer: `B: Purple`.
- Response excerpt: `The proposed answer B: Purple is supported by the visible evidence in the image.`
- Heuristic judgment: `supports`.
- Interpretation: the model can justify an intentionally wrong answer.

### DeepSeek Sanity: Rejects Shuffled Image

- Probe: `explain_correct_shuffle`.
- `question_id`: `droid_pick_up_the_object_5044_q7`.
- Original correct answer supplied: `A: Red`.
- Image control: `shuffle`.
- Response excerpt: `The current image does not visibly support the proposed answer.`
- Heuristic judgment: `rejects_or_insufficient`.
- Interpretation: DeepSeek is more sensitive to shuffled-image controls than
  Qwen in this diagnostic, even though its scored spatial accuracy is near
  chance.

## 10. Likely Questions and Short Answers

Q: Why is spatial chance about 20%?  
A: The evaluated tasks have five multiple-choice options A-E. Uniform guessing
is therefore 1/5 = 20%.

Q: Are the 200 examples random?  
A: No. The evaluator loads source indices from `curation_results.jsonl` in file
order and uses the first 200 matching examples when `max_samples: 200`.

Q: Did the main results use depth images?  
A: No. Main scored runs use `image.mode: rgb`. The evaluator supports
`rgb_depth`, and the poster reports a spatial RGB-depth validation score of 22%,
but the main headline table is RGB-only.

Q: Why is affordance higher than spatial?  
A: The aggregate affordance score is mostly driven by object-blockage questions,
where both zero-shot models perform relatively well. Grasp stability remains
much weaker.

Q: Does CoT help?  
A: Not reliably. It slightly helps DeepSeek spatial, slightly hurts Qwen, and
dramatically hurts DeepSeek affordance.

Q: Why does DeepSeek affordance CoT collapse?  
A: The strongest observed pattern is target/obstacle confusion. In a bottle
blockage example, CoT treats the target bottle itself as a movement obstacle.

Q: Are explanations grounded?  
A: Not necessarily. Sanity probes show that models can reject blank images while
still rationalizing wrong answers or failing to reject shuffled images.

Q: How exactly are letters extracted?  
A: The evaluator extracts standalone A-E letters from the generated text and
uses the last one. This is robust for direct answers but means CoT outputs must
end with the intended final letter.

Q: What curation model was used?  
A: The original curation artifact used for the poster reports `gpt-4.1-mini`
with prompt version `spatial_affordance_v1`. The current checked-in curation
config defaults to `gpt-5-mini`, so regenerated curation could differ.

Q: Can we claim human-level comparison numbers?  
A: The poster contains human-comparison radar charts, but I did not find the
underlying source table in this checkout. Use the visual comparison cautiously
unless the source spreadsheet/artifact is available.

## 11. Repro Commands

Dry-run one evaluation:

```bash
PYTHONPATH=src python3 -m vlm_bench eval \
  --config configs/eval/qwen25_3b_spatial_cot_200.yaml \
  --dry-run
```

Submit the eight main configs:

```bash
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_spatial_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_affordance_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/qwen25_3b_affordance_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_spatial_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_spatial_zeroshot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_affordance_cot_200.yaml
sbatch slurm/eval.sbatch configs/eval/deepseek_vl2_tiny_affordance_zeroshot_200.yaml
```

Dry-run curation:

```bash
PYTHONPATH=src python3 -m vlm_bench curate \
  --config configs/curation/robo2vlm_spatial_affordance.yaml \
  --dry-run
```

Run image-understanding sanity diagnostics:

```bash
python3 scripts/run_image_understanding_sanity.py \
  --result-json runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901.json \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --max-examples 20 \
  --temperature 0.0 \
  --max-tokens 512 \
  --batch-size 1 \
  --tensor-parallel-size 1
```

## 12. Final Takeaways

The benchmark is strongest as evidence of current limitations rather than as a
claim that one model robustly solves robotics VQA. The most defensible claims
are:

- Spatial reasoning remains unsolved in these tested settings.
- Object-blockage questions are easier than grasp stability.
- CoT is an experimental ablation and can actively harm performance.
- Evaluation should include visual sanity checks because answer accuracy alone
  can hide ungrounded explanations.

