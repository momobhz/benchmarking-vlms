# Evaluation Results And Sanity Checks

Generated from local artifacts on 2026-05-20.

## Scope

This report summarizes the 200-example Robo2VLM evaluation outputs under
`runs/eval/*/results/*.json` and the image-understanding diagnostic outputs
under `runs/diagnostics/image_understanding/*/image_understanding_sanity.json`.

The evaluation runs use:

- Dataset: `keplerccc/Robo2VLM-1`
- Split: `test`
- Curated subsets: `spatial` and `affordance`
- Image mode: `rgb`
- Temperature: `0.0`
- Scored eval sanity mode: `none`

Important metadata caveat: older Qwen and spatial DeepSeek result files store
`tag: unknown`, so their subcategory breakdowns below are derived from question
text. The latest DeepSeek affordance runs already include curated subcategory
tags in the result rows.

## Headline Results

| Model | Subset | Prompt | Accuracy | Avg response time |
|---|---|---:|---:|---:|
| DeepSeek-VL2 tiny | affordance | CoT | 21.5% | 0.27s |
| DeepSeek-VL2 tiny | affordance | zero-shot | 72.0% | 0.14s |
| DeepSeek-VL2 tiny | spatial | CoT | 22.0% | 0.74s |
| DeepSeek-VL2 tiny | spatial | zero-shot | 19.5% | 0.17s |
| Qwen2.5-VL-3B | affordance | CoT | 64.0% | 3.40s |
| Qwen2.5-VL-3B | affordance | zero-shot | 65.5% | 0.46s |
| Qwen2.5-VL-3B | spatial | CoT | 19.5% | 4.90s |
| Qwen2.5-VL-3B | spatial | zero-shot | 20.0% | 0.18s |

The Qwen spatial reruns from 2026-05-19 exactly reproduce the 2026-05-13
spatial runs: same 200 source examples, same predicted letters, and same
correctness pattern. That is useful for reproducibility, but it is not a
performance improvement.

## Prompt-Mode Comparison

| Model | Subset | Zero-shot accuracy | CoT accuracy | Delta | Same predicted letter | Zero right, CoT wrong | Zero wrong, CoT right |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | spatial | 20.0% | 19.5% | -0.5 pp | 53/200 | 29 | 28 |
| Qwen2.5-VL-3B | affordance | 65.5% | 64.0% | -1.5 pp | 149/200 | 24 | 21 |
| DeepSeek-VL2 tiny | spatial | 19.5% | 22.0% | +2.5 pp | 63/200 | 23 | 28 |
| DeepSeek-VL2 tiny | affordance | 72.0% | 21.5% | -50.5 pp | 69/200 | 114 | 13 |

Insights:

- Spatial remains near chance for both models. The spatial subset is 5-way
  multiple choice, so chance is roughly 20%.
- CoT is not reliably helpful. It slightly improves DeepSeek spatial, slightly
  hurts Qwen, and severely hurts DeepSeek affordance.
- DeepSeek affordance zero-shot is the strongest single run at 72.0%.
- Qwen affordance is stable across prompt modes, but CoT is much slower and not
  more accurate.

## Subcategory Breakdown

Spatial subcategories are inferred from question text for these artifacts:

| Model | Prompt | Direction arrows | Distance / depth | Cross-view correspondence |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B | zero-shot | 10.5% | 26.4% | 24.2% |
| Qwen2.5-VL-3B | CoT | 15.8% | 23.1% | 18.2% |
| DeepSeek-VL2 tiny | zero-shot | 18.4% | 18.7% | 24.2% |
| DeepSeek-VL2 tiny | CoT | 19.7% | 23.1% | 24.2% |

Affordance subcategories:

| Model | Prompt | Object blockage | Grasp stability |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | zero-shot | 79.5% |, 38.2% |
| Qwen2.5-VL-3B | CoT | 81.1% | 30.9% |
| DeepSeek-VL2 tiny | zero-shot | 84.8% | 47.1% |
| DeepSeek-VL2 tiny | CoT | 13.6% | 36.8% |

Insights:

- The affordance score is mostly carried by object-blockage questions. Both
  zero-shot models perform much better on object blockage than on grasp stability.
- Grasp stability remains weak: the best observed score is DeepSeek zero-shot at
  47.1%.
- DeepSeek CoT collapses specifically on object blockage, falling from 84.8% to
  13.6%. This is the clearest prompt-mode regression in the current artifacts.
- Spatial direction-arrow questions are especially weak for Qwen, with 10.5%
  zero-shot and 15.8% CoT.

## Sanity Evaluations

The image-understanding diagnostics are separate from scored evaluation. They
start from spatial CoT result files and ask five probe types over 20 selected
examples, producing 100 raw-generation probes per model.

| Probe | Expected behavior | Qwen2.5-VL-3B heuristic result | DeepSeek-VL2 tiny heuristic result |
|---|---|---:|---:|
| `explain_correct` | justify correct answer on original image | 12/20 supports, 8/20 unclear | 0/20 supports, 20/20 unclear |
| `explain_wrong` | reject deliberately wrong answer | 0/20 rejects, 16/20 supports, 4/20 unclear | 3/20 rejects, 17/20 unclear |
| `explain_correct_blank` | reject blank image as insufficient | 20/20 rejects | 20/20 rejects |
| `explain_correct_shuffle` | reject mismatched shuffled image | 3/20 rejects, 17/20 unclear | 20/20 rejects |
| `overlay_describe` | describe overlays without answering | all 20 unclear | all 20 unclear |

The heuristic labels are phrase-based triage, not a substitute for manual
review. They are still useful for identifying failure modes:

- Qwen reliably detects blank images, but often rationalizes wrong answers and
  usually fails to reject shuffled-image controls.
- DeepSeek reliably rejects blank and shuffled controls, but its correct-answer
  explanations are rarely phrased in a way the heuristic marks as support.
- `overlay_describe` needs manual review or a better rubric; the current
  heuristic does not score descriptive quality.

## Precise Examples

### DeepSeek Affordance: Zero-Shot Correct, CoT Wrong

Source:
`runs/eval/deepseek_vl2_tiny_affordance_zeroshot_200_t0/results/deepseek_vl2_tiny_affordance_zeroshot_200_t0_deepseek-vl2-tiny_affordance_zero_shot_20260519_222757.json`
and
`runs/eval/deepseek_vl2_tiny_affordance_cot_200_t0/results/deepseek_vl2_tiny_affordance_cot_200_t0_deepseek-vl2-tiny_affordance_cot_20260519_222609.json`

- `question_id`: `fractal20220817_data_place_blue_plastic_bottle_into_middle_drawer_3973_q43`
- `source_index`: 0
- Question: `Is there any obstacle blocking the robot from reaching bottle?`
- Expected: `A`
- Zero-shot predicted: `A`, correct
- CoT predicted: `D`, incorrect
- CoT response excerpt: `Reason: The presence of an object (bottle) near the end of the robotic arm suggests that it may pose a challenge for the robot's movement. Answer: D`

Interpretation: the CoT answer appears to treat the target object itself as an
obstacle. This matches the aggregate object-blockage collapse in DeepSeek CoT.

### Qwen Affordance: CoT Fixes One Zero-Shot Error

Source:
`runs/eval/qwen25_3b_affordance_zeroshot_200_t0/results/qwen25_3b_affordance_zeroshot_200_t0_Qwen2.5-VL-3B-Instruct_affordance_zero_shot_20260516_165054.json`
and
`runs/eval/qwen25_3b_affordance_cot_200_t0/results/qwen25_3b_affordance_cot_200_t0_Qwen2.5-VL-3B-Instruct_affordance_cot_20260519_031940.json`

- `question_id`: `fractal20220817_data_pick_brown_chip_bag_64_q9`
- `source_index`: 20
- Question: `Is there any obstacle blocking the robot from reaching chip bag?`
- Expected: `C`
- Zero-shot predicted: `D`, incorrect
- CoT predicted: `C`, correct
- CoT response excerpt: `The water bottle is to the right of the chip bag. There are no other objects or obstacles between the robot and the chip bag.`

Interpretation: Qwen CoT can help on individual object-blockage examples, but
the aggregate improvement is not enough to beat zero-shot overall.

### Qwen Spatial: CoT Uses Weak 2D Heuristics

Source:
`runs/eval/qwen25_3b_spatial_zeroshot_200_t0/results/qwen25_3b_spatial_zeroshot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_zero_shot_20260519_030705.json`
and
`runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260519_030446.json`

- `question_id`: `droid_pick_up_the_objects_from_the_table_and_put_them_on_the_plastic_5775_q17`
- `source_index`: 374
- Question: `In the image from ext1, which colored point is FARTHEST from the camera?`
- Expected: `B`
- Zero-shot predicted: `B`, correct
- CoT predicted: `D`, incorrect
- CoT response excerpt: `The distance from the camera to each point can be inferred from their positions.`

Interpretation: the reasoning text relies on 2D image position as a proxy for
depth. That is not reliable for these spatial questions.

### DeepSeek Spatial: CoT Helps One Depth Example

Source:
`runs/eval/deepseek_vl2_tiny_spatial_zeroshot_200_t0/results/deepseek_vl2_tiny_spatial_zeroshot_200_t0_deepseek-vl2-tiny_spatial_zero_shot_20260513_163639.json`
and
`runs/eval/deepseek_vl2_tiny_spatial_cot_200_t0/results/deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_20260513_163446.json`

- `question_id`: `droid_pick_up_the_objects_from_the_bowl_and_put_them_in_the_box_14259_q49`
- `source_index`: 370
- Question: `In the image from ext1, which colored point is CLOSEST to the camera?`
- Expected: `C`
- Zero-shot predicted: `A`, incorrect
- CoT predicted: `C`, correct
- CoT response excerpt: `The green point appears to be the nearest to the camera compared to the other colored points.`

Interpretation: DeepSeek gets some spatial benefit from CoT, but the net gain is
small: +2.5 percentage points on the full 200-example spatial subset.

### Qwen Sanity: Correctly Rejects Blank Image

Source:
`runs/diagnostics/image_understanding/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/image_understanding_sanity.json`

- `probe_type`: `explain_correct_blank`
- `question_id`: `droid_pick_up_the_object_5044_q7`
- Expected answer supplied to model: `A: Red`
- Image control: `blank`
- Response excerpt: `The current image is blank, which means it does not provide any visual information to support the answer.`
- Heuristic judgment: `rejects_or_insufficient`

Interpretation: Qwen can detect the strongest image ablation.

### Qwen Sanity: Rationalizes Wrong Answer

Source:
`runs/diagnostics/image_understanding/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/image_understanding_sanity.json`

- `probe_type`: `explain_wrong`
- `question_id`: `droid_pick_up_the_objects_and_put_them_in_the_box_1174_q15`
- Correct answer: `A: Red`
- Deliberately supplied wrong answer: `B: Purple`
- Response excerpt: `The proposed answer B: Purple is supported by the visible evidence in the image.`
- Heuristic judgment: `supports`

Interpretation: this is direct evidence of answer rationalization. The model is
not just failing to answer; it can justify a wrong answer when prompted to assess
it.

### Qwen Sanity: Fails A Shuffled-Image Control

Source:
`runs/diagnostics/image_understanding/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901_Qwen2.5-VL-3B-Instruct_20260517_140842/image_understanding_sanity.json`

- `probe_type`: `explain_correct_shuffle`
- `question_id`: `droid_pick_up_the_object_5044_q7`
- Original correct answer supplied to model: `A: Red`
- Image control: `shuffle`
- Response excerpt: `The image shows a robotic arm in a kitchen setting, with a frying pan on a table.`
- Heuristic judgment: `unclear`

Interpretation: Qwen did not flag the shuffled image as mismatched. It produced
a plausible explanation for the supplied answer on the wrong image.

### DeepSeek Sanity: Rejects Shuffled Image More Reliably

Source:
`runs/diagnostics/image_understanding/deepseek_vl2_tiny_spatial_cot_200_t0_deepseek-vl2-tiny_spatial_cot_20260513_163446_deepseek-vl2-tiny_20260517_141222/image_understanding_sanity.json`

- `probe_type`: `explain_correct_shuffle`
- `question_id`: `droid_pick_up_the_object_5044_q7`
- Original correct answer supplied to model: `A: Red`
- Image control: `shuffle`
- Response excerpt: `The current image does not visibly support the proposed answer.`
- Heuristic judgment: `rejects_or_insufficient`

Interpretation: DeepSeek is more sensitive to shuffled-image controls than Qwen
in this diagnostic, even though its scored spatial accuracy is still near chance.

## Conclusions

1. The evaluation framework is producing useful comparative evidence. We now
   have model, prompt, subset, and sanity-probe contrasts rather than just single
   headline accuracies.
2. Affordance is currently more promising than spatial, but this is mostly due
   to object-blockage questions. Grasp stability remains unresolved.
3. Spatial reasoning remains weak. The best spatial score is only 22.0%, which
   is close to 5-way chance.
4. CoT should not be assumed beneficial. It is neutral to slightly harmful for
   Qwen and dramatically harmful for DeepSeek affordance.
5. The sanity checks are valuable. They reveal that high-level accuracy can hide
   rationalization and weak visual grounding, especially for Qwen on wrong-answer
   and shuffled-image probes.

## Recommended Next Steps

1. Regenerate the older Qwen and spatial DeepSeek evals after the evaluator
   change that writes `curation_label` and `curation_subcategory` into every
   result row.
2. Add a small manual review sheet for the 20 sanity examples, because the
   current heuristic is useful for triage but too coarse for final claims.
3. Run subcategory-specific configs, especially `affordance_grasp_stability` and
   `spatial_direction`, to avoid hiding weak slices behind stronger ones.
4. Treat DeepSeek affordance CoT as a prompt-regression case and inspect whether
   the prompt format or answer extraction is causing the object-blockage collapse.
5. Try `rgb_depth` on the spatial distance/depth subset, then compare against
   RGB-only spatial runs using the same 200 source indices.
