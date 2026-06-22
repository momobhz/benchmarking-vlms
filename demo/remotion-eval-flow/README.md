# VLM Evaluation Flow Demo

This Remotion composition turns a saved benchmark result into a minimal 4K demo
video that shows:

1. a sample being selected,
2. a multiple-choice VQA prompt being sent to the VLM,
3. the model response being generated,
4. the answer letter being parsed, and
5. the score being updated,
6. a zoom-out to an 80-sample parallel evaluation grid.

The current sample comes from:

```text
runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901.json
```

## Run

```bash
npm install
npm run dev
```

## Render

```bash
npm run render
```

The MP4 is written to:

```text
out/eval-flow-4k.mp4
```

To swap in another example, update `src/demo-data.ts` and place the matching
image in `public/`.
