export type Choice = {
  letter: string;
  text: string;
};

export type DemoData = {
  runName: string;
  resultPath: string;
  model: string;
  dataset: string;
  subset: string;
  promptMode: string;
  imageMode: string;
  totalExamples: number;
  runAccuracy: number;
  correctCount: number;
  questionId: string;
  sourceIndex: number;
  responseTimeSeconds: number;
  imagePath: string;
  task: string;
  question: string;
  choices: Choice[];
  expectedLetter: string;
  expectedText: string;
  predictedLetter: string;
  predictedText: string;
  correct: boolean;
  responseExcerpt: string;
};

export const demoData: DemoData = {
  runName: 'qwen25_3b_spatial_cot_200_t0',
  resultPath:
    'runs/eval/qwen25_3b_spatial_cot_200_t0/results/qwen25_3b_spatial_cot_200_t0_Qwen2.5-VL-3B-Instruct_spatial_cot_20260513_170901.json',
  model: 'Qwen/Qwen2.5-VL-3B-Instruct',
  dataset: 'keplerccc/Robo2VLM-1',
  subset: 'spatial',
  promptMode: 'chain-of-thought',
  imageMode: 'rgb',
  totalExamples: 200,
  runAccuracy: 19.5,
  correctCount: 39,
  questionId: 'droid_pick_up_the_object_5044_q7',
  sourceIndex: 368,
  responseTimeSeconds: 8.31,
  imagePath: 'source-000368.png',
  task: 'The robot task is to pick up the object.',
  question: 'Which colored arrow correctly shows the direction the robot will move next?',
  choices: [
    {letter: 'A', text: 'Red'},
    {letter: 'B', text: 'Green'},
    {letter: 'C', text: 'Blue'},
    {letter: 'D', text: 'Yellow'},
    {letter: 'E', text: 'Purple'},
  ],
  expectedLetter: 'C',
  expectedText: 'Blue',
  predictedLetter: 'C',
  predictedText: 'Blue',
  correct: true,
  responseExcerpt:
    'Reasoning:\n\n- The task is to pick up the object.\n- The blue arrow points toward the object in the image.\n\nFinal answer: C. Blue',
};
