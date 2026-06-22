import React from 'react';
import {AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame} from 'remotion';
import type {DemoData} from './demo-data';
import './style.css';

type EvalFlowProps = {
  data: DemoData;
};

const FPS = 30;
const GRID_LEFT = 190;
const GRID_TOP = 390;
const GRID_WIDTH = 3460;
const GRID_COLUMNS = 10;
const GRID_GAP = 20;
const GRID_CELL_WIDTH = (GRID_WIDTH - GRID_GAP * (GRID_COLUMNS - 1)) / GRID_COLUMNS;
const GRID_CELL_HEIGHT = 168;

const timings = {
  image: 24,
  question: 120,
  response: 260,
  parse: 540,
  score: 660,
  grid: 748,
};

type GridSeed = {
  imagePath: string;
  question: string;
  expected: string;
  wrong: string;
};

type MiniEvaluationItem = {
  id: number;
  imagePath: string;
  question: string;
  expected: string;
  predicted: string;
  correct: boolean;
  progress: number;
};

const ease = (frame: number, start: number, end: number, from = 0, to = 1) =>
  interpolate(frame, [start, end], [from, to], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

const fade = (frame: number, start: number, duration = 28) => ease(frame, start, start + duration);

const smooth = (progress: number) => progress * progress * (3 - 2 * progress);

const softIn = (frame: number, start: number, duration = 34) => {
  const progress = fade(frame, start, duration);
  return {
    opacity: progress,
    transform: `translateY(${(1 - progress) * 24}px)`,
  };
};

const gridSeeds: GridSeed[] = [
  {imagePath: 'source-000369.png', question: 'Move direction?', expected: 'A', wrong: 'D'},
  {imagePath: 'source-000407.png', question: 'Farthest point?', expected: 'E', wrong: 'B'},
  {imagePath: 'source-000408.png', question: 'Same 3D point?', expected: 'E', wrong: 'A'},
  {imagePath: 'source-000000.png', question: 'Object location?', expected: 'B', wrong: 'D'},
  {imagePath: 'source-000407.png', question: 'Closest point?', expected: 'C', wrong: 'B'},
  {imagePath: 'source-000408.png', question: 'Point match?', expected: 'D', wrong: 'A'},
  {imagePath: 'source-000369.png', question: 'Next waypoint?', expected: 'A', wrong: 'C'},
  {imagePath: 'source-000000.png', question: 'Reachable object?', expected: 'A', wrong: 'E'},
  {imagePath: 'source-000368.png', question: 'Avoid obstacle?', expected: 'D', wrong: 'B'},
  {imagePath: 'source-000408.png', question: 'Camera match?', expected: 'C', wrong: 'E'},
];

const buildEvaluations = (data: DemoData): MiniEvaluationItem[] =>
  Array.from({length: 80}, (_, index) => {
    if (index === 0) {
      return {
        id: index,
        imagePath: data.imagePath,
        question: 'Move direction?',
        expected: data.expectedLetter,
        predicted: data.predictedLetter,
        correct: data.correct,
        progress: 1,
      };
    }

    const seed = gridSeeds[(index - 1) % gridSeeds.length];
    const correct = index % 6 !== 0 && index % 13 !== 0;

    return {
      id: index,
      imagePath: seed.imagePath,
      question: seed.question,
      expected: seed.expected,
      predicted: correct ? seed.expected : seed.wrong,
      correct,
      progress: ((index * 19) % 100) / 100,
    };
  });

const Header: React.FC = () => (
  <header className="header">
    <h1>How one VLM benchmark sample becomes a score</h1>
    <p>
      choose image → ask question → parse answer → repeat
    </p>
  </header>
);

const Question: React.FC<{data: DemoData}> = ({data}) => (
  <div className="question">
    <div className="section-label">2. ask a multiple-choice question</div>
    <h2>{data.task}</h2>
    <p>{data.question}</p>
    <div className="choice-list">
      {data.choices.map((choice) => (
        <span className={choice.letter === data.expectedLetter ? 'choice target' : 'choice'} key={choice.letter}>
          {choice.letter}. {choice.text}
        </span>
      ))}
    </div>
  </div>
);

const Response: React.FC<{data: DemoData}> = ({data}) => {
  const frame = useCurrentFrame();
  const visibleChars = Math.floor(ease(frame, timings.response + 24, timings.parse - 42, 0, data.responseExcerpt.length));
  const cursor = Math.floor(frame / 13) % 2 === 0 ? '_' : ' ';

  return (
    <div className="response">
      <div className="section-label">3. let the VLM answer</div>
      <pre>
        {data.responseExcerpt.slice(0, visibleChars)}
        {visibleChars < data.responseExcerpt.length ? cursor : ''}
      </pre>
    </div>
  );
};

const Parser: React.FC<{data: DemoData}> = ({data}) => {
  const frame = useCurrentFrame();
  const scan = ease(frame, timings.parse + 24, timings.parse + 86, 0, 1);
  const pop = spring({
    frame: Math.max(0, frame - timings.parse - 72),
    fps: FPS,
    config: {damping: 16, stiffness: 180},
  });

  return (
    <div className="parser">
      <div className="section-label">4. parse and compare</div>
      <div className="parse-line">
        <span className="parse-code">last option token</span>
        <span className="parse-arrow" style={{opacity: scan}}>
          →
        </span>
        <span className="parse-result" style={{transform: `scale(${0.82 + pop * 0.18})`}}>
          {data.predictedLetter}
        </span>
      </div>
      <div className="score-line" style={softIn(frame, timings.score, 30)}>
        parsed answer {data.predictedLetter} matches the answer key
        <span>score +1</span>
      </div>
    </div>
  );
};

const SingleEvaluation: React.FC<{data: DemoData}> = ({data}) => {
  const frame = useCurrentFrame();
  const exit = 1 - fade(frame, timings.grid - 28, 34);
  const imageStyle = softIn(frame, timings.image, 32);
  const questionStyle = softIn(frame, timings.question, 34);
  const responseStyle = softIn(frame, timings.response, 34);
  const parserStyle = softIn(frame, timings.parse, 34);

  return (
    <main
      className="single"
      style={{
        opacity: exit,
        transform: `scale(${1 - (1 - exit) * 0.06})`,
      }}
    >
      <div className="sample-image-wrap" style={imageStyle}>
        <Img className="sample-image" src={staticFile(data.imagePath)} />
        <div className="sample-caption">1. choose one benchmark image, source_index={data.sourceIndex}</div>
      </div>
      <div className="single-copy">
        <div style={questionStyle}>
          <Question data={data} />
        </div>
        <div style={responseStyle}>
          <Response data={data} />
        </div>
        <div style={parserStyle}>
          <Parser data={data} />
        </div>
      </div>
    </main>
  );
};

const MorphingSample: React.FC<{data: DemoData}> = ({data}) => {
  const frame = useCurrentFrame();
  const rawProgress = ease(frame, timings.grid - 28, timings.grid + 82);
  const progress = smooth(rawProgress);
  const opacity = fade(frame, timings.grid - 32, 14);
  const start = {
    left: 160,
    top: 570,
    width: 1450,
    height: 812,
  };
  const end = {
    left: GRID_LEFT,
    top: GRID_TOP,
    width: GRID_CELL_WIDTH,
    height: GRID_CELL_HEIGHT,
  };
  const left = start.left + (end.left - start.left) * progress;
  const top = start.top + (end.top - start.top) * progress;
  const width = start.width + (end.width - start.width) * progress;
  const height = start.height + (end.height - start.height) * progress;
  const imageWidth = start.width + (132 - start.width) * progress;
  const imageHeight = start.height + (74 - start.height) * progress;
  const imageTop = 14 * progress;
  const copyOpacity = fade(frame, timings.grid + 34, 24);

  return (
    <div
      className="morph-sample"
      style={{
        left,
        top,
        width,
        height,
        opacity,
        borderTopColor: `rgba(17, 17, 17, ${progress})`,
        paddingTop: 14 * progress,
      }}
    >
      <Img
        className="morph-image"
        src={staticFile(data.imagePath)}
        style={{
          top: imageTop,
          width: imageWidth,
          height: imageHeight,
          filter: `grayscale(${progress})`,
        }}
      />
      <div className="morph-copy" style={{opacity: copyOpacity}}>
        <div className="mini-id">eval 01</div>
        <div className="mini-question">Move direction?</div>
        <div className="mini-stream">
          <span style={{width: '100%'}} />
        </div>
        <div className="mini-parse">{data.predictedLetter} → +1</div>
      </div>
    </div>
  );
};

const MiniEvaluation: React.FC<{
  item: MiniEvaluationItem;
  index: number;
}> = ({item, index}) => {
  const frame = useCurrentFrame();
  const local = Math.max(0, frame - timings.grid);
  const delay = (index % 10) * 1.4 + Math.floor(index / 10) * 2;
  const responseWidth = ease(local, 28 + delay, 82 + delay, 14, 54 + item.progress * 46);
  const parseOpacity = fade(local, 72 + delay, 16);
  const tileOpacity = index === 0 ? 0 : fade(local, 0 + delay, 18);

  return (
    <div className={item.correct ? 'mini correct' : 'mini incorrect'} style={{opacity: tileOpacity}}>
      <Img className="mini-image" src={staticFile(item.imagePath)} />
      <div className="mini-text">
        <div className="mini-id">eval {String(index + 1).padStart(2, '0')}</div>
        <div className="mini-question">{item.question}</div>
        <div className="mini-stream">
          <span style={{width: `${responseWidth}%`}} />
        </div>
        <div className="mini-parse" style={{opacity: parseOpacity}}>
          {item.predicted} → {item.correct ? '+1' : '0'}
        </div>
      </div>
    </div>
  );
};

const ParallelGrid: React.FC<{data: DemoData}> = ({data}) => {
  const frame = useCurrentFrame();
  const evaluations = buildEvaluations(data);
  const gridOpacity = fade(frame, timings.grid - 6, 32);
  const zoom = ease(frame, timings.grid, timings.grid + 110, 2.15, 1);
  const accuracy = ease(frame, timings.grid + 84, timings.grid + 140, 0, data.runAccuracy);

  return (
    <div
      className="grid-scene"
      style={{
        left: GRID_LEFT,
        top: GRID_TOP,
        width: GRID_WIDTH,
        opacity: gridOpacity,
        transform: `scale(${zoom})`,
      }}
    >
      <div className="grid">
        {evaluations.map((item, index) => (
          <MiniEvaluation index={index} item={item} key={item.id} />
        ))}
      </div>
      <div className="grid-summary">
        <span>repeat the scoring loop; aggregate the results</span>
        <strong>{data.correctCount}/{data.totalExamples}</strong>
        <span>{accuracy.toFixed(1)}% accuracy</span>
      </div>
    </div>
  );
};

export const EvalFlow: React.FC<EvalFlowProps> = ({data}) => {
  return (
    <AbsoluteFill className="canvas">
      <Header />
      <SingleEvaluation data={data} />
      <ParallelGrid data={data} />
      <MorphingSample data={data} />
    </AbsoluteFill>
  );
};
