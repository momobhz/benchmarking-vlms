import React from 'react';
import {Composition} from 'remotion';
import {EvalFlow} from './EvalFlow';
import {demoData} from './demo-data';

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="EvalFlow"
      component={EvalFlow}
      durationInFrames={1080}
      fps={30}
      width={3840}
      height={2160}
      defaultProps={{
        data: demoData,
      }}
    />
  );
};
