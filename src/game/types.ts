export type Action = 'forward' | 'backward' | 'left' | 'right' | 'wait';

export type LaneKind = 'grass' | 'road' | 'rail' | 'river';

export type Direction = -1 | 1;

export type HazardKind = 'car' | 'truck' | 'train' | 'log';

export type Hazard = {
  kind: HazardKind;
  position: number;
  size: number;
};

export type Lane = {
  row: number;
  kind: LaneKind;
  direction?: Direction;
  speed?: number;
  phase?: number;
  hazards: Hazard[];
};

export type WorldChunk = {
  from: number;
  rows: Lane[];
};

export type Rng = {
  next(): number;
  integer(min: number, max: number): number;
  pick<T>(values: readonly T[]): T;
};
