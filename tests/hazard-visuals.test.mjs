import assert from 'node:assert/strict';
import test from 'node:test';

import {
  visualRoleForHazard,
} from '../src/game/hazardVisuals.ts';

test(
  'hazard visuals are deterministic and preserve logical kind',
  () => {
    const car = {
      kind: 'car',
      position: 4,
      size: 2,
    };

    const truck = {
      kind: 'truck',
      position: -6,
      size: 3,
    };

    const firstCar =
      visualRoleForHazard(
        'visual-test',
        8,
        car,
      );

    const secondCar =
      visualRoleForHazard(
        'visual-test',
        8,
        car,
      );

    assert.equal(
      firstCar,
      secondCar,
    );

    assert.match(
      firstCar,
      /^hazard\.car\./,
    );

    assert.match(
      visualRoleForHazard(
        'visual-test',
        9,
        truck,
      ),
      /^hazard\.truck\./,
    );

    assert.equal(
      visualRoleForHazard(
        'visual-test',
        10,
        {
          kind: 'train',
          position: 0,
          size: 4,
        },
      ),
      'hazard.train',
    );

    assert.equal(
      visualRoleForHazard(
        'visual-test',
        11,
        {
          kind: 'log',
          position: 0,
          size: 2,
        },
      ),
      null,
    );
  },
);

test(
  'different hazards produce visual variety',
  () => {
    const appearances = new Set();

    for (
      let position = -12;
      position <= 12;
      position += 1
    ) {
      appearances.add(
        visualRoleForHazard(
          'variety-test',
          position,
          {
            kind: 'car',
            position,
            size: 2,
          },
        ),
      );
    }

    assert.ok(
      appearances.size >= 4,
      `expected visual variety, got ${
        [...appearances].join(', ')
      }`,
    );
  },
);