import * as THREE from 'three';

import type { GameAssetLibrary } from './kenneyLoader.ts';
import { tiledLaneRowLayout } from './laneVisuals.ts';

export type TiledLaneRole = 'lane.road' | 'lane.rail';

export type RailSurfaceMode = 'kenney' | 'procedural';

export function railSurfaceMode(assetAvailable: boolean): RailSurfaceMode {
  return assetAvailable ? 'kenney' : 'procedural';
}

export function createTiledLaneRow(
  role: TiledLaneRole,
  library: GameAssetLibrary,
  circuitLength: number,
): THREE.Group | null {
  if (!library.templates.has(role)) return null;

  const row = new THREE.Group();
  row.name = `tiled-row:${role}`;
  for (const tileLayout of tiledLaneRowLayout(circuitLength)) {
    const tile = library.clone(role);
    if (!tile) return null;
    tile.position.set(...tileLayout.position);
    row.add(tile);
  }
  return row;
}
