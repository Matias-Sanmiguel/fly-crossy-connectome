import * as THREE from 'three';

import { decorationsForRow } from './scenery.ts';
import { KENNEY_ASSETS, type GameAssetRole } from './kenneyAssets.ts';
import {
  loadKenneyAssets,
  type GameAssetLibrary,
  type KenneyAssetLoader,
} from './kenneyLoader.ts';
import {
  RENDER_HAZARD_CAPACITY,
  RENDER_LANE_CAPACITY,
  selectRenderableInstances,
} from './rendering.ts';
import { hazardPositionAt, HAZARD_CIRCUIT } from './simulation.ts';
import type { GameEvent, GameState, GridPosition } from './simulation.ts';
import type { HazardKind, LaneKind } from './types.ts';

export type GameAssetStatus = 'loading' | 'ready' | 'fallback';

export type GameRenderer = {
  render(state: GameState, events: readonly GameEvent[]): void;
  resize(): void;
  dispose(): void;
  readonly assetStatus: GameAssetStatus;
};

export type GameRendererOptions = {
  assetLoader?: KenneyAssetLoader;
  onAssetStatus?: (status: GameAssetStatus) => void;
};

const HOP_MILLISECONDS = 160;
const DECORATION_CAPACITY = RENDER_LANE_CAPACITY * 2;

function worldPosition(position: GridPosition, target: THREE.Vector3): THREE.Vector3 {
  return target.set(position.column, 0.42, -position.row);
}

function roleForLane(kind: LaneKind): GameAssetRole | null {
  if (kind === 'road') return 'lane.road';
  if (kind === 'rail') return 'lane.rail';
  return null;
}

function roleForHazard(kind: HazardKind): GameAssetRole | null {
  if (kind === 'car') return 'hazard.car';
  if (kind === 'truck') return 'hazard.truck';
  if (kind === 'train') return 'hazard.train';
  return null;
}

export function createGameRenderer(
  element: HTMLElement,
  options: GameRendererOptions = {},
): GameRenderer {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x08110f);
  scene.fog = new THREE.Fog(0x08110f, 13, 31);

  const camera = new THREE.OrthographicCamera(-7, 7, 7, -7, 0.1, 60);
  const webgl = new THREE.WebGLRenderer({ antialias: true });
  webgl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  webgl.outputColorSpace = THREE.SRGBColorSpace;
  webgl.toneMapping = THREE.ACESFilmicToneMapping;
  webgl.toneMappingExposure = 1.08;
  element.appendChild(webgl.domElement);

  scene.add(new THREE.HemisphereLight(0xc9ebff, 0x182824, 2.5));
  const keyLight = new THREE.DirectionalLight(0xffefcc, 3.2);
  keyLight.position.set(-4, 10, 6);
  scene.add(keyLight);
  const rimLight = new THREE.DirectionalLight(0x62d9ff, 1.1);
  rimLight.position.set(7, 5, -8);
  scene.add(rimLight);

  const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  const bodyGeometry = new THREE.SphereGeometry(0.28, 8, 6);
  const eyeGeometry = new THREE.SphereGeometry(0.075, 7, 5);
  const wingGeometry = new THREE.PlaneGeometry(0.46, 0.22);
  const laneMaterials: Record<LaneKind, THREE.MeshStandardMaterial> = {
    grass: new THREE.MeshStandardMaterial({ color: 0x4f7f43, roughness: 0.95 }),
    road: new THREE.MeshStandardMaterial({ color: 0x35423f, roughness: 0.9 }),
    rail: new THREE.MeshStandardMaterial({ color: 0x544b40, roughness: 0.95 }),
    river: new THREE.MeshStandardMaterial({ color: 0x24718a, roughness: 0.48 }),
  };
  const hazardMaterials: Record<HazardKind, THREE.MeshStandardMaterial> = {
    car: new THREE.MeshStandardMaterial({ color: 0xeb6b42, roughness: 0.62 }),
    truck: new THREE.MeshStandardMaterial({ color: 0xe3b341, roughness: 0.68 }),
    train: new THREE.MeshStandardMaterial({ color: 0xd9e1e6, roughness: 0.55 }),
    log: new THREE.MeshStandardMaterial({ color: 0x8a5a34, roughness: 1 }),
  };
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: 0x8f6235, roughness: 0.7 });
  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0xc53d32, roughness: 0.45 });
  const wingMaterial = new THREE.MeshStandardMaterial({
    color: 0xc7dce4,
    transparent: true,
    opacity: 0.58,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const createInstances = (material: THREE.Material, maximum: number) => {
    const mesh = new THREE.InstancedMesh(boxGeometry, material, maximum);
    mesh.count = 0;
    scene.add(mesh);
    return mesh;
  };
  const laneMeshes: Record<LaneKind, THREE.InstancedMesh> = {
    grass: createInstances(laneMaterials.grass, RENDER_LANE_CAPACITY),
    road: createInstances(laneMaterials.road, RENDER_LANE_CAPACITY),
    rail: createInstances(laneMaterials.rail, RENDER_LANE_CAPACITY),
    river: createInstances(laneMaterials.river, RENDER_LANE_CAPACITY),
  };
  const hazardMeshes: Record<HazardKind, THREE.InstancedMesh> = {
    car: createInstances(hazardMaterials.car, RENDER_HAZARD_CAPACITY),
    truck: createInstances(hazardMaterials.truck, RENDER_HAZARD_CAPACITY),
    train: createInstances(hazardMaterials.train, RENDER_HAZARD_CAPACITY),
    log: createInstances(hazardMaterials.log, RENDER_HAZARD_CAPACITY),
  };

  const fly = new THREE.Group();
  const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
  body.scale.set(0.75, 0.72, 1.35);
  fly.add(body);
  const head = new THREE.Mesh(bodyGeometry, bodyMaterial);
  head.scale.setScalar(0.72);
  head.position.z = -0.3;
  fly.add(head);
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    eye.position.set(side * 0.13, 0.055, -0.45);
    fly.add(eye);
    const wing = new THREE.Mesh(wingGeometry, wingMaterial);
    wing.position.set(side * 0.29, 0.13, 0.03);
    wing.rotation.set(-Math.PI / 2, 0, side * -0.28);
    fly.add(wing);
  }
  scene.add(fly);

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const composeInstance = (
    mesh: THREE.InstancedMesh,
    index: number,
    x: number,
    y: number,
    z: number,
    sizeX: number,
    sizeY: number,
    sizeZ: number,
  ) => {
    position.set(x, y, z);
    scale.set(sizeX, sizeY, sizeZ);
    matrix.compose(position, rotation, scale);
    mesh.setMatrixAt(index, matrix);
  };

  let latest: { state: GameState; events: readonly GameEvent[] } | null = null;
  let renderedStep = -1;
  let renderedSeed = '';
  let cameraRow = 0;
  let hopStarted = performance.now() - HOP_MILLISECONDS;
  const hopFrom = new THREE.Vector3();
  const hopTo = new THREE.Vector3();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let library: GameAssetLibrary | null = null;
  let status: GameAssetStatus = 'loading';
  let disposed = false;
  const pools = new Map<GameAssetRole, THREE.Group[]>();
  const poolUsage = new Map<GameAssetRole, number>();

  const publishStatus = (next: GameAssetStatus) => {
    if (status === next) return;
    status = next;
    options.onAssetStatus?.(next);
  };
  options.onAssetStatus?.(status);

  const resetPools = () => {
    poolUsage.clear();
    for (const groups of pools.values()) {
      for (const group of groups) group.visible = false;
    }
  };

  const acquire = (role: GameAssetRole): THREE.Group | null => {
    if (!library) return null;
    const used = poolUsage.get(role) ?? 0;
    const maximum = role.startsWith('decoration.')
      ? DECORATION_CAPACITY
      : role.startsWith('lane.')
        ? RENDER_LANE_CAPACITY
        : RENDER_HAZARD_CAPACITY;
    if (used >= maximum) return null;

    let groups = pools.get(role);
    if (!groups) {
      groups = [];
      pools.set(role, groups);
    }
    let group: THREE.Group | undefined = groups[used];
    if (!group) {
      const clone = library.clone(role);
      if (!clone) return null;
      group = clone;
      groups.push(group);
      scene.add(group);
    }
    poolUsage.set(role, used + 1);
    group.visible = true;
    group.position.set(0, 0, 0);
    group.rotation.set(0, 0, 0);
    group.scale.set(1, 1, 1);
    return group;
  };

  const rebuildInstances = (game: GameState) => {
    const laneCounts: Record<LaneKind, number> = { grass: 0, road: 0, rail: 0, river: 0 };
    const hazardCounts: Record<HazardKind, number> = { car: 0, truck: 0, train: 0, log: 0 };
    const renderable = selectRenderableInstances(game);
    resetPools();

    for (const lane of renderable.lanes) {
      const laneRole = roleForLane(lane.kind);
      const laneModel = laneRole ? acquire(laneRole) : null;
      if (laneModel) {
        laneModel.position.set(0, -0.1, -lane.row);
      } else {
        const laneIndex = laneCounts[lane.kind]++;
        composeInstance(
          laneMeshes[lane.kind],
          laneIndex,
          0,
          -0.12,
          -lane.row,
          HAZARD_CIRCUIT,
          0.2,
          0.94,
        );
      }

      for (const decoration of decorationsForRow(game.seed, lane.row, lane.kind)) {
        const model = acquire(decoration.role);
        if (!model) continue;
        model.position.set(decoration.column, 0, -lane.row);
        model.rotation.y = decoration.rotationY;
        model.scale.setScalar(decoration.scale);
      }
    }

    for (const { lane, hazard } of renderable.hazards) {
      const x = hazardPositionAt(lane, hazard, game.time);
      const hazardRole = roleForHazard(hazard.kind);
      const hazardModel = hazardRole ? acquire(hazardRole) : null;
      if (hazardModel && hazardRole) {
        hazardModel.position.set(x, 0, -lane.row);
        hazardModel.rotation.y = lane.direction === -1 ? Math.PI : 0;
        hazardModel.scale.x = hazard.size / KENNEY_ASSETS[hazardRole].logicalSize[0];
      } else {
        const hazardIndex = hazardCounts[hazard.kind]++;
        const isLog = hazard.kind === 'log';
        const height = hazard.kind === 'train' ? 0.72 : isLog ? 0.25 : 0.48;
        const depth = hazard.kind === 'train' ? 0.72 : isLog ? 0.48 : 0.58;
        composeInstance(
          hazardMeshes[hazard.kind],
          hazardIndex,
          x,
          height / 2,
          -lane.row,
          hazard.size,
          height,
          depth,
        );
      }
    }

    for (const kind of Object.keys(laneMeshes) as LaneKind[]) {
      laneMeshes[kind].count = laneCounts[kind];
      laneMeshes[kind].instanceMatrix.needsUpdate = true;
    }
    for (const kind of Object.keys(hazardMeshes) as HazardKind[]) {
      hazardMeshes[kind].count = hazardCounts[kind];
      hazardMeshes[kind].instanceMatrix.needsUpdate = true;
    }
  };

  void loadKenneyAssets(options.assetLoader).then((assets) => {
    if (disposed) return;
    library = assets;
    publishStatus(assets.failures.size === 0 ? 'ready' : 'fallback');
    if (latest) rebuildInstances(latest.state);
  }).catch(() => {
    if (!disposed) publishStatus('fallback');
  });

  const resize = () => {
    const { width, height } = element.getBoundingClientRect();
    const safeWidth = Math.max(1, width);
    const safeHeight = Math.max(1, height);
    webgl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    webgl.setSize(safeWidth, safeHeight, false);
    const halfHeight = 5.5;
    const aspect = safeWidth / safeHeight;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.left = -halfHeight * aspect;
    camera.right = halfHeight * aspect;
    camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(element);
  resize();

  let frame = 0;
  let previous = performance.now();
  const animate = (now: number) => {
    const game = latest?.state;
    if (game) {
      const elapsed = Math.min(1, Math.max(0, (now - hopStarted) / HOP_MILLISECONDS));
      const eased = elapsed * elapsed * (3 - 2 * elapsed);
      fly.position.lerpVectors(hopFrom, hopTo, eased);
      fly.position.y += Math.sin(elapsed * Math.PI) * 0.38;

      const deltaSeconds = Math.min(0.05, Math.max(0, (now - previous) / 1000));
      const cameraEase = 1 - Math.exp(-5 * deltaSeconds);
      cameraRow += (game.fly.row - cameraRow) * cameraEase;
      const focusZ = -cameraRow;
      camera.position.set(9, 10, focusZ + 9);
      camera.lookAt(0, 0, focusZ - 0.75);

      if (!document.hidden) webgl.render(scene, camera);
    }
    previous = now;
    frame = requestAnimationFrame(animate);
  };
  frame = requestAnimationFrame(animate);

  return {
    render(state, events) {
      latest = { state, events };
      if (state.step === renderedStep && state.seed === renderedSeed) return;

      const movement = events.find((event) => event.type === 'moved');
      if (renderedStep < 0 || state.step === 0) {
        worldPosition(state.fly, hopFrom);
        hopTo.copy(hopFrom);
        fly.position.copy(hopFrom);
        cameraRow = state.fly.row;
      } else {
        if (movement) worldPosition(movement.from, hopFrom);
        else hopFrom.copy(fly.position);
        worldPosition(state.fly, hopTo);
      }
      hopStarted = reducedMotion.matches
        ? performance.now() - HOP_MILLISECONDS
        : performance.now();
      renderedStep = state.step;
      renderedSeed = state.seed;
      rebuildInstances(state);
    },
    resize,
    dispose() {
      if (disposed) return;
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      boxGeometry.dispose();
      bodyGeometry.dispose();
      eyeGeometry.dispose();
      wingGeometry.dispose();
      Object.values(laneMaterials).forEach((material) => material.dispose());
      Object.values(hazardMaterials).forEach((material) => material.dispose());
      bodyMaterial.dispose();
      eyeMaterial.dispose();
      wingMaterial.dispose();
      webgl.dispose();
      webgl.domElement.remove();
      scene.clear();
    },
    get assetStatus() {
      return status;
    },
  };
}
