import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import {
  RENDER_HAZARD_CAPACITY,
  RENDER_LANE_CAPACITY,
  selectRenderableInstances,
} from '../game/rendering';
import { hazardPositionAt, HAZARD_CIRCUIT } from '../game/simulation';
import type { GameEvent, GameState, GridPosition } from '../game/simulation';
import type { HazardKind, LaneKind } from '../game/types';

type GameSceneProps = {
  state: GameState;
  events: GameEvent[];
};

const HOP_MILLISECONDS = 160;

function worldPosition(position: GridPosition, target: THREE.Vector3): THREE.Vector3 {
  return target.set(position.column, 0.42, -position.row);
}

export function GameScene({ state, events }: GameSceneProps) {
  const host = useRef<HTMLDivElement>(null);
  const latest = useRef({ state, events });
  latest.current = { state, events };

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1518);
    scene.fog = new THREE.Fog(0x0d1518, 12, 28);

    const camera = new THREE.OrthographicCamera(-7, 7, 7, -7, 0.1, 60);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    element.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xc9ebff, 0x1d2520, 2.4));
    const keyLight = new THREE.DirectionalLight(0xffefcc, 3.2);
    keyLight.position.set(-4, 10, 6);
    scene.add(keyLight);

    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const bodyGeometry = new THREE.SphereGeometry(0.28, 8, 6);
    const eyeGeometry = new THREE.SphereGeometry(0.075, 7, 5);
    const wingGeometry = new THREE.PlaneGeometry(0.46, 0.22);
    const laneMaterials: Record<LaneKind, THREE.MeshStandardMaterial> = {
      grass: new THREE.MeshStandardMaterial({ color: 0x47733c, roughness: 0.95 }),
      road: new THREE.MeshStandardMaterial({ color: 0x303941, roughness: 0.9 }),
      rail: new THREE.MeshStandardMaterial({ color: 0x4d443b, roughness: 0.95 }),
      river: new THREE.MeshStandardMaterial({ color: 0x255d72, roughness: 0.55 }),
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

    let renderedStep = state.step;
    let renderedSeed = state.seed;
    let cameraRow = state.fly.row;
    let hopStarted = performance.now() - HOP_MILLISECONDS;
    const hopFrom = worldPosition(state.fly, new THREE.Vector3());
    const hopTo = hopFrom.clone();
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    const rebuildInstances = (game: GameState) => {
      const laneCounts: Record<LaneKind, number> = { grass: 0, road: 0, rail: 0, river: 0 };
      const hazardCounts: Record<HazardKind, number> = { car: 0, truck: 0, train: 0, log: 0 };
      const renderable = selectRenderableInstances(game);

      for (const lane of renderable.lanes) {
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

      for (const { lane, hazard } of renderable.hazards) {
        const hazardIndex = hazardCounts[hazard.kind]++;
        const isLog = hazard.kind === 'log';
        const height = hazard.kind === 'train' ? 0.72 : isLog ? 0.25 : 0.48;
        const depth = hazard.kind === 'train' ? 0.72 : isLog ? 0.48 : 0.58;
        composeInstance(
          hazardMeshes[hazard.kind],
          hazardIndex,
          hazardPositionAt(lane, hazard, game.time),
          height / 2,
          -lane.row,
          hazard.size,
          height,
          depth,
        );
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

    rebuildInstances(state);
    worldPosition(state.fly, fly.position);

    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      const safeWidth = Math.max(1, width);
      const safeHeight = Math.max(1, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.setSize(safeWidth, safeHeight, false);
      const aspect = safeWidth / safeHeight;
      const halfHeight = 5.5;
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
      const game = latest.current.state;
      if (game.step !== renderedStep || game.seed !== renderedSeed) {
        const movement = latest.current.events.find((event) => event.type === 'moved');
        if (game.step === 0) {
          worldPosition(game.fly, hopFrom);
          hopTo.copy(hopFrom);
        } else {
          if (movement) worldPosition(movement.from, hopFrom);
          else hopFrom.copy(fly.position);
          worldPosition(game.fly, hopTo);
        }
        hopStarted = reducedMotion.matches ? now - HOP_MILLISECONDS : now;
        renderedStep = game.step;
        renderedSeed = game.seed;
        rebuildInstances(game);
      }

      const elapsed = Math.min(1, Math.max(0, (now - hopStarted) / HOP_MILLISECONDS));
      const eased = elapsed * elapsed * (3 - 2 * elapsed);
      fly.position.lerpVectors(hopFrom, hopTo, eased);
      fly.position.y += Math.sin(elapsed * Math.PI) * 0.38;

      const deltaSeconds = Math.min(0.05, Math.max(0, (now - previous) / 1000));
      previous = now;
      const cameraEase = 1 - Math.exp(-5 * deltaSeconds);
      cameraRow += (game.fly.row - cameraRow) * cameraEase;
      const focusZ = -cameraRow;
      camera.position.set(9, 10, focusZ + 9);
      camera.lookAt(0, 0, focusZ - 0.75);

      if (!document.hidden) renderer.render(scene, camera);
      frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);

    return () => {
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
      renderer.dispose();
      renderer.domElement.remove();
      scene.clear();
    };
  }, []);

  return <div ref={host} className="three-viewport game-viewport" aria-label="Isometric fly crossing game" />;
}
