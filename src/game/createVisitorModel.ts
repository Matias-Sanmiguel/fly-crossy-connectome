import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import type { FlyModel } from './createFlyModel.ts';
import { asset } from '../lib/atlas.ts';

const PLAYER_ASSET = 'assets/play/character-male-c.glb';
const PLAYER_HEIGHT = 0.96;
// createGameRenderer positions the player pivot 0.42 units above the ground.
const PLAYER_PIVOT_HEIGHT = 0.42;
const gltfLoader = new GLTFLoader();

function disposeModel(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    geometries.add(object.geometry);
    const meshMaterials = Array.isArray(object.material)
      ? object.material : [object.material];
    meshMaterials.forEach((material) => materials.add(material));
  });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
}

/** A visual-only replacement, using the exact same group/motion API as the fly. */
export function createVisitorModel(): FlyModel {
  const group = new THREE.Group();
  group.name = 'visitor-kenney-character';

  let disposed = false;
  let visual: THREE.Group | null = null;
  let mixer: THREE.AnimationMixer | null = null;
  let idleAction: THREE.AnimationAction | null = null;
  let walkAction: THREE.AnimationAction | null = null;
  let activeAction: THREE.AnimationAction | null = null;
  let previousNow: number | null = null;

  const ready: Promise<void> = gltfLoader.loadAsync(asset(PLAYER_ASSET))
    .then((gltf) => {
      const character = gltf.scene;
      if (disposed) {
        disposeModel(character);
        return;
      }

      // Kenney characters are Y-up. Face toward the forward (-Z) lane direction.
      character.rotation.y = Math.PI;
      character.updateMatrixWorld(true);
      const bounds = new THREE.Box3().setFromObject(character);
      const height = bounds.getSize(new THREE.Vector3()).y;
      if (bounds.isEmpty() || !Number.isFinite(height) || height <= 1e-6) {
        disposeModel(character);
        throw new Error('Kenney GLB contains no valid character geometry.');
      }

      // Keep the logical character in roughly one Crossy grid cell.
      character.scale.multiplyScalar(PLAYER_HEIGHT / height);
      character.updateMatrixWorld(true);
      const fitted = new THREE.Box3().setFromObject(character);
      const center = fitted.getCenter(new THREE.Vector3());
      character.position.x -= center.x;
      character.position.y -= fitted.min.y + PLAYER_PIVOT_HEIGHT;
      character.position.z -= center.z;
      group.add(character);
      visual = character;

      // The GLB ships with idle/walk clips. No gameplay logic is changed.
      const idle = gltf.animations.find((clip) => clip.name.toLowerCase() === 'idle');
      const walk = gltf.animations.find((clip) => clip.name.toLowerCase() === 'walk');
      if (idle || walk) {
        mixer = new THREE.AnimationMixer(character);
        idleAction = idle ? mixer.clipAction(idle) : null;
        walkAction = walk ? mixer.clipAction(walk) : null;
        activeAction = idleAction ?? walkAction;
        activeAction?.play();
      }
    })
    .catch((error: unknown) => {
      console.error('Could not load the visitor Kenney GLB:', error);
    });

  return {
    group,
    ready,
    update(nowMilliseconds, hopProgress, moving) {
      const deltaSeconds = previousNow === null
        ? 0
        : Math.min(0.05, Math.max(0, (nowMilliseconds - previousNow) / 1000));
      previousNow = nowMilliseconds;
      if (!mixer) return;

      const next = moving && hopProgress < 1
        ? (walkAction ?? idleAction)
        : (idleAction ?? walkAction);
      if (next && next !== activeAction) {
        activeAction?.fadeOut(0.08);
        next.reset().fadeIn(0.08).play();
        activeAction = next;
      }
      mixer.update(deltaSeconds);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      mixer?.stopAllAction();
      if (mixer && visual) mixer.uncacheRoot(visual);
      if (visual) {
        group.remove(visual);
        disposeModel(visual);
      }
      group.clear();
      visual = null;
      mixer = null;
      idleAction = null;
      walkAction = null;
      activeAction = null;
    },
  };
}
