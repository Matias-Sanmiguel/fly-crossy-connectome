import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export type FlyAssetLoader = (url: string) => Promise<THREE.Group>;

export type FlyModel = {
  group: THREE.Group;
  ready: Promise<void>;
  update(nowMilliseconds: number, hopProgress: number, moving: boolean): void;
  dispose(): void;
};

const FLY_ASSET_URL = 'assets/fly/fly-jeremy-rigged.glb';
const MODEL_SCALE = 0.16;

// The source model's neutral wing pose is already fully spread.
// Keep that horizontal spread at all times and only flap vertically.
const FLAP_AMPLITUDE = 0.58;
const FLAP_RADIANS_PER_MILLISECOND = 0.11;

const gltfLoader = new GLTFLoader();

const defaultLoader: FlyAssetLoader = async (url) => {
  const gltf = await gltfLoader.loadAsync(url);
  return gltf.scene;
};

function disposeHierarchy(root: THREE.Object3D): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();

  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    geometries.add(object.geometry);
    const meshMaterials = Array.isArray(object.material)
      ? object.material
      : [object.material];
    meshMaterials.forEach((material) => materials.add(material));
  });

  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
}

export function createFlyModel(
  loader: FlyAssetLoader = defaultLoader,
): FlyModel {
  const group = new THREE.Group();
  group.name = 'fly-jeremy';

  let visual: THREE.Group | null = null;
  let leftWing: THREE.Object3D | null = null;
  let rightWing: THREE.Object3D | null = null;
  let disposed = false;

  const ready = loader(FLY_ASSET_URL)
    .then((source) => {
      if (disposed) {
        disposeHierarchy(source);
        return;
      }

      visual = source;
      visual.name = 'fly-jeremy-model';
      visual.scale.setScalar(MODEL_SCALE);
      visual.position.set(-0.009, -0.327, 0.112);

      leftWing = visual.getObjectByName('wing-left-pivot') ?? null;
      rightWing = visual.getObjectByName('wing-right-pivot') ?? null;

      if (!leftWing || !rightWing) {
        disposeHierarchy(source);
        visual = null;
        leftWing = null;
        rightWing = null;
        throw new Error('Prepared fly asset is missing its wing pivots.');
      }

      group.add(visual);
    });

  const update = (
    nowMilliseconds: number,
    hopProgress: number,
    moving: boolean,
  ) => {
    if (!leftWing || !rightWing) return;

    void hopProgress;
    void moving;

    // Wings stay completely spread and flap continuously, even while idle.
    // Only the vertical beat changes; they never fold back toward the body.
    const flap =
      Math.sin(nowMilliseconds * FLAP_RADIANS_PER_MILLISECOND)
      * FLAP_AMPLITUDE;

    leftWing.rotation.set(0, 0, flap);
    rightWing.rotation.set(0, 0, -flap);
  };

  return {
    group,
    ready,
    update,
    dispose() {
      if (disposed) return;
      disposed = true;
      if (visual) {
        group.remove(visual);
        disposeHierarchy(visual);
      }
      group.clear();
      visual = null;
      leftWing = null;
      rightWing = null;
    },
  };
}
