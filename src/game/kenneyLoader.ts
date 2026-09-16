import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import {
  GAME_ASSET_ROLES,
  KENNEY_ASSETS,
  type GameAssetRole,
  type KenneyAssetDefinition,
} from './kenneyAssets.ts';

export type KenneyAssetLoader = (url: string) => Promise<THREE.Group>;

export type GameAssetLibrary = {
  templates: ReadonlyMap<GameAssetRole, THREE.Group>;
  failures: ReadonlyMap<GameAssetRole, string>;
  clone(role: GameAssetRole): THREE.Group | null;
};

const gltfLoader = new GLTFLoader();
const defaultLoader: KenneyAssetLoader = async (url) => {
  const gltf = await gltfLoader.loadAsync(url);
  return gltf.scene;
};
const loaderCaches = new WeakMap<
  KenneyAssetLoader,
  Map<string, Promise<THREE.Group>>
>();

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return String(error);
}

function cachedLoad(loader: KenneyAssetLoader, path: string): Promise<THREE.Group> {
  let cache = loaderCaches.get(loader);
  if (!cache) {
    cache = new Map();
    loaderCaches.set(loader, cache);
  }

  let pending = cache.get(path);
  if (!pending) {
    pending = loader(path);
    cache.set(path, pending);
  }
  return pending;
}

function normalizedTemplate(
  source: THREE.Group,
  definition: KenneyAssetDefinition,
): THREE.Group {
  const content = source.clone(true);
  content.rotation.set(...definition.rotation);
  content.updateMatrixWorld(true);

  const bounds = new THREE.Box3().setFromObject(content);
  const measured = bounds.getSize(new THREE.Vector3());
  if (
    bounds.isEmpty()
    || ![measured.x, measured.y, measured.z].every(
      (value) => Number.isFinite(value) && value > Number.EPSILON,
    )
  ) {
    throw new Error('Asset geometry has empty or non-finite bounds.');
  }

  const uniformScale = Math.min(
    definition.logicalSize[0] / measured.x,
    definition.logicalSize[1] / measured.y,
    definition.logicalSize[2] / measured.z,
  );
  content.scale.multiplyScalar(uniformScale);
  content.updateMatrixWorld(true);

  bounds.setFromObject(content);
  const center = bounds.getCenter(new THREE.Vector3());
  content.position.set(
    -center.x,
    definition.verticalOffset - bounds.min.y,
    -center.z,
  );
  content.updateMatrixWorld(true);

  const template = new THREE.Group();
  template.add(content);
  template.updateMatrixWorld(true);
  template.name = `kenney:${definition.path}`;
  return template;
}

export async function loadKenneyAssets(
  loader: KenneyAssetLoader = defaultLoader,
): Promise<GameAssetLibrary> {
  const templates = new Map<GameAssetRole, THREE.Group>();
  const failures = new Map<GameAssetRole, string>();

  await Promise.all(GAME_ASSET_ROLES.map(async (role) => {
    const definition = KENNEY_ASSETS[role];
    try {
      const source = await cachedLoad(loader, definition.path);
      templates.set(role, normalizedTemplate(source, definition));
    } catch (error) {
      failures.set(role, errorMessage(error));
    }
  }));

  return {
    templates,
    failures,
    clone(role) {
      return templates.get(role)?.clone(true) ?? null;
    },
  };
}
