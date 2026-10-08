import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { KeyboardHighlightTarget } from './keyboardLayout.ts';

export const CHERRY_KEYCAP_PATH = 'assets/expo/keycaps/cherry-wasd.glb';
export const CHERRY_KEYCAP_NAMES = { w: 'cherry-r2-1u', a: 'cherry-r3-1u', s: 'cherry-r3-1u', d: 'cherry-r3-1u' } as const;
export type CherryKeycaps = Record<KeyboardHighlightTarget, THREE.BufferGeometry>;

/** One bounded shared geometry library. A/S/D reuse the exact R3 geometry. */
export function createCherryKeycapLibrary(load: () => Promise<THREE.Object3D>) {
  let pending: Promise<CherryKeycaps> | null = null;
  return () => {
    pending ??= load().then(scene => {
      const geometries = new Map<string, THREE.BufferGeometry>();
      for (const name of new Set(Object.values(CHERRY_KEYCAP_NAMES))) {
        const mesh = scene.getObjectByName(name) as THREE.Mesh | undefined;
        if (!mesh?.isMesh || !mesh.geometry.attributes.position?.count) throw Error(`Missing Cherry geometry: ${name}`);
        mesh.geometry.computeBoundingBox();
        const size = mesh.geometry.boundingBox!.getSize(new THREE.Vector3());
        if (!size.toArray().every(v => Number.isFinite(v) && v > 0)) throw Error('Invalid Cherry dimensions');
        geometries.set(name, mesh.geometry);
      }
      scene.traverse(object => {
        if ((object as THREE.Mesh).isMesh) {
          const material = (object as THREE.Mesh).material;
          for (const item of Array.isArray(material) ? material : [material]) item.dispose();
        }
      });
      return Object.fromEntries(Object.entries(CHERRY_KEYCAP_NAMES).map(([key, name]) => [key, geometries.get(name)!])) as CherryKeycaps;
    });
    return pending;
  };
}

export const loadCherryKeycaps = createCherryKeycapLibrary(async () => {
  const loaded = await new GLTFLoader().loadAsync(`${import.meta.env.BASE_URL}${CHERRY_KEYCAP_PATH}`);
  return loaded.scene;
});

/** Fit the letter to the real concave top, rather than floating a flat plane over it. */
export function createCherryKeycapLabel(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  const surface = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  surface.updateMatrixWorld(true);
  const label = new THREE.PlaneGeometry(0.034, 0.034, 8, 8).rotateX(-Math.PI / 2);
  const points = label.attributes.position;
  const ray = new THREE.Raycaster();
  const direction = new THREE.Vector3(0, -1, 0);
  for (let index = 0; index < points.count; index++) {
    ray.set(new THREE.Vector3(points.getX(index), 0.1, points.getZ(index)), direction);
    const hit = ray.intersectObject(surface, false)[0];
    if (!hit) { label.dispose(); surface.material.dispose(); throw Error('Letter does not fit Cherry surface'); }
    points.setY(index, hit.point.y + 0.0006);
  }
  surface.material.dispose();
  label.computeVertexNormals();
  return label;
}
