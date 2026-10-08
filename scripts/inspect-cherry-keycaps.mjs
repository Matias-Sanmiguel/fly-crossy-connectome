import { readFile } from 'node:fs/promises';
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';

const bytes = await readFile(new URL('../artifacts/cherry-keycaps/full.fbx', import.meta.url));
const scene = new FBXLoader().parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), '');
scene.updateMatrixWorld(true);
const meshes = [];
scene.traverse(object => {
  if (!object.isMesh || (process.argv.length > 2 && !process.argv.slice(2).includes(object.name))) return;
  const box = new THREE.Box3().setFromObject(object);
  meshes.push({ name: object.name, parent: object.parent?.name, vertices: object.geometry.attributes.position.count,
    min: box.min.toArray(), size: box.getSize(new THREE.Vector3()).toArray() });
});
console.log(JSON.stringify(meshes, null, 2));
