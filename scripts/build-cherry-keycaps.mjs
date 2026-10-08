import { createHash } from 'node:crypto';
import { basename } from 'node:path';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';

// GLTFExporter uses FileReader to assemble its binary Blob in browser environments.
globalThis.FileReader = class {
  readAsArrayBuffer(blob) {
    blob.arrayBuffer().then(result => { this.result = result; this.onloadend?.(); });
  }
};
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
if (!process.argv[2]) throw Error('Supply the original user-provided archive path.');
const staging = new URL('../artifacts/cherry-keycaps/', import.meta.url);
const destination = new URL('../public/assets/expo/keycaps/', import.meta.url);
const fbx = await readFile(new URL('full.fbx', staging));
const source = new FBXLoader().parse(fbx.buffer.slice(fbx.byteOffset, fbx.byteOffset + fbx.byteLength), '');
source.updateMatrixWorld(true);
const exported = new THREE.Group();
const models = [];
for (const [sourceName, name, keys] of [['R2_1u', 'cherry-r2-1u', ['w']], ['R3_1u', 'cherry-r3-1u', ['a', 's', 'd']]]) {
  const original = source.getObjectByName(sourceName);
  if (!original?.isMesh) throw Error(`Missing source mesh: ${sourceName}`);
  let geometry = original.geometry.clone().applyMatrix4(original.matrixWorld).rotateX(-Math.PI / 2);
  geometry.clearGroups(); // Discard invalid FBX material indices; runtime assigns the Expo material.
  geometry.computeBoundingBox();
  const center = geometry.boundingBox.getCenter(new THREE.Vector3());
  const size = geometry.boundingBox.getSize(new THREE.Vector3());
  geometry.translate(-center.x, -center.y, -center.z).scale(...Array(3).fill(0.086 / size.x));
  geometry = mergeVertices(geometry);
  geometry.computeBoundingBox();
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: 0xf3f5ff, roughness: 0.34, metalness: 0.04 }));
  mesh.name = name;
  exported.add(mesh);
  models.push({ sourceName, name, keys, vertices: geometry.attributes.position.count,
    triangles: geometry.index.count / 3, size: geometry.boundingBox.getSize(new THREE.Vector3()).toArray() });
}
const glb = Buffer.from(await new GLTFExporter().parseAsync(exported, { binary: true, onlyVisible: true }));
await mkdir(destination, { recursive: true });
await writeFile(new URL('cherry-wasd.glb', destination), glb);
await copyFile(new URL('LICENSE', staging), new URL('LICENSE', destination));
await copyFile(new URL('README.md', staging), new URL('SOURCE-README.md', destination));
const manifest = {
  version: 1, author: 'Santiago Castelo', license: 'MIT', sourceUrl: 'https://github.com/endeavoursc/cherry-mx-keycaps',
  sourceArchive: { file: basename(process.argv[2]), sha256: hash(await readFile(process.argv[2])) },
  sourceFile: { file: 'Cherry Keycaps Full.fbx', sha256: hash(fbx) },
  output: { file: 'cherry-wasd.glb', sha256: hash(glb), bytes: glb.length }, models,
  modifications: 'Extract R2_1u/R3_1u only; bake source transforms; rotate Z-up to Y-up; center; uniform scale to 0.086 width; weld duplicate vertices preserving attributes; replace invalid source material indices. Letters and press animation are applied at runtime.',
};
await writeFile(new URL('manifest.json', destination), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest));
