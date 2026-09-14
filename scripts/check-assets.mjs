import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const root = new URL('../public/', import.meta.url);
for (const [directory,manifestFile,field] of [['brain-atlas','manifest.json','exportSha256'],['flybody','checksums.json','sha256']]) {
  const manifest=JSON.parse(await readFile(new URL(`data/${directory}/${manifestFile}`,root),'utf8'));
  for(const [name,digest] of Object.entries(manifest[field])) {
    const bytes=await readFile(new URL(`data/${directory}/${name}`,root));
    if(createHash('sha256').update(bytes).digest('hex')!==digest) throw Error(`Asset checksum mismatch: ${directory}/${name}`);
  }
}
const connectomeManifest=JSON.parse(await readFile(new URL('data/connectome/manifest.json',root),'utf8'));
const connectomeBytes=await readFile(new URL('data/connectome/graph.json',root));
if(createHash('sha256').update(connectomeBytes).digest('hex')!==connectomeManifest.graphSha256) throw Error('Asset checksum mismatch: connectome/graph.json');
const connectomeNotice=await readFile(new URL('data/connectome/NOTICE.md',root),'utf8');
const protocolPath='docs/experiments/reduced-connectome-v1.md';
if(!connectomeNotice.includes(protocolPath)) throw Error(`Connectome notice must link to ${protocolPath}`);
await readFile(new URL(`../${protocolPath}`,root));
const upstreamBuilder='https://github.com/cobanov/flyjump/blob/c08c86bc18efd8125964b1d2ca4fc1df59700f30/scripts/build-connectome.py';
if(!connectomeNotice.includes(upstreamBuilder)) throw Error('Connectome notice must pin its upstream derivation script.');
console.log('Anatomical and connectome asset hashes verified.');
