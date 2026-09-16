import * as THREE from 'three';

export type FlyModel = {
  group: THREE.Group;
  dispose(): void;
};

function segmentBetween(
  geometry: THREE.CylinderGeometry,
  material: THREE.Material,
  start: THREE.Vector3,
  end: THREE.Vector3,
  radius: number,
  name: string,
): THREE.Mesh {
  const direction = end.clone().sub(start);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = name;
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.scale.set(radius, direction.length(), radius);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.normalize(),
  );
  return mesh;
}

export function createFlyModel(): FlyModel {
  const group = new THREE.Group();
  group.name = 'stylized-low-poly-fly';

  const thoraxMaterial = new THREE.MeshStandardMaterial({
    color: 0x5f3922,
    roughness: 0.82,
  });
  const abdomenMaterial = new THREE.MeshStandardMaterial({
    color: 0x2a211b,
    roughness: 0.88,
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: 0xc58b32,
    roughness: 0.7,
  });
  const eyeMaterial = new THREE.MeshStandardMaterial({
    color: 0xb72e28,
    roughness: 0.38,
  });
  const limbMaterial = new THREE.MeshStandardMaterial({
    color: 0x211b17,
    roughness: 0.9,
  });
  const wingMaterial = new THREE.MeshStandardMaterial({
    color: 0xbce8e6,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
    depthWrite: false,
    roughness: 0.32,
  });

  const thorax = new THREE.Mesh(
    new THREE.DodecahedronGeometry(0.25, 0),
    thoraxMaterial,
  );
  thorax.name = 'fly-thorax';
  thorax.scale.set(0.92, 0.78, 1.08);
  thorax.position.z = -0.02;
  group.add(thorax);

  const abdomen = new THREE.Mesh(
    new THREE.SphereGeometry(0.24, 8, 6),
    abdomenMaterial,
  );
  abdomen.name = 'fly-abdomen';
  abdomen.scale.set(0.72, 0.62, 1.48);
  abdomen.position.z = 0.27;
  group.add(abdomen);

  for (const [index, z] of [0.12, 0.28].entries()) {
    const band = new THREE.Mesh(
      new THREE.TorusGeometry(0.17 - index * 0.018, 0.025, 4, 8),
      accentMaterial,
    );
    band.name = `fly-abdomen-band-${index}`;
    band.position.z = z;
    band.scale.y = 0.82;
    group.add(band);
  }

  const head = new THREE.Mesh(
    new THREE.DodecahedronGeometry(0.18, 0),
    thoraxMaterial,
  );
  head.name = 'fly-head';
  head.scale.set(1, 0.88, 0.86);
  head.position.set(0, 0.015, -0.32);
  group.add(head);

  const eyeGeometry = new THREE.SphereGeometry(0.09, 7, 5);
  const wingGeometry = new THREE.CircleGeometry(0.25, 7);
  for (const side of [-1, 1] as const) {
    const eye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    eye.name = `fly-eye-${side < 0 ? 'left' : 'right'}`;
    eye.scale.set(0.72, 0.95, 0.62);
    eye.position.set(side * 0.13, 0.055, -0.43);
    group.add(eye);

    const wing = new THREE.Mesh(wingGeometry, wingMaterial);
    wing.name = `fly-wing-${side < 0 ? 'left' : 'right'}`;
    wing.scale.set(1.22, 0.68, 1);
    wing.position.set(side * 0.27, 0.16, 0.07);
    wing.rotation.set(-Math.PI / 2, 0, side * -0.36);
    group.add(wing);
  }

  const segmentGeometry = new THREE.CylinderGeometry(1, 0.82, 1, 5);
  const legRows = [-0.2, 0, 0.2] as const;
  for (const [pair, z] of legRows.entries()) {
    for (const side of [-1, 1] as const) {
      const sideName = side < 0 ? 'left' : 'right';
      const start = new THREE.Vector3(side * 0.13, -0.08, z);
      const knee = new THREE.Vector3(
        side * (0.31 + pair * 0.025),
        -0.17,
        z + (pair - 1) * 0.07,
      );
      const foot = new THREE.Vector3(
        side * (0.5 + pair * 0.025),
        -0.32,
        z + (pair - 1) * 0.14,
      );
      group.add(segmentBetween(
        segmentGeometry,
        limbMaterial,
        start,
        knee,
        0.026,
        `fly-leg-${pair}-${sideName}-upper`,
      ));
      group.add(segmentBetween(
        segmentGeometry,
        limbMaterial,
        knee,
        foot,
        0.019,
        `fly-leg-${pair}-${sideName}-lower`,
      ));
    }
  }

  for (const side of [-1, 1] as const) {
    group.add(segmentBetween(
      segmentGeometry,
      limbMaterial,
      new THREE.Vector3(side * 0.07, 0.08, -0.43),
      new THREE.Vector3(side * 0.16, 0.15, -0.61),
      0.014,
      `fly-antenna-${side < 0 ? 'left' : 'right'}`,
    ));
  }

  return {
    group,
    dispose() {
      const geometries = new Set<THREE.BufferGeometry>();
      const materials = new Set<THREE.Material>();
      group.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        geometries.add(object.geometry);
        const meshMaterials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        meshMaterials.forEach((material) => materials.add(material));
      });
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
    },
  };
}
