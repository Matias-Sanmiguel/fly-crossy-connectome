import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

import { keyboardHighlightTargets, type KeyboardHighlightTarget } from '../expo/keyboardLayout.ts';
import type { Action } from '../game/types.ts';
import { asset } from '../lib/atlas.ts';

const KEY_POSITIONS: Readonly<Record<KeyboardHighlightTarget, readonly [number, number]>> = {
  w: [-0.112, -0.13],
  a: [-0.153, -0.053],
  d: [-0.067, -0.053],
  'arrow-up': [0.413, 0.016],
  'arrow-left': [0.374, 0.066],
  'arrow-right': [0.452, 0.066],
};

export function ExpoKeyboardScene({ action }: { action: Action }) {
  const host = useRef<HTMLDivElement>(null);
  const currentAction = useRef(action);
  const repaint = useRef<(() => void) | null>(null);
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  currentAction.current = action;

  useEffect(() => {
    repaint.current?.();
  }, [action]);

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    let disposed = false;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 10);
    camera.position.set(0.23, 1.02, 0.45);
    camera.lookAt(0.23, 0.025, -0.125);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    element.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x2033ff, 2.8));
    const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
    keyLight.position.set(-0.5, 1.2, 1);
    scene.add(keyLight);

    const keyGeometry = new THREE.BoxGeometry(0.034, 0.009, 0.034);
    const keyMaterials = new Map<KeyboardHighlightTarget, THREE.MeshStandardMaterial>();
    for (const [target, [x, z]] of Object.entries(KEY_POSITIONS) as [KeyboardHighlightTarget, readonly [number, number]][]) {
      const material = new THREE.MeshStandardMaterial({
        color: 0x2033ff,
        emissive: 0x2033ff,
        emissiveIntensity: 0.18,
        transparent: true,
        opacity: 0.2,
        roughness: 0.42,
      });
      const key = new THREE.Mesh(keyGeometry, material);
      key.position.set(x, 0.124, z);
      scene.add(key);
      keyMaterials.set(target, material);
    }

    const render = () => {
      if (disposed) return;
      const active = new Set(keyboardHighlightTargets(currentAction.current));
      for (const [target, material] of keyMaterials) {
        const selected = active.has(target);
        material.color.setHex(selected ? 0xb3ff3b : 0x2033ff);
        material.emissive.setHex(selected ? 0xb3ff3b : 0x2033ff);
        material.emissiveIntensity = selected ? 0.8 : 0.18;
        material.opacity = selected ? 0.94 : 0.2;
      }
      renderer.render(scene, camera);
    };
    repaint.current = render;

    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      const safeWidth = Math.max(1, width);
      const safeHeight = Math.max(1, height);
      renderer.setSize(safeWidth, safeHeight, false);
      camera.aspect = safeWidth / safeHeight;
      camera.updateProjectionMatrix();
      render();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(element);
    resize();

    const loader = new GLTFLoader();
    loader.load(
      asset('assets/expo/keyboard.glb'),
      (gltf) => {
        if (disposed) {
          gltf.scene.traverse((object) => {
            if (!(object instanceof THREE.Mesh)) return;
            object.geometry.dispose();
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            materials.forEach((material) => material.dispose());
          });
          return;
        }
        gltf.scene.traverse((object) => {
          if (!(object instanceof THREE.Mesh)) return;
          const sources = Array.isArray(object.material) ? object.material : [object.material];
          const materials = sources.map((source) => {
            const material = source.clone();
            if (material instanceof THREE.MeshStandardMaterial) material.roughness = 0.66;
            return material;
          });
          object.material = materials.length === 1 ? materials[0]! : materials;
        });
        scene.add(gltf.scene);
        setStatus('ready');
        render();
      },
      undefined,
      () => {
        if (!disposed) setStatus('error');
      },
    );

    return () => {
      disposed = true;
      repaint.current = null;
      observer.disconnect();
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        if (object.geometry !== keyGeometry) object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      });
      keyGeometry.dispose();
      keyMaterials.forEach((material) => material.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div ref={host} className="expo-keyboard-viewport" role="img" aria-label="Teclado 3D con las teclas activas resaltadas">
      {status !== 'ready' && (
        <span className="expo-media-status" role="status">
          {status === 'error' ? 'Teclado no disponible' : 'Cargando teclado'}
        </span>
      )}
    </div>
  );
}
