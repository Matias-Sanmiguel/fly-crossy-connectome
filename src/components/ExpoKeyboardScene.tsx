import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';

import { keyboardHighlightTargets, type KeyboardHighlightTarget } from '../expo/keyboardLayout.ts';
import { createCherryKeycapLabel, loadCherryKeycaps } from '../expo/cherryKeycaps.ts';
import type { Action } from '../game/types.ts';

const KEY_POSITIONS: Readonly<Record<KeyboardHighlightTarget, readonly [number, number]>> = {
  w: [0, -0.06],
  a: [-0.1, 0.05],
  s: [0, 0.05],
  d: [0.1, 0.05],
};

const RESTING_KEY_Y = 0.05;
const PRESSED_KEY_Y = 0.036;

function createKeyLabel(letter: string) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas unavailable');
  context.clearRect(0, 0, 256, 256);
  context.fillStyle = '#eef1ff';
  context.font = '900 150px Arial, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(letter.toUpperCase(), 128, 139);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export function ExpoKeyboardScene({ action }: { action: Action }) {
  const host = useRef<HTMLDivElement>(null);
  const currentAction = useRef(action);
  currentAction.current = action;

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    let visible = true;
    let disposed = false;
    element.dataset.keycaps = 'loading';
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 10);
    // Centered view: a slight tilt retains visible travel without a diagonal layout.
    camera.position.set(0, 0.68, 0.24);
    camera.lookAt(0, 0.025, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    element.appendChild(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xeef1ff, 0x111735, 2));
    const keyLight = new THREE.DirectionalLight(0xeef1ff, 2.8);
    keyLight.position.set(-0.35, 0.85, 0.65);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xaab8ff, 0.7);
    rimLight.position.set(0.45, 0.25, -0.5);
    scene.add(rimLight);

    const keyGeometry = new RoundedBoxGeometry(0.086, 0.055, 0.086, 5, 0.012);
    const labelGeometry = new THREE.PlaneGeometry(0.044, 0.044);
    const keys = new Map<KeyboardHighlightTarget, {
      group: THREE.Group;
      material: THREE.MeshStandardMaterial;
      mesh: THREE.Mesh;
      label: THREE.Mesh;
    }>();
    const labelMaterials: THREE.MeshBasicMaterial[] = [];
    const labelTextures: THREE.CanvasTexture[] = [];
    const projectedLabelGeometries: THREE.BufferGeometry[] = [];

    for (const [target, [x, z]] of Object.entries(KEY_POSITIONS) as [KeyboardHighlightTarget, readonly [number, number]][]) {
      const group = new THREE.Group();
      group.position.set(x, RESTING_KEY_Y, z);

      const material = new THREE.MeshStandardMaterial({
        color: 0x222b55,
        emissive: 0xb3ff3b,
        emissiveIntensity: 0,
        roughness: 0.72,
        metalness: 0,
      });
      const mesh = new THREE.Mesh(keyGeometry, material);
      group.add(mesh);

      const texture = createKeyLabel(target);
      const labelMaterial = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
        toneMapped: false,
      });
      const label = new THREE.Mesh(labelGeometry, labelMaterial);
      label.position.y = 0.0285;
      label.rotation.x = -Math.PI / 2;
      group.add(label);

      scene.add(group);
      keys.set(target, { group, material, mesh, label });
      labelTextures.push(texture);
      labelMaterials.push(labelMaterial);
    }

    const idleColor = new THREE.Color(0x222b55);
    const activeColor = new THREE.Color(0xb3ff3b);
    let frame = 0;
    let previous = performance.now();
    const animate = (now: number) => {
      const dt = Math.min(0.05, (now - previous) / 1000);
      previous = now;
      if (!document.hidden && visible) {
        const active = new Set(keyboardHighlightTargets(currentAction.current));
        for (const [target, key] of keys) {
          const pressed = active.has(target);
          key.group.position.y = THREE.MathUtils.damp(
            key.group.position.y,
            pressed ? PRESSED_KEY_Y : RESTING_KEY_Y,
            28,
            dt,
          );
          key.material.color.lerp(pressed ? activeColor : idleColor, 1 - Math.exp(-20 * dt));
          key.material.emissiveIntensity = THREE.MathUtils.damp(
            key.material.emissiveIntensity,
            pressed ? 0.42 : 0,
            24,
            dt,
          );
        }
        renderer.render(scene, camera);
      }
      frame = requestAnimationFrame(animate);
    };

    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      const safeWidth = Math.max(1, width);
      const safeHeight = Math.max(1, height);
      renderer.setSize(safeWidth, safeHeight, false);
      camera.aspect = safeWidth / safeHeight;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(element);
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      visible = entry?.isIntersecting ?? false;
    });
    visibilityObserver.observe(element);
    resize();
    frame = requestAnimationFrame(animate);

    // Swap geometry on the existing moving groups; no per-frame asset creation.
    // Shared Cherry geometry stays in the bounded two-profile cache across mounts.
    void loadCherryKeycaps().then(caps => {
      if (disposed) return;
      const wLabel = createCherryKeycapLabel(caps.w);
      projectedLabelGeometries.push(wLabel);
      const rowLabel = createCherryKeycapLabel(caps.a);
      projectedLabelGeometries.push(rowLabel);
      for (const [target, key] of keys) {
        key.mesh.geometry = caps[target];
        key.label.geometry = target === 'w' ? wLabel : rowLabel;
        key.label.position.set(0, 0, 0);
        key.label.rotation.set(0, 0, 0);
      }
      element.dataset.keycaps = 'cherry';
      resize();
    }).catch(() => {
      if (!disposed) element.dataset.keycaps = 'fallback';
      // The existing rounded keys remain usable if the imported asset fails.
    });

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      resizeObserver.disconnect();
      visibilityObserver.disconnect();
      keyGeometry.dispose();
      labelGeometry.dispose();
      projectedLabelGeometries.forEach(geometry => geometry.dispose());
      keys.forEach(({ material }) => material.dispose());
      labelMaterials.forEach((material) => material.dispose());
      labelTextures.forEach((texture) => texture.dispose());
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return (
    <div
      ref={host}
      className="expo-keyboard-viewport"
      role="img"
      aria-label="Teclas W, A, S y D en 3D; la acción actual presiona su tecla"
    />
  );
}
