MaleCNS Crossy V2 — final perspective neural camera gate
==========================================================

This REPLACES the rejected orthographic/canonical raster sensor.

The new neural sensor is:
  - 160x120 RGB
  - 108 degree horizontal FOV
  - camera origin at the fly position
  - 25 degree downward pitch
  - true perspective projection
  - simple 3-D lane/hazard/scenery geometry with a z-buffer
  - generated only from authoritative GameState
  - independent from the browser/UI/Three.js presentation camera

It contains no planner values, action hints, ObservationV4 values, or safety labels.

Install from repo root:
  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-perspective-camera.zip" -DestinationPath . -Force

Then:
  cd python
  pytest -q tests/test_v2_crossy_camera.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.camera_gate --device cuda

Representative real-world samples are written to:
  reports\malecns-crossy-v2-perspective-samples\
    road.bmp
    rail.bmp
    river.bmp

Do not start the information probe until these images are visually checked.
