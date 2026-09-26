MaleCNS Crossy V2 — authoritative neural camera
================================================

Design choice
-------------
The neural input is NOT a screenshot of the page and NOT the human Three.js camera.

Instead, the controller uses a fixed 160x120 egocentric RGB sensor renderer built directly
from authoritative GameState. This is deliberate:

- UI/CSS/layout changes cannot alter neural input.
- training and final inference use the exact same renderer;
- there is no screenshot/readback/browser timing dependency;
- the renderer contains no planner values, action labels or ObservationV4 values;
- moving hazards remain visual motion and therefore still drive T4/T5 through the retina.

The human UI can later display this frame as "what the fly sees", but it never generates it.

Install
-------
From repository root:

  Expand-Archive "$env:USERPROFILE\Downloads\malecns-crossy-v2-crossy-camera.zip" -DestinationPath . -Force

Then:

  cd python
  pytest -q tests/test_v2_crossy_camera.py --basetemp .pytest-tmp
  python -m fly_crossy.v2.camera_gate --device cuda

The gate also writes a few BMP files under:
  reports/malecns-crossy-v2-camera-samples/

Open them in Windows and inspect that road / rail / river / hazards are visually legible.

No training happens in this stage.
