# Environment v10 / ObservationV4

This is the perception/validation foundation for Connectome Controller V2.

ObservationV4 keeps the original 492 V3 values and appends a 25-value long-range traffic radar: current row plus four rows ahead, each with lane code, signed traffic speed, and time-to-contact for left/current/right columns. The radar uses the full periodic traffic circuit, so a vehicle can be sensed before it enters the playable +/-5 area. Total width: 517.

A training-only action teacher evaluates all five actions against the real short-horizon traffic physics. The browser never receives those labels.

The browser Safety Reflex is reduced to recovery from physically blocked scenery/bounds moves only. It no longer performs traffic vetoes or forced-forward navigation.

No long training run should be launched at this stage.
