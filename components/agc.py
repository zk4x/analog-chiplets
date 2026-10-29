# AGC scaler (feedback auto-ranging gain stage)
#
# Serves as dynamic current range normalizer, the other Scaler is static.
#
# param           | circuit element                     | error class
# ----------------------------------------------------------------------------------
# target          | reference voltage (servo setpoint)  | -
# loop_gain       | integrator transconductance         | settling lag (finite loop gain)
# detector_offset | rectifier/peak-detector diode drop  | systematic (steady-state gain bias)
# detector_noise  | detector kT/C sampling noise        | zero-mean, per update
# gain_offset     | mirror ratio error in the V->gain   | systematic (residual gain error)
# g               | integrator state (held gain)        | state
#

import torch


class AGCScaler:
    def __init__(
        self,
        target: float = 1.0,
        loop_gain: float = 1.0,
        detector_offset: float = 0.0,
        detector_noise_std: float = 0.0,
        gain_offset: float = 0.0,
        g0: float = 1.0,
    ):
        self.target = target
        self.loop_gain = loop_gain
        self.detector_offset = detector_offset
        self.detector_noise_std = detector_noise_std
        self.gain_offset = gain_offset
        self.g = g0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # servo on the peak of the (previous) output; the corrected gain is
        # applied immediately (sample-then-adjust, one-call settling)
        y = x * self.g * (1.0 + self.gain_offset)
        s = y.abs().max() + self.detector_offset
        if self.detector_noise_std > 0:
            s = s + torch.randn(1).item() * self.detector_noise_std
        if s > 0:
            # integrator: exponential approach to the servo fixed point
            # (output peak == target); stable for loop_gain in (0, 2)
            self.g = self.g * (self.target / s) ** self.loop_gain
        return x * self.g * (1.0 + self.gain_offset)
