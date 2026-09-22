# Range estimation derivations

## 1. Calibrated pinhole projection

For a drone with measured span `W`, focal length `F` in pixels, and detected
image width `p`, the range estimate is:

`z = W F / p`

`W` and `F` must correspond to the actual drone and active camera resolution.
Use `--drone-width` for an exact span, then provide a known distance with
`--calibration-distance` and press `K` while the detected drone is stationary.
This estimates `F = p z / W`.

## 2. Measurement uncertainty

With pixel-edge noise `sigma_p`, the width-measurement Fisher information is:

`I_w(z) = W^2 F^2 / (sigma_p^2 z^4)`

and the pixel-only lower-bound variance is `1 / I_w`. Pixel noise is not the
whole error, so the implementation also adds pose and calibration terms:

`R = 1 / I_total + (0.045 z)^2 + (0.10 z)^2`

The last term intentionally prevents the HUD from reporting a misleadingly
small CRLB when the focal length or physical span is not perfectly calibrated.

## 3. How the weights change

The code computes width, height, and diagonal range cues. Each cue receives a
normalized information weight:

`alpha_i = I_i / (I_w + I_h + I_d)`

and the fused measurement is:

`z_measurement = alpha_w z_w + alpha_h z_h + alpha_d z_d`

The height cue is deliberately down-weighted by `0.40` because camera pitch,
drone tilt, and a YOLO box height are less stable than horizontal span.

Across frames, the Kalman update changes the measurement weight again:

`K = P_prior / (P_prior + R)`

`z_filtered = z_prior + K (z_measurement - z_prior)`

As range increases or calibration uncertainty grows, `R` rises, `K` falls, and
the filter trusts the predicted track more than a noisy one-frame box. After
calibration, `R` is lower in practice because the systematic scale bias is
removed; the displayed confidence interval remains conservative.
