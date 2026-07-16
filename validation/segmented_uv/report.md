# Segmented UV Validation Report

## Status

The wavelength-resolved boundary-calibration runner completed against the
saved high-photon full-domain reference. The calibrated run produced four
segments with split radii of 33982, 10554, and 3587 au. The previous runner
split at 33982, 10554, and 2996 au.

## Boundary Joins

The one-pass calibration reduced the maximum wavelength-resolved parent-child
comparison-shell mismatch from 8.98%, 6.18%, and 4.64% to 1.64%, 1.39%, and
1.16%. The corresponding shell-mean `chi_broad` offsets fell from 2.15%, 1.25%,
and 0.56% to 0.072%, 0.058%, and 0.042%.

The calibrated correction factors span 0.975-1.093, 0.981-1.074, and
0.978-1.066 across the three handoffs. Calibration provenance is retained in
each child segment and its disposable RADMC-3D directory is removed after a
successful handoff.

## Full-Domain Comparison

For `chi_broad`, the volume-weighted RMS residual improved from 0.673% to
0.442%, and the P99 absolute residual improved from 2.89% to 1.35%. The radial
profile no longer shows the previous cumulative inward deficit: at 814 au the
residual changed from -6.34% to -2.63%, and the fraction of radial shells within
1% increased from 39.8% to 52.3%.

The `G_CO_pdes` RMS residual improved from 0.606% to 0.437%, its P99 residual
from 2.47% to 1.34%, and the `G_CO_diss` P99 residual from 6.85% to 1.68%.
Hard-UV inner-domain maxima remain unsuitable as acceptance metrics because
the very large full-domain reference is geometrically undersampled there.

The calibrated workflow accounts for 11.15 billion photon packages by
execution role, compared with 11.5 billion for the previous algorithm. It adds
one half-scout calibration per child but avoids the previous terminal paired
scout and duplicate final run. A clean laptop run is expected to take about
4-4.5 hours based on the measured RADMC-3D stages.

## Decision

The calibration solves the accumulating scalar boundary-normalization error
without adding iterative transport or increasing the total package count. It
does not establish 1% radial accuracy: a few-percent inner residual remains
even though shell-mean joins agree below 0.1%. This is consistent with replacing
the parent radiation by an isotropic angle-averaged spectrum; one scalar
spectral correction cannot recover its inward angular distribution.

The implementation is therefore a clear improvement over the previous runner,
but production use must accept a few-percent radial approximation. Requiring
strict 1% agreement would need a directional boundary representation and a
separate compact hard-UV validation.
