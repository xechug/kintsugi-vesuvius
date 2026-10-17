# Data Provenance

## Verified Kaggle 2023 Baseline

The verified baseline uses all three labeled Vesuvius Challenge 2023 competition fragments. Every volume contains layers 24-39 and has Zarr chunks `16 x 128 x 128`.

- Source: Kaggle competition `vesuvius-challenge-ink-detection`.
- Volume subset: layers `24.tif` through `39.tif`, downloaded individually on 2026-07-15.
- Fragment 1: shape `16 x 8181 x 6330`, volume metadata SHA-256 `a38d8c08ef90e41a9cd1fa5cd19d6089affbfbc5edf846a66392b02d07b527e6`.
- Fragment 2: shape `16 x 14830 x 9506`, volume metadata SHA-256 `f116547f2aec68c2ae4f20dc7fb0093e08d8b7eaf5f4adb72cdfc1951783f7c1`.
- Fragment 3: shape `16 x 7606 x 5249`, volume metadata SHA-256 `6b69c642bd2fbd0a942d622c9ba1d16b8b018d84ec1b0bda0b2d191f5ab8f615`.

Original PNG hashes are retained as source provenance:

| Fragment | Mask PNG SHA-256 | Ink PNG SHA-256 |
|---|---|---|
| 1 | `1fdc13f7dc149dc4e00272af597ba236b650e54bb2dca38865f211207cee069f` | `0b08dbf059c5d9feec7caf11e95390920589517b8816660ebec6474334a0693c` |
| 2 | `babde5508f7c314a8a34cfbe372ca163873ca3d3d435c1414943f25171736bc6` | `a82ba7abe10c3395a5240ecf924d3712663cf9e72edd7ae499b7bbc66dbe97be` |
| 3 | `f77c33cef758ffb17f63af0624aac5b04e93020a71e5103fa8370347427a87d8` | `d914c97085d1652451743dd4f468797eca6491b81ee7447d6a6f82a4faf44361` |

Operational rasters are pixel-identical tiled TIFF derivatives, which avoid full PNG decoding in every worker:

| Fragment | Mask TIFF SHA-256 | Ink TIFF SHA-256 |
|---|---|---|
| 1 | `51d04f4e0c9bd0e4c8aa17fa26166e8d82ea3728b912bdcdf632ce7682bab627` | `26d3913cf48357864964231b27e408c04bc87d40e6e650445b8507909e959ed1` |
| 2 | `b121a2c27863189439c7185fc7e5285b858fd3dc545c5c57ce80b4dd0138fa3a` | `40633030e58c1fa6aedb2d7cf6dc3dcc201a96f95fe0758ffe2efc38c1c984ce` |
| 3 | `d940dc089a5bc9029371dbf4357597f3719fc03c043ac895b4dfa5f9e1010ef1` | `a0718f560f4af3233be79346f7ff65dd8f49fe775b3f9524603a9f6d36d815c9` |

Each fragment has `data_audit.json` and a frozen 2048-pixel block split with 256-pixel guards. Region counts are fragment 1: `8/2/2`, fragment 2: `25/5/5`, fragment 3: `8/2/2` for train/validation/test. The active manifest trains on fragments 1+2 and validates/tests on disjoint regions of fragment 3.

These data support a legitimate 3D-to-2D surface-volume baseline. They do not contain voxel-level 3D ink labels and cannot validate direct 3D segmentation.

## Paris 4 Public Data

Public S3 access always runs unsigned with all `AWS_*` environment variables removed:

```bash
env -u AWS_PROFILE -u AWS_DEFAULT_PROFILE -u AWS_SESSION_TOKEN \
  -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY \
  aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/
```

Downloaded metadata object:

- S3 key: `metadata.json`.
- Local path: `data/public/vesuvius-open-data-metadata.json`.
- The object is gzip-compressed despite its `.json` suffix.
- SHA-256: `7dc56583c7a8eaf38d0a582237eebf091f3c0900e79f0c006a768db496994946`.

The Paris 4 segment inventory at `data/public/inventories/paris4-segment-20230702185753-ink-detection.json` contains a model-produced ink TIFF and a downsampled JPEG. The filename identifies `new_canon_autoresearch_recipe`, tile 256, stride 128. These are teacher predictions, not independent ground truth, and are excluded from validation and test.

The 2026 metadata identifies the full masked Paris 4 CT as `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/`, shape `75784 x 32693 x 32693`, uint8 at 2.4 micrometers. It is not downloaded locally.

Only metadata was downloaded from the public 3D ink prediction at `PHercParis4/representations/predictions/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/`:

- Local directory: `data/public/paris4_ink3d_v3_78k_metadata/`.
- Level 0: shape `75784 x 32693 x 32693`, chunks `256 x 256 x 256`, uint8, Blosc Zstd.
- Root `.zattrs` SHA-256: `97e968dbbf11cc856dc57648aaaa9e6cade5e6575472dcfdeac7cfa50f546dd9`.
- Level-0 `.zarray` SHA-256: `e1c66e1de53a4afd82da5f06a4bc6b40b90e602c9acda35b5e8ac4fa9fa01b53`.

This Zarr is a public model prediction. It is eligible only as a provenance-tracked teacher or research target, not as independent truth.

## Quarantined Local Files

`sample_data/mask.tif` and `sample_data/inklabels.tif` both have SHA-256 `51cdda75e7d4c3e5ab81da1488f69c5bda246ff53b11d709ef534a12b81637f9`. They are the same grayscale optimized-inference output and must not be used as separate validity and ink labels.

`sample_data/volume.zarr` remains useful for out-of-core I/O and inference engineering. Its metadata identifies a Paris 4 2.4 micrometer surface volume, not PHerc0332 and not a complete raw-roll CT dataset.
