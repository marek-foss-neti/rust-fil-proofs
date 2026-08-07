# ZigZag devnet integration notes

**Status:** integration record for the local `restore-zigzag` to `porep-market-devnet`
experiment. This documents what was actually needed to run a ZigZag-backed
seal -> prove -> verify -> unseal path in the local devnet. It is not a production deployment
plan.

The integration goal was to change as little as possible in both repositories:

- keep ZigZag implementation work inside `rust-fil-proofs` where possible,
- keep Filecoin stack changes as overlay patches,
- avoid editing managed source checkouts under `porep-market-devnet/.cache/sources`,
- keep the existing devnet registered seal proof and actor surface,
- make the ZigZag path explicit and easy to disable.

## Final validation result

The current local stack passed a full seal/unseal roundtrip with ZigZag enabled.

Validated path:

- `rust-fil-proofs` exposes the required ZigZag devnet APIs.
- `porep-market-devnet` builds Curio and Lotus with ZigZag overlays.
- Lotus runtime verification uses the ZigZag verifier through the patched FVM/filecoin-ffi path.
- Curio unseals from the sealed replica through ZigZag extraction.
- The E2E report records which proof backend was used.

Observed `test-seal-unseal` result:

- run ID: `2026-08-06T08-14-28-779Z-seal-unseal-roundtrip`
- registered proof: `StackedDrg8MiBV1_1` (`6`)
- sector size: `8 MiB`
- reported proof backend: `ZigZag`
- unseal path: `ZigZag filecoinffi.Unseal`; `SDRKeyRegen` is skipped as a
  scheduler-compatible no-op
- source SHA-256 and recovered SHA-256:
  `e4fdb830586f07e5e29b5d5e91b9eab9fb4246f96e53bffef375bca4c8740022`

## Changes in rust-fil-proofs

The devnet integration needed a restart-safe ZigZag sealing API surface. Curio is a task-based
sealer and does not keep a Rust `ZigZagProverState` value in memory between pre-commit and
commit.

Required changes:

- `filecoin-proofs/src/api/zigzag.rs`
  - Added or stabilized `zigzag_prove_from_cache`.
  - Added `zigzag_pre_commit_phase1_with_replica_id` for split seal pipelines where the caller
    already computed Filecoin's replica ID.
  - Reworked pre-commit helpers so sector-size validation, Sha256 CommD calculation, and
    replica-id based replication can be reused by the normal and split paths.
  - Converted arbitrary Filecoin challenge randomness into an Fr-safe challenge seed before
    calling the ZigZag compound proof.
  - Kept ZigZag cache state limited to `zigzag-tree-d`, `zigzag-tree-r-*`, and
    `zigzag-aux.json`.
  - Kept `zigzag_prove_from_cache` independent of Stacked-style on-disk labels. It reopens
    ZigZag Merkle tree stores and validates `zigzag-aux.json`; it does not need `labels`,
    `p_aux`, or `t_aux`.

- `filecoin-proofs/src/caches.rs`
  - Added `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS` for local/dev lazy parameter generation.
  - Kept this flag separate from `FIL_PROOFS_USE_ZIGZAG`.
  - This is intentionally dev-only. Production nodes must use published parameter manifests and
    fail closed when parameters are missing.

- `filecoin-proofs/tests/zigzag_api.rs`
  - Covered cached prove, split pre-commit, verification with a challenge seed, and unseal.
  - Added an ignored diagnostic test that can verify a Curio-produced cache and proof sidecar.

## Changes in filecoin-ffi

`filecoin-ffi` was the main compatibility shim. The patch lets existing Curio and Lotus calls keep
their normal function names while changing the proof implementation when
`FIL_PROOFS_USE_ZIGZAG=1`.

Required overlay:

- `porep-market-devnet/patches/filecoin-ffi/0001-zigzag-devnet-ffi.patch`
- `porep-market-devnet/patches/filecoin-ffi/0002-zigzag-devnet-fvm4-path.patch`

Required behavior:

- Add direct local Rust dependencies on the local `rust-fil-proofs` checkout.
- Gate ZigZag behavior with `FIL_PROOFS_USE_ZIGZAG`.
- Limit the devnet ZigZag shim to `2 KiB` and `8 MiB` sectors.
- Map Curio's split sealing pipeline onto ZigZag:
  - `generate_sdr` persists a small ZigZag replica-id manifest instead of Stacked SDR labels.
  - `seal_pre_commit_phase2` recognizes Curio's placeholder phase-1 data, reads the TreeD sector
    data, calls `zigzag_pre_commit_phase1_with_replica_id`, and returns `comm_d` plus a bound
    sealed commitment.
  - `seal_commit_phase1` calls `zigzag_prove_from_cache`.
  - `seal_commit_phase2` passes through the raw Groth16 proof.
  - `verify_seal` calls the ZigZag verifier when the raw proof and sidecar indicate the ZigZag
    path.
  - `unseal_range` calls `zigzag_unseal_range`.

The commit proof had to stay as raw 192-byte Groth16 proof bytes because the existing actor proof
type does not accept a larger ZigZag JSON envelope. The extra ZigZag public inputs are therefore
stored in a devnet sidecar under:

```text
${FIL_PROOFS_PARAMETER_CACHE}/zigzag-proof-sidecars
```

The sidecar records the registered proof, sector id, `comm_d`, bound `comm_r`, ZigZag `comm_r`,
ZigZag `comm_r_star`, prover id, ticket, seed, proof length, and proof prefix. This is a local
devnet ABI shortcut, not a production proof format.

## Changes in FVM and Lotus

Patching Curio alone was not enough. A real devnet seal test submits the proof to Lotus, and Lotus
must verify the same ZigZag public inputs that Curio proved.

Required overlays:

- `porep-market-devnet/patches/fvm/0001-zigzag-devnet-verifier.patch`
- `porep-market-devnet/patches/curio/0002-lotus-entrypoint-zigzag-runtime-only.patch`

Required behavior:

- Patch the FVM verifier used by Lotus so `FIL_PROOFS_USE_ZIGZAG=1` routes raw 192-byte ZigZag
  proofs through `zigzag_verify_seal`.
- Read the devnet sidecar from the shared proof-parameter cache.
- Check that sidecar fields match `SealVerifyInfo`.
- Check that `zigzag_comm_r_bound(comm_r, comm_r_star)` equals the sealed commitment supplied to
  Lotus.
- Build Lotus binaries from the patched source path instead of copying stock binaries from the
  base devnet image.
- Keep `lotus-seed` genesis and preseal commands outside the ZigZag gate by unsetting
  `FIL_PROOFS_USE_ZIGZAG` and `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS` for those commands.

The genesis exception matters because the bootstrap miner is created before the runtime devnet
path starts. Letting genesis preseal inherit the ZigZag runtime gate mixes two incompatible
assumptions in one bootstrap path.

## Changes in Curio

The first target was to avoid Curio Go changes. That was possible for most sealing work because
the existing SDR/TreeD/TreeRC task shape can be interpreted by the patched `filecoin-ffi`.

Sealed-only unseal required a Curio overlay:

- `porep-market-devnet/patches/curio/0003-zigzag-devnet-unseal.patch`

Required behavior:

- Keep Curio's task scheduling shape intact.
- In ZigZag mode, let the unseal SDR task mark `SDRKeyRegen` complete without creating a Stacked
  SDR key.
- Pass ticket and CommD into the unseal decode call.
- In ZigZag mode, make decode call `filecoinffi.Unseal`.
- Pad the unpadded ZigZag output back into Curio's expected `FTUnsealed` file shape.

This patch is needed because ZigZag does not have SDR labels to regenerate, and Curio's stock
`DecodeSDR` path expects exactly that Stacked-specific key material.

## Changes in porep-market-devnet

The devnet repository mostly acts as overlay and verification glue.

Required build wiring:

- Add the local `rust-fil-proofs` checkout as a Docker build context.
- Copy only the required Rust proof crates into the build image.
- Apply the `filecoin-ffi` ZigZag patch while building both Curio and Lotus.
- Copy the FVM crate used by the selected Lotus/filecoin-ffi version into a local patched path and
  apply the ZigZag verifier patch there.
- Build Curio from source with the patched FFI.
- Build Lotus from source with the patched FFI and FVM.
- Build derived service images from the resulting all-in-one base image.
- Label built images with:
  - `io.porep-market.zigzag.filecoin-ffi.patch.sha256`
  - `io.porep-market.zigzag.rust-fil-proofs.api.sha256`

Required runtime wiring:

- Set `FIL_PROOFS_USE_ZIGZAG=1` on `lotus`.
- Set `FIL_PROOFS_USE_ZIGZAG=1` on `curio`.
- Set `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1` on `curio` only.
- Keep `FIL_PROOFS_USE_ZIGZAG` unset for `lotus-miner`.
- Share the proof-parameter cache between Curio, Lotus, and services that need to inspect proof
  artifacts.

Required safety checks:

- Verify the local `rust-fil-proofs` path exists and is not a symlink.
- Verify that `filecoin-proofs/src/api/zigzag.rs` exposes
  `zigzag_prove_from_cache` and `zigzag_pre_commit_phase1_with_replica_id`.
- Include the FFI, Curio, FVM, Docker, and ZigZag API inputs in build hashes.
- Reject stale image manifests when patch or API hashes differ.
- Keep managed source checkouts clean and detached.

Required E2E/reporting changes:

- Detect whether the active run uses SDR or ZigZag from Curio runtime environment and registered
  proof metadata.
- Record `PROOF_BACKEND`, `PROOF_BACKEND_LABEL`, `PROOF_BACKEND_REASON`, and `UNSEAL_PATH`.
- Render a `Proof backend` row in `summary.md`.
- Rename the unseal wait step so ZigZag runs report `wait for ZigZag UnsealDecode with
  SDRKeyRegen skipped`.

## Decisions made during integration

| Decision | Reason | Consequence |
|----------|--------|-------------|
| Use an explicit environment gate. | The devnet should be able to switch back to SDR without broad code changes. | `FIL_PROOFS_USE_ZIGZAG=1` selects ZigZag; unset uses the stock path. |
| Reuse the existing 8 MiB registered proof type. | This minimized actor and chain changes for the local devnet. | The setup is not a production ABI and must not be treated as protocol registration. |
| Bind `comm_r` and `comm_r_star` into one sealed commitment. | Filecoin's existing seal path carries one sealed CID. | `SealedCID` uses `zigzag_comm_r_bound(comm_r, comm_r_star)`. |
| Keep the chain proof as raw 192-byte Groth16 bytes. | The existing actor proof-size limit rejects a larger proof envelope. | A shared sidecar is required for devnet verification. |
| Put sidecars under the proof-parameter cache. | Curio and Lotus already share that cache volume in the devnet. | The verifier can recover ZigZag public inputs without changing `SealVerifyInfo`. |
| Add `zigzag_prove_from_cache`. | Curio tasks do not preserve Rust in-memory prover state across phases. | Proving can resume from persisted ZigZag trees and `zigzag-aux.json`. |
| Do not require on-disk labels. | ZigZag persists tree stores, not Stacked SDR label layers. | The cached prove path validates aux and reopens trees directly. |
| Generate missing ZigZag params only in Curio. | Local smoke tests need quick parameter bootstrapping, while Lotus should verify against existing files. | `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1` is runtime-local and dev-only. |
| Patch Curio only for sealed-only unseal. | Sealing could be adapted in FFI, but unseal decode was Stacked-specific in Go. | The Curio patch stays narrow and scheduler-compatible. |
| Patch Lotus/FVM instead of bypassing chain verification. | The goal was a true seal/verify flow, not only local proof generation. | The devnet catches verifier mismatches before unseal. |
| Hash patch and API surfaces. | The devnet should not silently reuse stale images after proof changes. | Build and deploy fail when the ZigZag overlay identity changes. |

## What remains devnet-only

The following pieces are useful for local integration but should not be carried into production:

- reusing `StackedDrg8MiBV1_1` for ZigZag sectors,
- the raw-proof-plus-sidecar ABI,
- lazy local Groth16 parameter generation,
- environment-only proof dispatch,
- patching FVM internals instead of adding a registered proof family,
- skipping Stacked `SDRKeyRegen` as a compatibility no-op,
- omitting ZigZag PoSt and aggregation support.

Production work is tracked separately in
[zigzag-production-setup.md](zigzag-production-setup.md),
[zigzag-security-params.md](zigzag-security-params.md),
[zigzag-post-strategy.md](zigzag-post-strategy.md), and
[zigzag-ceremony-and-adoption.md](zigzag-ceremony-and-adoption.md).

## Curio Docker Devnet comparison

The official Curio Docker Devnet documentation describes a devnet built from the root of the Curio
repository with:

```text
make clean docker/devnet
make devnet/up
```

It also supports selecting a Lotus tag with `lotus_version=...`, and building Lotus manually when
the requested branch or tag is not available as a prebuilt image. The documented stack starts
`lotus`, `lotus-miner`, `yugabyte`, `curio`, and `piece-server`, stores temporary data under
`./docker/data`, and downloads Filecoin proof parameters during initial setup.

Source: <https://docs.curiostorage.org/docker-devnet>

### Would the same steps be needed?

For a real on-chain ZigZag seal/unseal test, the proof-stack changes would be almost the same:

| Area | Same in Curio Docker Devnet? | Notes |
|------|------------------------------|-------|
| `rust-fil-proofs` API changes | Yes | Curio still needs cached proving, split pre-commit, ZigZag aux validation, unseal, and dev params. |
| `filecoin-ffi` ZigZag shim | Yes | Curio and Lotus still enter proofs through `filecoin-ffi`. |
| FVM/Lotus verifier patch | Yes, if ProveCommit is submitted on chain | Lotus must verify the ZigZag proof, not only accept a local Curio result. |
| Curio sealed-only unseal patch | Yes, if unseal is tested from sealed data | Stock Curio unseal still expects SDR labels for `DecodeSDR`. |
| Shared proof-parameter cache | Yes | The raw proof sidecar and ZigZag params must be visible to both Curio and Lotus. |
| Runtime ZigZag env gates | Yes | `FIL_PROOFS_USE_ZIGZAG=1` is still the local dispatch mechanism. |
| Genesis/preseal gate isolation | Probably | If the same Lotus container performs genesis preseal and later runs with the ZigZag env, preseal commands must run with ZigZag disabled. |
| `porep-market-devnet` build manifest and contract wiring | No | These are specific to the PoRep Market devnet harness. |
| MK20 deal flow and custom E2E reporting | No, unless the same PoRep Market test surface is required | Curio Docker Devnet has its own `piece-server` sample deal flow. |

### Would it be fewer or more changes?

It depends on what "devnet" means for the experiment.

For a Curio-only local network, it should be fewer repository-level changes than
`porep-market-devnet`. The Curio Docker Devnet already owns a small stack with Lotus, Curio,
Yugabyte, and a piece server, so the PoRep Market-specific contract bootstrap, indexer wiring,
typed lifecycle scripts, and E2E summary changes would not be required.

The core proof changes would not be fewer. A chain-verified ZigZag sector still requires the same
`rust-fil-proofs`, `filecoin-ffi`, Curio unseal, and Lotus/FVM verifier work. The difference is
where those changes live.

It may even require more image-build plumbing if starting from the stock Curio Docker Devnet
unchanged. The documented flow can select Lotus versions and optionally build Lotus, but it is not
described as a general overlay system for a locally patched `rust-fil-proofs`, patched
`filecoin-ffi`, and patched FVM crate. To keep the changes minimal there, the recommended path
would be a small Curio Docker Devnet overlay:

1. Reuse the existing Curio Docker Devnet compose/services.
2. Override the Curio and Lotus images with locally built ZigZag-aware images.
3. Add the local `rust-fil-proofs` checkout as a build context.
4. Apply the same `filecoin-ffi` and FVM patches during both builds.
5. Apply the narrow Curio unseal patch.
6. Set `FIL_PROOFS_USE_ZIGZAG=1` on Curio and Lotus runtime containers.
7. Set `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1` only where dev parameter generation is
   allowed.
8. Ensure the shared proof-parameter cache is mounted into both Curio and Lotus.
9. Keep genesis preseal outside the ZigZag runtime gate.
10. Add a small seal/unseal smoke script or adapt the documented `piece-server` deal flow to
    assert the backend and recovered bytes.

So the practical answer is:

- fewer changes outside the proof stack,
- about the same changes inside the proof stack,
- possibly more Docker build customization unless Curio Docker Devnet grows an explicit overlay
  hook for patched `rust-fil-proofs`, `filecoin-ffi`, and FVM.

