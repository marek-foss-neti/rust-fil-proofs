# ZigZag devnet deployment setup

**Status:** integration checklist for a Filecoin devnet. This is not a production deployment plan
and must not be used as evidence that ZigZag is mainnet-ready.

The goal of a devnet deployment is to run a full seal -> prove -> verify -> unseal path with
ZigZag while changing as little as possible in the surrounding Filecoin stack. The preferred
approach is a feature-gated shim: keep the existing devnet proof type and actor surface where
possible, but route the proof implementation through ZigZag inside `filecoin-ffi`, Curio, and
Lotus.

## Scope

The initial devnet target should be the smallest sector size used by the target devnet, normally
8 MiB for `porep-market-devnet`.

In scope:

- ZigZag PoRep replication, proof generation, verification, and unseal.
- Curio miner sealing enough to pass a seal/unseal roundtrip.
- Lotus on-chain seal verification with the same ZigZag verifier used by Curio.
- Dev-only Groth16 parameters generated locally or baked into the devnet image.

Out of scope:

- Mainnet security parameters.
- FIP registration.
- Long-running WindowPoSt/WinningPoSt unless the devnet explicitly exercises deadlines.
- Proving aggregation, SnapDeals, and sector update paths.

## Existing ZigZag support

The in-tree ZigZag API already provides:

- `zigzag_pre_commit` / `zigzag_pre_commit_phase1`
- `zigzag_pre_commit_phase1_with_replica_id`, for split seal pipelines where the caller already
  computed Filecoin's replica ID.
- `zigzag_pre_commit_phase2`
- `zigzag_prove`
- `zigzag_verify_seal`
- `zigzag_unseal` / `zigzag_unseal_range`
- `zigzag_comm_r_bound(comm_r, comm_r_star)`, which binds ZigZag's two replica commitments into
  one 32-byte value suitable for a devnet `SealedCID` shim.

The dev parameter path also exists:

```text
paramcache --only-zigzag -z <sector_size> --api-version <api_version>
```

`paramcache --only-zigzag` uses `OsRng`. These parameters are useful for local/dev testing only
and are not reproducible or ceremony-grade.

ZigZag does not use Stacked-style on-disk labels, `p_aux`, or `t_aux`. When this document refers
to ZigZag cache state, it means the persisted Merkle tree stores (`zigzag-tree-d` and
`zigzag-tree-r-*`) plus the `zigzag-aux.json` manifest written by the ZigZag pre-commit path.

## Required rust-fil-proofs work

### 1. Add a resumable prove path

Curio does not keep an in-memory Rust `ZigZagProverState` between pre-commit and commit tasks.
ZigZag currently persists disk-backed Merkle tree stores and `zigzag-aux.json`, but
`zigzag_prove` still requires the in-memory state returned by phase 1.

Add an API such as:

```text
zigzag_prove_from_cache(porep_config, cache_path, comm_d, comm_r, comm_r_star, prover_id,
sector_id, ticket, seed)
```

This API should:

- Load `zigzag-aux.json`.
- Re-open the persisted `zigzag-tree-d` and `zigzag-tree-r-*` stores.
- Reconstruct the `Tau` and private inputs required by `ZigZagCompound::prove`.
- Verify that supplied commitments match the persisted aux manifest.
- Return the same `SealCommitOutput` shape as `zigzag_prove`.

This is the key compatibility bridge for task-based sealers.

### 2. Add a split pre-commit path

Curio's split SDR/TreeD/TreeRC pipeline computes the Filecoin replica ID before `filecoin-ffi`
receives the phase-2 pre-commit call. That phase-2 call only carries Curio's placeholder
phase-1 JSON, the cache path, and the sealed sector path, so ZigZag cannot recompute the replica ID
there without extra state.

Expose an API such as:

```text
zigzag_pre_commit_phase1_with_replica_id(porep_config, cache_path, replica_id, comm_d, data)
```

This API should:

- Verify that `data` is exactly one fr32-padded sector.
- Verify that the Sha256 data-tree root equals the supplied `comm_d`.
- Use the supplied replica ID for ZigZag replication.
- Persist `zigzag-tree-d`, `zigzag-tree-r-*`, and `zigzag-aux.json`.
- Store the supplied replica ID in `zigzag-aux.json` so `zigzag_prove_from_cache` can later
  confirm it against the prover, sector, ticket, and `comm_d`.

This keeps the Curio integration in `filecoin-ffi` instead of requiring Curio Go task changes.

### 3. Stabilize dev parameter generation

The devnet must generate exactly the parameters used by the selected `PoRepConfig`, including:

- sector size
- `porep_id`
- API version
- partition count
- ZigZag tree shape

The simplest devnet path is to run `paramcache --only-zigzag` during image build or during the
devnet bootstrap step with `FIL_PROOFS_PARAMETER_CACHE` pointing at the runtime parameter cache.

For an iterative local devnet, it is also acceptable to generate missing ZigZag Groth16
parameters lazily on first use by setting:

```text
FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1
```

Keep this flag separate from `FIL_PROOFS_USE_ZIGZAG`. Enabling ZigZag selects the proof path;
enabling lazy generation decides whether the runtime may create missing local/dev parameters.

If deterministic dev images are needed, add a dev-only seeded parameter helper and label the
output as unsafe for production.

### 4. Keep ZigZag behind an explicit gate

Do not silently replace Stacked DRG in the public API. Add a feature or environment gate such as:

```text
FIL_PROOFS_USE_ZIGZAG=1
```

The gate should only affect the intended devnet sector size and proof type. All other proof paths
should continue to use the existing Stacked implementation.

## Required filecoin-ffi work

The current `filecoin-ffi` Rust crate depends on `filecoin-proofs-api`. ZigZag is exported from
the local `filecoin-proofs` crate, so the devnet shim needs one of the following:

- Add a direct local dependency on `filecoin-proofs` for ZigZag calls.
- Add a local `filecoin-proofs-api` bridge that re-exports the ZigZag APIs.
- Move the required ZigZag exports into the API crate used by `filecoin-ffi`.

Because `filecoin-ffi` builds with `--locked`, update `Cargo.lock` as part of the patch.

### Seal mapping

In ZigZag mode, map the existing FFI functions as follows:

| FFI function | ZigZag behavior |
|--------------|-----------------|
| `seal_pre_commit_phase1` | Read the staged sector, call `zigzag_pre_commit_phase1`, write the sealed sector and `zigzag-aux.json`, return ZigZag phase-1 JSON. |
| `generate_sdr` | In Curio split mode, create the cache directory and persist a small ZigZag replica-ID manifest instead of Stacked labels. |
| `seal_pre_commit_phase2` | For normal ZigZag envelopes, validate the phase-1 output against `zigzag-aux.json`; for Curio split mode, read the sector data prefix from `sc-02-data-tree-d.dat`, load the replica-ID manifest, call `zigzag_pre_commit_phase1_with_replica_id`, and return `comm_d` plus `zigzag_comm_r_bound(comm_r, comm_r_star)` as `comm_r`. |
| `seal_commit_phase1` | Call `zigzag_prove_from_cache`, write a devnet sidecar containing `comm_r` and `comm_r_star` into the shared proof-parameter cache, and return the raw Groth16 proof bytes. |
| `seal_commit_phase2` | Detect the raw 192-byte ZigZag proof and return it unchanged, so the commit proof stays within the existing actor proof-size limit. |
| `verify_seal` | For raw ZigZag proof bytes, load the shared sidecar, check the bound commitment against `SealedCID`, then call `zigzag_verify_seal`. |
| `unseal_range` | Read the full sealed sector, call `zigzag_unseal_range`, and write unpadded bytes to the output fd. |

The existing miner actor rejects individual seal proofs larger than one Groth16 proof
(`192` bytes for the devnet 8 MiB proof type). Do not pass a JSON proof envelope to chain commit.
For the local devnet shim, use:

- raw Groth16 proof bytes as the `SealCommitOutput`
- a sidecar JSON file under `${FIL_PROOFS_PARAMETER_CACHE}/zigzag-proof-sidecars`
- a sidecar record containing format marker, version, registered proof, sector id, `comm_d`,
  bound `comm_r`, ZigZag `comm_r`, ZigZag `comm_r_star`, prover id, ticket, seed, proof length,
  and a proof prefix

This avoids changing `SealVerifyInfo` during local devnet testing while still giving the Lotus
verifier enough public input to validate ZigZag. It is a devnet-only shortcut and requires Curio
and Lotus to share the same proof-parameter cache volume.

## Required Curio work

The minimal devnet path should avoid Curio Go changes. Curio's existing task pipeline can remain
intact if `filecoin-ffi` handles the split seal calls described above:

Required changes:

- Set `FIL_PROOFS_USE_ZIGZAG=1` on the Curio runtime service.
- Let Curio continue to run `SDR`, `TreeD`, and `TreeRC` tasks.
- Ensure `filecoin-ffi` treats `GenerateSDR` as a ZigZag marker step and persists the replica ID.
- Ensure `filecoin-ffi` treats Curio's placeholder TreeRC phase-1 JSON as the signal to run
  ZigZag pre-commit from `sc-02-data-tree-d.dat`.
- Route C1 self-checks through the ZigZag-aware `SealCommitPhase1`.
- Confirm the final unseal assertion uses the ZigZag-aware FFI unseal path, not a retained
  unsealed copy.

The last point is required for a real sealed-only unseal test. Keeping an unsealed copy around is
useful for debugging, but it does not prove ZigZag extraction works inside the devnet.

## Required Lotus work

A true devnet seal test must patch Lotus too. The current devnet image copies Lotus binaries from
a prebuilt `LOTUS_TEST_IMAGE`; those binaries use the stock verifier and will reject a ZigZag
proof.

Build Lotus from the managed Lotus source with the same patched `filecoin-ffi`, then copy those
binaries into the devnet image. At minimum this must cover:

- `lotus`
- `lotus-seed`
- `lotus-shed`
- `lotus-miner`

For the minimal devnet shim, the builtin actor surface can remain unchanged if `filecoin-ffi`
recognizes raw ZigZag proof bytes plus the shared devnet sidecar under the existing 8 MiB proof
type. Set `FIL_PROOFS_USE_ZIGZAG=1` on Lotus so the verifier takes this path. Do not set
`FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1` on Lotus; missing parameter generation should stay
on the Curio smoke-test side or be done ahead of time. This is explicitly a devnet shortcut and
must not be carried into production.

Keep the Lotus genesis pre-seal path outside the ZigZag gate. `lotus-seed` is used to create the
bootstrap miner and genesis template before the runtime daemon starts; it should run with
`FIL_PROOFS_USE_ZIGZAG` unset. The Lotus daemon can then inherit `FIL_PROOFS_USE_ZIGZAG=1` for
runtime seal verification once genesis exists.

## Required porep-market-devnet work

Keep the devnet repository changes as overlay/build wiring:

- Add a build context for the local `rust-fil-proofs` checkout or a pinned managed copy.
- Add an overlay patch for `filecoin-ffi` and apply it while building both Curio and Lotus.
- Build Curio and Lotus from verified source plus overlays.
- Pass an explicit ZigZag gate into the Curio runtime container and the Lotus runtime container.
  Curio uses it for sealing and unseal; Lotus uses it for seal verification.
- Generate or install ZigZag dev parameters into `FIL_PROOFS_PARAMETER_CACHE`.
- For local-only smoke tests, set `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS=1` on the Curio
  runtime container so the first ZigZag proof can create missing dev parameters.
- Keep `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS` out of Lotus. Lotus should read generated or
  pre-installed params from the shared proof-parameter cache.
- Record the ZigZag source commit and patch digests in the devnet build manifest.

Avoid editing `porep-market-devnet/.cache/sources/*` directly; those checkouts are expected to be
clean and detached.

## Validation sequence

Run the validation in increasing order:

1. `cargo test -p storage-proofs-porep zigzag --no-default-features`
2. `cargo test -p filecoin-proofs --test zigzag_api --no-default-features -- --nocapture`
3. `cargo run -p fil-proofs-param --bin paramcache --no-default-features -- --only-zigzag -z 8388608 --api-version <api_version>`
4. Rebuild the devnet images with ZigZag overlays enabled.
5. Reset and deploy the devnet.
6. Run the seal/unseal roundtrip.
7. Confirm that:
   - the sector activates on chain,
   - Lotus verifies the ZigZag proof,
   - Curio can unseal from the sealed sector,
   - recovered bytes match the source bytes,
   - no Stacked-only decode path was used for the final unseal assertion.

## Devnet readiness checklist

- [ ] `zigzag_prove_from_cache` or equivalent resumable prove API exists.
- [ ] `zigzag_pre_commit_phase1_with_replica_id` or equivalent split pre-commit API exists.
- [ ] ZigZag dev params are generated for the exact devnet `PoRepConfig`.
- [ ] Curio sealing path uses ZigZag in gated mode.
- [ ] Curio unseal path uses ZigZag extraction in gated mode.
- [ ] Lotus binaries are rebuilt with the same ZigZag-aware verifier.
- [ ] Raw seal proof remains 192 bytes and the devnet sidecar carries `comm_r_star`.
- [ ] `SealedCID` equals `zigzag_comm_r_bound(comm_r, comm_r_star)`.
- [ ] The existing Stacked path still passes when the ZigZag gate is disabled.
- [ ] The devnet seal/unseal test passes from a clean build.

## Known limitations

This setup is enough to measure devnet integration cost and basic operational fit. It does not
answer mainnet security, ceremony, PoSt, aggregation, or long-running sector proving questions.
Those are production requirements and are tracked separately in
[zigzag-production-setup.md](zigzag-production-setup.md),
[zigzag-security-params.md](zigzag-security-params.md),
[zigzag-post-strategy.md](zigzag-post-strategy.md), and
[zigzag-ceremony-and-adoption.md](zigzag-ceremony-and-adoption.md).
