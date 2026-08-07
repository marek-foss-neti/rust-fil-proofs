# ZigZag production deployment setup

**Status:** production readiness checklist. ZigZag is not production-ready until every item in
this document, [zigzag-security-params.md](zigzag-security-params.md),
[zigzag-post-strategy.md](zigzag-post-strategy.md), and
[zigzag-ceremony-and-adoption.md](zigzag-ceremony-and-adoption.md) is resolved.

Production deployment means adding ZigZag as a protocol-recognized Filecoin proof family. It is
not a feature-gated replacement hidden under an existing Stacked DRG proof type.

## 1. Freeze security parameters

Do not reuse Stacked DRG values without a ZigZag security argument.

Required outputs for every supported sector size:

- number of ZigZag layers
- layer challenge schedule, including whether challenges are fixed or tapered
- minimum challenge counts for interactive proofs
- partition counts and per-partition challenge caps
- expansion degree and graph construction assumptions
- soundness target
- constraint-cost estimates for the selected circuit

The finalized values should be wired into ZigZag-specific parameter maps rather than continuing
to share Stacked constants such as `LAYERS`, `minimum_challenges`, or `POREP_PARTITIONS`.

## 2. Finalize the commitment model

ZigZag exposes:

- `comm_d`: standard Sha256 CommD
- `comm_r`: final-layer replica root
- `comm_r_star`: binding over the replica id and per-layer replica roots

Filecoin miner state currently stores a single sealed commitment. The FIP must choose and specify
the on-chain representation. The two viable options are:

- Store `comm_r_star` as the sealed commitment.
- Store `zigzag_comm_r_bound(comm_r, comm_r_star) = H(comm_r || comm_r_star)` as the sealed
  commitment.

If the bound commitment is chosen, the seal proof or proof inputs must still carry enough data for
the verifier to recover both `comm_r` and `comm_r_star`. Production code should not rely on an
undocumented proof envelope.

Local devnets may use a shared sidecar file to map a raw 192-byte proof back to ZigZag's
`comm_r` and `comm_r_star` while reusing an existing actor proof type. That is not a production
ABI. Production must encode the commitment layout, proof byte limits, and verifier inputs in the
registered proof type and actor/syscall interfaces.

## 3. Register protocol proof types

Allocate official protocol identifiers for:

- ZigZag seal proof variants
- ZigZag WinningPoSt variants
- ZigZag WindowPoSt variants
- any aggregation variants, if aggregation is supported at launch

The registration must define:

- `porep_id` derivation
- supported sector sizes
- API version
- proof serialization format
- commitment layout
- parameter cache identifiers
- verifier behavior in actors and node implementations

Existing Stacked proof types must remain valid for existing sectors.

## 4. Complete trusted setup and parameter publication

ZigZag Groth16 circuits use a distinct cache namespace and require distinct parameters.

Required work:

1. Complete security parameterization first.
2. Run or reuse an appropriate Powers-of-Tau phase.
3. Run circuit-specific phase-2 ceremonies for every supported ZigZag seal configuration.
4. Run ceremonies for ZigZag binary-tree WinningPoSt and WindowPoSt parameters.
5. Publish `.params`, `.vk`, and metadata digests in `parameters.json`.
6. Update `paramfetch`, `paramcache`, release packaging, and checksum validation scripts.
7. Document that local/dev parameters are unsafe for production.

`paramcache --only-zigzag` is acceptable for local seal parameters. It is not ceremony-grade. The
production path must use published parameter manifests and verifiable digests.

Do not enable `FIL_PROOFS_ZIGZAG_GENERATE_MISSING_PARAMS` in production. Production nodes should
fail closed when a ZigZag parameter file is missing or fails digest validation.

## 5. Implement production PoSt support

The chosen strategy is binary-tree FallbackPoSt for ZigZag-native deployment.

Required work:

- Persist the final ZigZag replica tree in a PoSt-compatible disk layout.
- Add ZigZag WinningPoSt and WindowPoSt public parameter builders.
- Add `cache_zigzag_post_params` to `paramcache`.
- Add ZigZag PoSt parameter IDs to `parameters.json`.
- Expose generation and verification APIs from `filecoin-proofs`.
- Expose FFI functions for node and miner implementations.
- Wire Lotus, Curio, Venus, and any other supported implementations to generate and verify ZigZag
  PoSt proofs.

Reusing Stacked PoSt parameters is not the recommended production path unless the PoSt strategy is
formally revised.

## 6. Make sealing resumable and cache-safe

Production sealers need restart-safe, task-safe, and versioned cache state.

Required work:

- Add a stable persisted aux format for ZigZag.
- Re-open persisted `zigzag-tree-d` and final/intermediate replica trees after process restart.
- Add a `zigzag_prove_from_cache` API or equivalent.
- Validate cache contents against `comm_d`, `comm_r`, `comm_r_star`, `replica_id`, sector id,
  ticket, `porep_id`, API version, and sector size.
- Use atomic writes for aux and tree manifests.
- Define cleanup behavior for failed or partial replication.
- Define cache compatibility across releases.

The current in-memory `ZigZagProverState` path is acceptable for tests, but not sufficient for
production miners.

## 7. Expose stable APIs and FFI

Production should expose ZigZag as first-class APIs rather than hidden branches under Stacked
functions.

Required work:

- Add or update the API crate used by `filecoin-ffi`.
- Expose pre-commit, prove, verify, unseal, WinningPoSt, and WindowPoSt APIs.
- Add Go FFI wrappers and headers.
- Preserve the existing Stacked APIs.
- Add explicit proof-type dispatch rather than environment-only switching.
- Define serialization formats with version tags.
- Add compatibility tests across Rust, FFI, Go wrappers, and node call sites.

The FFI surface must support both new ZigZag sectors and existing Stacked sectors in the same
node.

## 8. Integrate node and miner implementations

At minimum, production requires coordinated changes in:

| Component | Required production work |
|-----------|--------------------------|
| Builtin actors | Accept ZigZag proof types and call the correct verifier. |
| Lotus | Verify ZigZag seal and PoSt proofs; fetch ZigZag parameters; preserve Stacked sectors. |
| Curio | Select ZigZag sealing tasks; persist ZigZag caches; unseal through ZigZag extraction; generate ZigZag PoSt. |
| Venus and other nodes | Implement the same verifier and parameter behavior. |
| Tooling | Support ZigZag proof type inspection, parameter verification, and diagnostics. |

The network must support mixed sectors: existing Stacked sectors and new ZigZag sectors may
coexist for a long migration window.

## 9. Define aggregation and batch behavior

Decide whether ZigZag launches with seal aggregation.

If aggregation is supported:

- define aggregation proof types,
- generate aggregation parameters,
- implement aggregation and verification APIs,
- update actors and node syscalls,
- add cross-implementation tests.

If aggregation is not supported at launch:

- explicitly reject aggregation for ZigZag proof types,
- ensure miners submit individual proofs,
- document the operational cost.

Batch verification must support mixed Stacked and ZigZag sectors.

## 10. Audit and test

Required audit scope:

- ZigZag graph construction and graph reversal
- VDE encode/decode correctness
- SHA256 KDF substitution
- Poseidon `comm_r_star` binding
- Sha256 CommD compatibility
- challenge derivation and seed binding
- circuit public input ordering
- fast-extraction assumptions
- persisted cache integrity
- PoSt integration over the final replica tree

Required tests:

- unit tests for every ZigZag parameter schedule
- seal/prove/verify/unseal tests for every supported sector size
- restart/resume sealing tests
- corrupt-cache negative tests
- wrong seed, wrong ticket, wrong prover, wrong sector id, wrong `comm_r_star` tests
- FFI roundtrips
- Lotus/Curio integration tests
- WindowPoSt and WinningPoSt tests
- mixed Stacked/ZigZag chain tests
- parameter manifest checksum tests

## 11. Network upgrade and rollout

The rollout should follow this order:

```text
security params -> PoSt strategy -> ceremonies -> parameters.json -> FIP -> implementations -> calibration/devnet -> audit -> network upgrade
```

Before mainnet activation:

- publish all parameters and digests,
- ship node releases with ZigZag disabled before the upgrade epoch,
- complete calibration or long-running devnet soak tests,
- verify mixed-sector behavior,
- document miner operational requirements,
- document rollback and incident-response procedures.

## Production readiness checklist

- [ ] ZigZag-specific security parameters are finalized.
- [ ] Commitment layout is specified in a FIP.
- [ ] Official ZigZag proof types are registered.
- [ ] Seal parameters are generated through a production ceremony.
- [ ] Binary-tree ZigZag PoSt parameters are generated and published.
- [ ] `parameters.json` and `paramfetch` include ZigZag seal and PoSt params.
- [ ] `filecoin-proofs` exposes resumable seal, prove, verify, unseal, and PoSt APIs.
- [ ] `filecoin-ffi` exposes first-class ZigZag APIs.
- [ ] Builtin actors verify ZigZag proof types.
- [ ] Lotus, Curio, and other implementations support mixed Stacked/ZigZag sectors.
- [ ] Aggregation support is either implemented or explicitly disabled.
- [ ] External audits are complete.
- [ ] Long-running devnet/calibration tests have passed.
