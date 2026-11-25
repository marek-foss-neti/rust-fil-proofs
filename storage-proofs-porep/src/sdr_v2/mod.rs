//! SDR v2: Stacked DRG labels computed with a keyed PRP, decodable one layer at a time in
//! parallel.
//!
//! The construction is from "Parallel decoding for Filecoin's PoRep: SDR v2 and ZigZag"
//! (Giacomelli, Nizzardo; draft of 16 November 2026), Section 4. The graph is the stacked graph of
//! SDR. A label is `Threefish-256(key, input)`: the key is a SHA-256 over the labels of the DRG
//! parents (same layer), and the input is one 32-bit component from each of the 8 expander parents
//! (previous layer), or the data in layer 1. The replica is the last layer.
//!
//! This module holds what the unseal benchmark needs: the label, a simple encoder to produce
//! replicas, and the parallel decoder. Commitments, proofs and the circuit are not implemented.
//! The byte-level layout is a working specification that the report does not fix; it is defined
//! in [`label`] and [`threefish`].
//!
//! `label`, `threefish`, `permutation`, `drg`, `encode` and `decode` depend only on `sha2`, `rayon`,
//! `blake2b_simd`, `anyhow` and `log`, so they can be compiled and tested outside this crate.

mod decode;
mod drg;
mod encode;
mod graph;
pub mod label;
mod permutation;
mod sector;
pub mod threefish;

pub use decode::{decode_layers, DEFAULT_DECODE_BATCH, MAX_DECODE_BATCH};
pub use drg::{DrgParentTable, DrgSampler};
pub use encode::encode_layers;
pub use graph::SdrV2Graph;
pub use label::{
    LabelGraph, LayerEvent, LayerKeyer, SdrV2Params, COMPONENTS, COMPONENT_BYTES, DRG_DEGREE,
    MAX_KEY_BLOCKS, MIN_KEY_BLOCKS, NODE_BYTES,
};
pub use permutation::WordPermutation;
pub use sector::{
    decode_sector, encode_sector, LayerBuffer, SdrV2Settings, DEFAULT_KEY_BLOCKS, DEFAULT_LAYERS,
};
