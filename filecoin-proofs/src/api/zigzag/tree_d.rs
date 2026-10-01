//! ZigZag-only TreeD preparation. Reuses the SDR StoreConfig / binary SHA-256 store format.
//! Curio's lib/proof/treed_build.go at ce15c0c92209366a5523b803e9c159baa2ffb66a writes the
//! same layout: unmodified 32-byte leaves, then SHA-256 nodes (top two bits cleared), root last.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use filecoin_hashers::{Domain, HashFunction, Hasher};
use merkletree::store::StoreConfig;
use rayon::prelude::*;
use storage_proofs_core::{
    merkle::{create_base_merkle_tree, BinaryMerkleTree},
    util::NODE_SIZE,
};

use crate::{
    constants::{DefaultPieceDomain, DefaultPieceHasher},
    types::Commitment,
};

const ID: &str = "zigzag-tree-d";
// At most 1 MiB of children plus 512 KiB of parents, independent of sector size / worker count.
const CHILD_BUFFER_BYTES: usize = 1024 * 1024;

pub(super) fn build(
    data: &[u8],
    cache_path: Option<&Path>,
) -> Result<BinaryMerkleTree<DefaultPieceHasher>> {
    let started = Instant::now();
    if let Some(path) = cache_path {
        fs::create_dir_all(path).context("create ZigZag TreeD cache")?;
        // DiskStore's configured builder can reopen an existing file without hashing the supplied
        // data. Never let that implicit cache hit bypass piece/CommD validation. Retry adapters use
        // fresh private workspaces; intentional reuse goes through the fully validated import API.
        match fs::symlink_metadata(StoreConfig::data_path(path, ID)) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
            Ok(_) => anyhow::bail!(
                "ZigZag TreeD already exists; use a fresh cache or explicit validated import"
            ),
        }
    }
    let config = cache_path.map(|path| StoreConfig::new(path, ID, 0));
    let tree = create_base_merkle_tree::<BinaryMerkleTree<DefaultPieceHasher>>(
        config,
        data.len() / NODE_SIZE,
        data,
    )?;
    log::info!(target: "zigzag_precommit", "phase=tree_d_build elapsed_ms={}", started.elapsed().as_millis());
    Ok(tree)
}

pub(super) fn import(
    source: &Path,
    cache_path: &Path,
    data: &[u8],
    comm_d: Commitment,
) -> Result<BinaryMerkleTree<DefaultPieceHasher>> {
    let leaves = data.len() / NODE_SIZE;
    ensure!(
        leaves >= 2 && leaves.is_power_of_two(),
        "invalid ZigZag TreeD leaf count"
    );
    let nodes = leaves
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .context("TreeD size overflow")?;
    let expected_bytes = nodes
        .checked_mul(NODE_SIZE)
        .context("TreeD byte size overflow")? as u64;
    ensure!(
        fs::symlink_metadata(source)?.file_type().is_file(),
        "TreeD source must be a regular file"
    );
    let mut input = File::open(source).context("open source TreeD")?;
    ensure!(
        input.metadata()?.len() == expected_bytes,
        "TreeD source size mismatch"
    );
    fs::create_dir_all(cache_path)?;
    let destination = StoreConfig::data_path(cache_path, ID);
    // Never overwrite an existing generation or hard-link the caller's mutable source. A copy
    // keeps Curio's lifetime/cleanup independent and makes subsequent validation self-contained.
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .context("create private ZigZag TreeD copy (destination must not exist)")?;
    let result = (|| {
        let started = Instant::now();
        ensure!(
            io::copy(&mut input, &mut output)? == expected_bytes,
            "TreeD changed while copying"
        );
        output.sync_all()?;
        log::info!(target: "zigzag_precommit", "phase=tree_d_copy elapsed_ms={}", started.elapsed().as_millis());
        let started = Instant::now();
        validate(&destination, data, comm_d)?;
        let tree = super::reopen_zigzag_tree::<BinaryMerkleTree<DefaultPieceHasher>>(
            cache_path, ID, nodes,
        )?;
        log::info!(target: "zigzag_precommit", "phase=tree_d_validate elapsed_ms={}", started.elapsed().as_millis());
        Ok(tree)
    })();
    drop(output);
    if result.is_err() {
        // Only this invocation's newly-created file is removed. The source and older cache survive.
        fs::remove_file(&destination).context("remove rejected ZigZag TreeD copy")?;
    }
    result
}

fn validate(path: &Path, data: &[u8], comm_d: Commitment) -> Result<()> {
    let mut stored = BufReader::new(File::open(path)?);
    let mut children = vec![0u8; CHILD_BUFFER_BYTES.min(data.len())];
    // Checking only the stored root would miss damaged leaves or internal nodes. First bind
    // every leaf to the actual replica input, then authenticate every stored parent bottom-up.
    for chunk in data.chunks(children.len()) {
        stored.read_exact(&mut children[..chunk.len()])?;
        ensure!(
            children[..chunk.len()] == *chunk,
            "TreeD leaves do not match sector data"
        );
    }
    let mut source = BufReader::new(File::open(path)?);
    let mut parents = vec![0u8; children.len() / 2];
    let mut width = data.len();
    let mut level_offset = 0;
    while width > NODE_SIZE {
        source.seek(SeekFrom::Start(level_offset as u64))?;
        let mut remaining = width;
        while remaining > 0 {
            let count = children.len().min(remaining);
            source.read_exact(&mut children[..count])?;
            stored.read_exact(&mut parents[..count / 2])?;
            children[..count]
                .par_chunks_exact(NODE_SIZE * 2)
                .zip(parents[..count / 2].par_chunks_exact(NODE_SIZE))
                .try_for_each(|(pair, expected)| -> Result<()> {
                    let left = DefaultPieceDomain::try_from_bytes(&pair[..NODE_SIZE])?;
                    let right = DefaultPieceDomain::try_from_bytes(&pair[NODE_SIZE..])?;
                    let computed = <DefaultPieceHasher as Hasher>::Function::hash2(&left, &right);
                    ensure!(
                        AsRef::<[u8]>::as_ref(&computed) == expected,
                        "invalid TreeD internal node"
                    );
                    Ok(())
                })?;
            remaining -= count;
        }
        level_offset += width;
        width /= 2;
    }
    ensure!(
        parents[..NODE_SIZE] == comm_d,
        "TreeD root does not match comm_d"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use storage_proofs_core::merkle::{MerkleProofTrait, MerkleTreeTrait};

    fn fixture(size: usize) -> (tempfile::TempDir, Vec<u8>, Commitment) {
        let root = tempfile::tempdir().unwrap();
        let mut data = vec![0u8; size];
        for (i, node) in data.chunks_exact_mut(NODE_SIZE).enumerate() {
            node[..8].copy_from_slice(&(i as u64).to_le_bytes());
        }
        let tree = build(&data, Some(root.path())).unwrap();
        let comm_d = super::super::commitment_from_domain(tree.root());
        (root, data, comm_d)
    }

    #[test]
    fn imported_tree_d_validates_across_buffer_boundaries_and_owns_its_copy() {
        let (source, data, comm_d) = fixture(CHILD_BUFFER_BYTES * 2);
        let cache = tempfile::tempdir().unwrap();
        let path = StoreConfig::data_path(source.path(), ID);
        let source_bytes = fs::read(&path).unwrap();
        let tree = import(&path, cache.path(), &data, comm_d).unwrap();
        assert_eq!(fs::read(&path).unwrap(), source_bytes);
        assert_eq!(
            fs::read(StoreConfig::data_path(cache.path(), ID)).unwrap(),
            source_bytes
        );
        drop(source);
        assert!(tree.gen_proof(17).unwrap().validate(17));
        assert_eq!(super::super::commitment_from_domain(tree.root()), comm_d);
    }

    #[test]
    fn rejects_damaged_tree_d_without_trusting_its_root_or_modifying_the_source() {
        let (source, data, comm_d) = fixture(2048);
        let path = StoreConfig::data_path(source.path(), ID);
        let original = fs::read(&path).unwrap();
        for offset in [0, data.len(), original.len() - NODE_SIZE] {
            let mut damaged = original.clone();
            damaged[offset] ^= 1;
            fs::write(&path, &damaged).unwrap();
            let cache = tempfile::tempdir().unwrap();
            assert!(import(&path, cache.path(), &data, comm_d).is_err());
            assert!(!StoreConfig::data_path(cache.path(), ID).exists());
            assert_eq!(fs::read(&path).unwrap(), damaged);
        }
    }

    #[test]
    fn rejects_wrong_length_data_commitment_and_existing_destination() {
        let (source, data, comm_d) = fixture(2048);
        let path = StoreConfig::data_path(source.path(), ID);
        let original = fs::read(&path).unwrap();
        for size in [data.len(), original.len() - 1, original.len() + 1] {
            let mut bytes = original.clone();
            bytes.resize(size, 0);
            fs::write(&path, bytes).unwrap();
            let cache = tempfile::tempdir().unwrap();
            assert!(import(&path, cache.path(), &data, comm_d).is_err());
        }
        fs::write(&path, &original).unwrap();
        let cache = tempfile::tempdir().unwrap();
        let mut wrong_data = data.clone();
        wrong_data[0] ^= 1;
        assert!(import(&path, cache.path(), &wrong_data, comm_d).is_err());
        let mut wrong_comm_d = comm_d;
        wrong_comm_d[0] ^= 1;
        assert!(import(&path, cache.path(), &data, wrong_comm_d).is_err());
        let destination = StoreConfig::data_path(cache.path(), ID);
        fs::write(&destination, b"previous generation").unwrap();
        assert!(import(&path, cache.path(), &data, comm_d).is_err());
        assert_eq!(fs::read(destination).unwrap(), b"previous generation");
        assert_eq!(fs::read(&path).unwrap(), original);
        assert!(import(&path, source.path(), &data, comm_d).is_err());
        assert_eq!(fs::read(&path).unwrap(), original);
    }

    #[test]
    fn ordinary_build_never_implicitly_trusts_an_existing_tree_d() {
        let (source, mut data, _) = fixture(2048);
        let path = StoreConfig::data_path(source.path(), ID);
        let before = fs::read(&path).unwrap();
        data[0] ^= 1;
        assert!(build(&data, Some(source.path())).is_err());
        assert_eq!(fs::read(path).unwrap(), before);
    }
}
