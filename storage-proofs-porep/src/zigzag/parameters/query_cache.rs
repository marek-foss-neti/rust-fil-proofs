//! A bounded, single-C2 cache over the existing Bellperson ParameterSource.
//!
//! Families are reserved before any decoding. Entries are never evicted while
//! the prover can hold builders, so a live Arc cannot escape budget accounting
//! through an eviction. Successful batches await their MSM results. On an error,
//! Bellperson can leave CPU tasks finishing with their own Arcs; dropping this
//! adapter releases its references, not those owned by outstanding prover work.
use std::{env, sync::Mutex};

use super::*;

const GIB: u64 = 1 << 30;
const MEMORY_CEILING: u64 = 60 * GIB;
const DEFAULT_RESERVE: u64 = 56 * GIB;
const FAMILIES: [&str; 5] = ["h", "l", "a", "b_g1", "b_g2"];

#[derive(Clone, Copy, Debug, Serialize)]
pub struct QueryCacheConfig {
    pub requested_bytes: u64,
    pub reserve_bytes: u64,
    pub memory_limit_bytes: Option<u64>,
    pub effective_bytes: u64,
}

impl QueryCacheConfig {
    /// The reserve includes witness, FFT/MSM, uncached queries, mmap residency
    /// and allocator overhead. It is a calibrated allowance, not an RSS reading.
    pub fn from_env() -> Result<Self> {
        fn value(name: &str, default: u64) -> Result<u64> {
            match env::var(name) {
                Err(env::VarError::NotPresent) => Ok(default),
                Ok(raw) if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) => {
                    let parsed: u64 = raw.parse().with_context(|| format!("invalid {name}"))?;
                    ensure!(parsed <= MEMORY_CEILING, "{name} exceeds 60 GiB");
                    Ok(parsed)
                }
                _ => anyhow::bail!("{name} must be an unsigned byte count"),
            }
        }
        let requested = value("FIL_PROOFS_ZIGZAG_QUERY_CACHE_BYTES", 0)?;
        let reserve = value(
            "FIL_PROOFS_ZIGZAG_QUERY_CACHE_RESERVE_BYTES",
            DEFAULT_RESERVE,
        )?;
        ensure!(reserve > 0, "ZigZag query cache reserve must be positive");
        // Resolve the process group, including tighter ancestor limits. An
        // unbounded, unavailable or nonstandard mount disables retention.
        let limit = process_memory_limit();
        Ok(Self::with_limit(requested, reserve, limit))
    }

    fn with_limit(
        requested_bytes: u64,
        reserve_bytes: u64,
        memory_limit_bytes: Option<u64>,
    ) -> Self {
        let effective_bytes = requested_bytes.min(
            memory_limit_bytes
                .map(|limit| limit.min(MEMORY_CEILING).saturating_sub(reserve_bytes))
                .unwrap_or(0),
        );
        Self {
            requested_bytes,
            reserve_bytes,
            memory_limit_bytes,
            effective_bytes,
        }
    }
}

fn process_memory_limit() -> Option<u64> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    for line in groups.lines() {
        let mut fields = line.splitn(3, ':');
        let hierarchy = fields.next()?;
        let controllers = fields.next()?;
        let relative = fields.next()?;
        let (mount, file) = if hierarchy == "0" && controllers.is_empty() {
            ("/sys/fs/cgroup", "memory.max")
        } else if controllers.split(',').any(|c| c == "memory") {
            ("/sys/fs/cgroup/memory", "memory.limit_in_bytes")
        } else {
            continue;
        };
        return limit_in_hierarchy(std::path::Path::new(mount), relative, file);
    }
    None
}

fn limit_in_hierarchy(mount: &std::path::Path, group: &str, file: &str) -> Option<u64> {
    use std::path::Component;
    let group = std::path::Path::new(group);
    if !group.is_absolute()
        || group
            .components()
            .any(|c| !matches!(c, Component::RootDir | Component::Normal(_)))
    {
        return None;
    }
    let mut current = mount.join(group.strip_prefix("/").ok()?);
    let mut limit = None;
    loop {
        let raw = match std::fs::read_to_string(current.join(file)) {
            Ok(raw) => raw,
            // The real cgroup v2 root has no memory.max. Preserve limits
            // already read below it. A namespaced root can have memory.max;
            // read and apply it normally, including a tighter root limit.
            Err(error)
                if current == mount
                    && file == "memory.max"
                    && error.kind() == std::io::ErrorKind::NotFound =>
            {
                return limit;
            }
            // Missing intermediate files, other read errors and v1 failures
            // leave the hierarchy unresolved, so retention stays disabled.
            Err(_) => return None,
        };
        // cgroup v1 represents an unlimited setting with a value near i64::MAX.
        let value = if raw.trim() == "max" {
            None
        } else {
            Some(raw.trim().parse::<u64>().ok()?).filter(|n| *n < (1 << 60))
        };
        if let Some(value) = value {
            limit = Some(limit.map_or(value, |previous: u64| previous.min(value)));
        }
        if current == mount {
            return limit;
        }
        current.pop();
    }
}

type Slot<C> = Mutex<Option<Arc<Vec<C>>>>;

pub(crate) struct CachedParameters<'a> {
    source: &'a ZigZagParameters,
    selected: [bool; 5],
    sizes: [u64; 5],
    // Each family has its own initialization lock; unrelated families never
    // wait for a global lock covering disk I/O or point decoding.
    g1: [Slot<G1Affine>; 4],
    g2: Slot<G2Affine>,
    config: QueryCacheConfig,
    reserved_bytes: u64,
}

impl<'a> CachedParameters<'a> {
    pub(crate) fn new(source: &'a ZigZagParameters, config: QueryCacheConfig) -> Result<Self> {
        let mut sizes = [0; 5];
        for (i, family) in source.layout().families.iter().enumerate() {
            let point_size = if i == 4 {
                std::mem::size_of::<G2Affine>()
            } else {
                std::mem::size_of::<G1Affine>()
            };
            sizes[i] = (family.points as u64)
                .checked_mul(point_size as u64)
                .context("query cache size overflow")?;
        }
        // Prefer the smallest families: the first 512-profile candidate is
        // B-G1 (~2.99 GiB). Never retain everything merely because it fits disk.
        let mut order = [0usize, 1, 2, 3, 4];
        order.sort_by_key(|&i| (sizes[i], i));
        let mut selected = [false; 5];
        let mut reserved_bytes = 0;
        for i in order {
            if sizes[i] > 0 && sizes[i] <= config.effective_bytes - reserved_bytes {
                selected[i] = true;
                reserved_bytes += sizes[i];
            }
        }
        let selected_names: Vec<_> = FAMILIES
            .iter()
            .enumerate()
            .filter_map(|(i, name)| selected[i].then_some(*name))
            .collect();
        let family_decoded_bytes: serde_json::Map<_, _> = FAMILIES
            .iter()
            .enumerate()
            .map(|(i, name)| (name.to_string(), serde_json::json!(sizes[i])))
            .collect();
        log::info!(target: "zigzag_parameters", "zigzag_parameters {}", serde_json::json!({
            "kind": "query_cache", "scope": "one_c2", "config": config,
            "reserved_bytes": reserved_bytes, "selected_families": selected_names,
            "family_decoded_bytes": family_decoded_bytes, "parameter_id": source.diagnostics()["parameter_id"],
            "policy": "smallest families first; no eviction; reserve includes uncached prover work",
        }));
        Ok(Self {
            source,
            selected,
            sizes,
            config,
            reserved_bytes,
            g1: std::array::from_fn(|_| Mutex::new(None)),
            g2: Mutex::new(None),
        })
    }

    fn get<C>(
        &self,
        family: usize,
        slot: &Slot<C>,
        read: impl FnOnce() -> Result<Arc<Vec<C>>, SynthesisError>,
    ) -> Result<Arc<Vec<C>>, SynthesisError> {
        if !self.selected[family] {
            return read();
        }
        let mut value = slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(points) = value.as_ref() {
            self.record(family, "hit");
            return Ok(Arc::clone(points));
        }
        // On error or unwind the slot stays empty and the fixed reservation is
        // unchanged. A later call can retry the same validated source.
        let points = read()?;
        let allocated = (points.capacity() as u64).checked_mul(std::mem::size_of::<C>() as u64);
        if allocated.is_some_and(|bytes| bytes <= self.sizes[family]) {
            *value = Some(Arc::clone(&points));
            self.record(family, "miss_retained");
        } else {
            // A future source may overallocate its Vec. Never retain more than
            // the preflight reservation; proving can continue without retention.
            self.record(family, "skip_capacity");
        }
        Ok(points)
    }

    fn record(&self, family: usize, outcome: &str) {
        log::info!(target: "zigzag_parameters", "zigzag_parameters {}", serde_json::json!({
            "kind": "query_cache_access", "family": FAMILIES[family],
            "outcome": outcome, "decoded_bytes": self.sizes[family],
            "reserved_bytes": self.reserved_bytes, "budget_bytes": self.config.effective_bytes,
        }));
    }

    fn validate_split(&self, family: usize, inputs: usize) -> Result<(), SynthesisError> {
        if inputs > self.source.layout().families[family].points {
            return Err(invalid("input query split out of bounds").into());
        }
        Ok(())
    }
}

impl ParameterSource<Bls12> for &CachedParameters<'_> {
    type G1Builder = (Arc<Vec<G1Affine>>, usize);
    type G2Builder = (Arc<Vec<G2Affine>>, usize);
    fn get_vk(&self, inputs: usize) -> Result<&VerifyingKey<Bls12>, SynthesisError> {
        self.source.get_vk(inputs)
    }
    fn get_h(&self, points: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((
            self.get(0, &self.g1[0], || Ok(self.source.get_h(points)?.0))?,
            0,
        ))
    }
    fn get_l(&self, points: usize) -> Result<Self::G1Builder, SynthesisError> {
        Ok((
            self.get(1, &self.g1[1], || Ok(self.source.get_l(points)?.0))?,
            0,
        ))
    }
    fn get_a(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.validate_split(2, inputs)?;
        let points = self.get(2, &self.g1[2], || Ok(self.source.get_a(inputs, aux)?.0 .0))?;
        Ok(((Arc::clone(&points), 0), (points, inputs)))
    }
    fn get_b_g1(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G1Builder, Self::G1Builder), SynthesisError> {
        self.validate_split(3, inputs)?;
        let points = self.get(3, &self.g1[3], || {
            Ok(self.source.get_b_g1(inputs, aux)?.0 .0)
        })?;
        Ok(((Arc::clone(&points), 0), (points, inputs)))
    }
    fn get_b_g2(
        &self,
        inputs: usize,
        aux: usize,
    ) -> Result<(Self::G2Builder, Self::G2Builder), SynthesisError> {
        self.validate_split(4, inputs)?;
        let points = self.get(4, &self.g2, || Ok(self.source.get_b_g2(inputs, aux)?.0 .0))?;
        Ok(((Arc::clone(&points), 0), (points, inputs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zigzag::parameters::tests::fixture;

    fn fixture_source() -> (tempfile::NamedTempFile, ZigZagParameters) {
        let (file, _) = fixture();
        let compact = CompactParameters::open(file.path(), "cache-test".into(), true).unwrap();
        (file, ZigZagParameters::Compact(compact))
    }

    #[test]
    fn budget_reserves_prover_memory_and_never_assumes_unbounded_ram() {
        assert_eq!(
            QueryCacheConfig::with_limit(4 * GIB, 56 * GIB, Some(60 * GIB)).effective_bytes,
            4 * GIB
        );
        assert_eq!(
            QueryCacheConfig::with_limit(4 * GIB, 56 * GIB, Some(58 * GIB)).effective_bytes,
            2 * GIB
        );
        assert_eq!(
            QueryCacheConfig::with_limit(4 * GIB, 56 * GIB, Some(55 * GIB)).effective_bytes,
            0
        );
        assert_eq!(
            QueryCacheConfig::with_limit(8 * GIB, 56 * GIB, Some(80 * GIB)).effective_bytes,
            4 * GIB
        );
        assert_eq!(
            QueryCacheConfig::with_limit(4 * GIB, 56 * GIB, None).effective_bytes,
            0
        );
    }

    #[test]
    fn hierarchy_uses_lowest_limit_and_fails_closed_when_unresolved() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("worker/job");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.path().join("memory.max"), (60 * GIB).to_string()).unwrap();
        std::fs::write(
            root.path().join("worker/memory.max"),
            (58 * GIB).to_string(),
        )
        .unwrap();
        std::fs::write(nested.join("memory.max"), "max\n").unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            Some(58 * GIB)
        );
        // A container's namespaced root can be the tightest constraint.
        std::fs::write(root.path().join("memory.max"), (57 * GIB).to_string()).unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            Some(57 * GIB)
        );
        assert_eq!(
            limit_in_hierarchy(root.path(), "/missing", "memory.max"),
            None
        );
        assert_eq!(
            limit_in_hierarchy(root.path(), "/../worker", "memory.max"),
            None
        );
        std::fs::write(root.path().join("memory.max"), "max").unwrap();
        assert_eq!(limit_in_hierarchy(root.path(), "/", "memory.max"), None);
        std::fs::write(
            root.path().join("memory.limit_in_bytes"),
            "9223372036854771712",
        )
        .unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/", "memory.limit_in_bytes"),
            None
        );
    }

    #[test]
    fn v2_root_without_memory_max_preserves_descendant_limits() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("worker/job");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.path().join("worker/memory.max"), "max\n").unwrap();
        std::fs::write(nested.join("memory.max"), (60 * GIB).to_string()).unwrap();
        let limit = limit_in_hierarchy(root.path(), "/worker/job", "memory.max");
        assert_eq!(limit, Some(60 * GIB));
        assert_eq!(
            QueryCacheConfig::with_limit(4 * GIB, 56 * GIB, limit).effective_bytes,
            4 * GIB
        );
        std::fs::write(root.path().join("worker/memory.max"), (58 * GIB).to_string()).unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            Some(58 * GIB)
        );
        assert_eq!(limit_in_hierarchy(root.path(), "/", "memory.max"), None);
        std::fs::write(root.path().join("worker/memory.max"), "max").unwrap();
        std::fs::write(nested.join("memory.max"), "max").unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            None
        );
    }

    #[test]
    fn hierarchy_read_errors_do_not_reuse_a_partial_limit() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("worker/job");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("memory.max"), (60 * GIB).to_string()).unwrap();
        // The missing-file exception must not apply to an intermediate group.
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            None
        );
        std::fs::write(root.path().join("worker/memory.max"), "max").unwrap();
        // An unreadable or malformed root is not the v2 missing-file case.
        let root_limit = root.path().join("memory.max");
        std::fs::create_dir(&root_limit).unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            None
        );
        std::fs::remove_dir(&root_limit).unwrap();
        std::fs::write(&root_limit, "invalid").unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.max"),
            None
        );
        // A missing v1 root limit must still disable retention.
        std::fs::write(nested.join("memory.limit_in_bytes"), (60 * GIB).to_string()).unwrap();
        std::fs::write(
            root.path().join("worker/memory.limit_in_bytes"),
            "9223372036854771712",
        )
        .unwrap();
        assert_eq!(
            limit_in_hierarchy(root.path(), "/worker/job", "memory.limit_in_bytes"),
            None
        );
    }

    #[test]
    fn hits_share_points_preserve_offsets_and_validate_even_after_fill() {
        let (_file, source) = fixture_source();
        let cache =
            CachedParameters::new(&source, QueryCacheConfig::with_limit(10000, 1, Some(10001)))
                .unwrap();
        assert!(cache.selected.iter().all(|&x| x));
        assert!(cache.reserved_bytes <= cache.config.effective_bytes);
        for i in [0, 1, 2, 3] {
            assert_eq!(
                (&cache).get_a(i, 4 - i).unwrap(),
                (&source).get_a(i, 4 - i).unwrap()
            );
            assert_eq!(
                (&cache).get_b_g1(i, 3 - i).unwrap(),
                (&source).get_b_g1(i, 3 - i).unwrap()
            );
            assert_eq!(
                (&cache).get_b_g2(i, 3 - i).unwrap(),
                (&source).get_b_g2(i, 3 - i).unwrap()
            );
        }
        let a = (&cache).get_a(1, 3).unwrap();
        let b = (&cache).get_a(2, 2).unwrap();
        assert!(Arc::ptr_eq(&a.0 .0, &b.1 .0));
        assert_eq!(a.1 .1, 1);
        assert_eq!(b.1 .1, 2);
        assert!((&cache).get_a(5, 0).is_err());
        assert!((&cache).get_b_g1(4, 0).is_err());
        assert!((&cache).get_b_g2(4, 0).is_err());
        assert!((&cache).get_vk(3).is_err());
        for get in [
            |p: &CachedParameters<'_>| (&p).get_h(2).unwrap().0,
            |p: &CachedParameters<'_>| (&p).get_l(3).unwrap().0,
        ] {
            assert!(Arc::ptr_eq(&get(&cache), &get(&cache)));
        }
        let weak = Arc::downgrade(&a.0 .0);
        drop(a);
        drop(b);
        drop(cache);
        assert!(weak.upgrade().is_none());
        let reopened =
            CachedParameters::new(&source, QueryCacheConfig::with_limit(0, 1, Some(10001)))
                .unwrap();
        let x = (&reopened).get_h(2).unwrap().0;
        let y = (&reopened).get_h(2).unwrap().0;
        assert_eq!(x, y);
        assert!(!Arc::ptr_eq(&x, &y));
    }

    #[test]
    fn concurrent_requests_single_fill_and_families_have_independent_locks() {
        let (_file, source) = fixture_source();
        let cache =
            CachedParameters::new(&source, QueryCacheConfig::with_limit(10000, 1, Some(10001)))
                .unwrap();
        std::thread::scope(|scope| {
            let calls: Vec<_> = (0..8)
                .map(|_| scope.spawn(|| (&cache).get_b_g2(1, 2).unwrap().0 .0))
                .collect();
            let arrays: Vec<_> = calls.into_iter().map(|x| x.join().unwrap()).collect();
            assert!(arrays.iter().all(|x| Arc::ptr_eq(x, &arrays[0])));
        });
        // Blocking one family cannot block an unrelated family initializer.
        let lock = cache.g1[0].lock().unwrap();
        assert!((&cache).get_l(3).is_ok());
        drop(lock);
    }

    #[test]
    fn failed_or_panicking_fill_can_retry_without_losing_reservation() {
        let (_file, source) = fixture_source();
        let cache = CachedParameters::new(&source, QueryCacheConfig::with_limit(192, 1, Some(193)))
            .unwrap();
        assert_eq!(cache.selected, [true, false, false, false, false]);
        assert_eq!(cache.reserved_bytes, 192);
        let error = cache.get(0, &cache.g1[0], || Err(SynthesisError::Unsatisfiable));
        assert!(error.is_err());
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cache.get(0, &cache.g1[0], || panic!("interrupted decode"));
        }));
        assert!(panic.is_err());
        assert_eq!(cache.reserved_bytes, 192);
        let a = (&cache).get_h(2).unwrap().0;
        assert!(Arc::ptr_eq(&a, &(&cache).get_h(2).unwrap().0));
        let l = (&cache).get_l(3).unwrap().0;
        assert!(!Arc::ptr_eq(&l, &(&cache).get_l(3).unwrap().0));
    }

    #[test]
    fn source_identity_is_local_and_overcapacity_is_never_retained() {
        let (_file, source) = fixture_source();
        let (_other_file, other) = fixture_source();
        let config = QueryCacheConfig::with_limit(10000, 1, Some(10001));
        let cache = CachedParameters::new(&source, config).unwrap();
        let independent = CachedParameters::new(&other, config).unwrap();
        let a = (&cache).get_h(2).unwrap().0;
        let b = (&independent).get_h(2).unwrap().0;
        assert_eq!(a, b);
        assert!(!Arc::ptr_eq(&a, &b));
        let mut oversized = Vec::with_capacity(100);
        oversized.resize(3, G1Affine::generator());
        cache
            .get(1, &cache.g1[1], || Ok(Arc::new(oversized)))
            .unwrap();
        assert!(cache.g1[1].lock().unwrap().is_none());
    }
}
