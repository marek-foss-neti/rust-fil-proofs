use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct SetupResources {
    pub cpu_limit: usize,
    pub process_anon_bytes: u64,
    pub host_available_bytes: Option<u64>,
    pub cgroup_limit_bytes: Option<u64>,
    pub cgroup_available_bytes: Option<u64>,
}

impl SetupResources {
    pub fn detect() -> Result<Self> {
        let mut resources = Self {
            cpu_limit: std::thread::available_parallelism()?
                .get()
                .min(rayon::current_num_threads()),
            process_anon_bytes: 0,
            host_available_bytes: None,
            cgroup_limit_bytes: None,
            cgroup_available_bytes: None,
        };
        if cfg!(target_os = "linux") {
            let status = fs::read_to_string("/proc/self/status")?;
            resources.process_anon_bytes = kib_value(&status, "RssAnon:")?
                .checked_add(kib_value(&status, "VmSwap:")?)
                .context("setup RSS overflow")?;
            resources.host_available_bytes = Some(kib_value(
                &fs::read_to_string("/proc/meminfo")?,
                "MemAvailable:",
            )?);
            let groups = fs::read_to_string("/proc/self/cgroup")?;
            let mounts = fs::read_to_string("/proc/self/mountinfo")?;
            if let Some((mount, current, v2)) = memory_group(&groups, &mounts)? {
                let (limit, available) = memory_in_hierarchy(&mount, &current, v2)?;
                resources.cgroup_limit_bytes = limit;
                resources.cgroup_available_bytes = available;
            }
        }
        Ok(resources)
    }

    pub fn budget(self, configured: u64) -> u64 {
        let mut budget = configured;
        if let Some(limit) = self.cgroup_limit_bytes {
            budget = budget.min(limit);
        }
        for available in [self.host_available_bytes, self.cgroup_available_bytes]
            .iter()
            .copied()
            .flatten()
        {
            budget = budget.min(self.process_anon_bytes.saturating_add(available));
        }
        budget
    }
}

fn kib_value(text: &str, key: &str) -> Result<u64> {
    let mut fields = text
        .lines()
        .find(|line| line.split_whitespace().next() == Some(key))
        .with_context(|| format!("missing {key} in setup memory counters"))?
        .split_whitespace();
    fields.next();
    let value: u64 = fields.next().context("missing memory counter")?.parse()?;
    ensure!(
        fields.next() == Some("kB"),
        "unexpected memory counter units"
    );
    value.checked_mul(1024).context("memory counter overflow")
}

fn mount_path(raw: &str) -> PathBuf {
    PathBuf::from(
        raw.replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

fn memory_group(groups: &str, mounts: &str) -> Result<Option<(PathBuf, PathBuf, bool)>> {
    let has_v1_memory = groups.lines().any(|line| {
        line.split(':')
            .nth(1)
            .is_some_and(|controllers| controllers.split(',').any(|name| name == "memory"))
    });
    for line in groups.lines() {
        let fields: Vec<_> = line.splitn(3, ':').collect();
        ensure!(fields.len() == 3, "invalid cgroup membership");
        let v2 = fields[0] == "0" && fields[1].is_empty();
        if v2 && has_v1_memory {
            continue;
        }
        if !v2 && !fields[1].split(',').any(|name| name == "memory") {
            continue;
        }
        let group = Path::new(fields[2]);
        ensure!(
            group.is_absolute()
                && group
                    .components()
                    .all(|part| matches!(part, Component::RootDir | Component::Normal(_))),
            "invalid memory cgroup path"
        );
        for line in mounts.lines() {
            let Some((before, after)) = line.split_once(" - ") else {
                continue;
            };
            let before: Vec<_> = before.split_whitespace().collect();
            let after: Vec<_> = after.split_whitespace().collect();
            if before.len() < 5 || after.len() < 3 {
                continue;
            }
            if !(v2 && after[0] == "cgroup2"
                || !v2 && after[0] == "cgroup" && after[2].split(',').any(|name| name == "memory"))
            {
                continue;
            }
            let root = mount_path(before[3]);
            let mount = mount_path(before[4]);
            if let Ok(relative) = group.strip_prefix(&root) {
                return Ok(Some((mount.clone(), mount.join(relative), v2)));
            }
        }
        bail!("cannot resolve memory cgroup mount for auto setup");
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(path: &Path, limit: &str, usage: u64) {
        fs::create_dir_all(path).unwrap();
        fs::write(path.join("memory.max"), limit).unwrap();
        fs::write(path.join("memory.current"), usage.to_string()).unwrap();
        fs::write(
            path.join("memory.stat"),
            "inactive_file 100\nfile_dirty 20\nfile_writeback 10\n",
        )
        .unwrap();
    }

    #[test]
    fn v2_missing_real_root_keeps_limits_and_tightest_ancestor_headroom() {
        let temp = tempfile::tempdir().unwrap();
        let parent = temp.path().join("service");
        let child = parent.join("worker");
        counters(&parent, "1000", 900);
        counters(&child, "800", 200);
        assert_eq!(
            memory_in_hierarchy(temp.path(), &child, true).unwrap(),
            (Some(800), Some(170))
        );
        fs::write(parent.join("memory.max"), "max").unwrap();
        assert_eq!(
            memory_in_hierarchy(temp.path(), &child, true).unwrap(),
            (Some(800), Some(670))
        );
        fs::remove_file(parent.join("memory.max")).unwrap();
        assert!(memory_in_hierarchy(temp.path(), &child, true).is_err());
    }

    #[test]
    fn namespace_root_and_missing_stats_never_overstate_headroom() {
        let temp = tempfile::tempdir().unwrap();
        counters(temp.path(), "500", 490);
        assert_eq!(
            memory_in_hierarchy(temp.path(), temp.path(), true).unwrap(),
            (Some(500), Some(80))
        );
        fs::write(temp.path().join("memory.stat"), "inactive_file 100\n").unwrap();
        assert_eq!(
            memory_in_hierarchy(temp.path(), temp.path(), true).unwrap(),
            (Some(500), Some(10))
        );
        fs::write(temp.path().join("memory.current"), "700").unwrap();
        assert_eq!(
            memory_in_hierarchy(temp.path(), temp.path(), true).unwrap(),
            (Some(500), Some(0))
        );
        fs::remove_file(temp.path().join("memory.current")).unwrap();
        assert!(memory_in_hierarchy(temp.path(), temp.path(), true).is_err());
    }

    #[test]
    fn v1_usage_and_dirty_counters_are_supported() {
        let temp = tempfile::tempdir().unwrap();
        fs::write(temp.path().join("memory.limit_in_bytes"), "500").unwrap();
        fs::write(temp.path().join("memory.usage_in_bytes"), "450").unwrap();
        fs::write(
            temp.path().join("memory.stat"),
            "total_inactive_file 100\ntotal_dirty 20\ntotal_writeback 10\n",
        )
        .unwrap();
        assert_eq!(
            memory_in_hierarchy(temp.path(), temp.path(), false).unwrap(),
            (Some(500), Some(120))
        );
    }

    #[test]
    fn mounts_handle_subtree_namespaces_nonstandard_paths_and_hybrid_cgroups() {
        let mounts = "1 2 0:1 / /cg rw - cgroup2 cgroup rw\n3 2 0:2 / /legacy rw - cgroup cgroup rw,memory\n";
        assert_eq!(
            memory_group("0::/service\n", mounts).unwrap(),
            Some(("/cg".into(), "/cg/service".into(), true))
        );
        assert_eq!(
            memory_group("0::/\n1:memory:/job\n", mounts).unwrap(),
            Some(("/legacy".into(), "/legacy/job".into(), false))
        );
        let subtree = "1 2 0:1 /service /custom\\040mount rw - cgroup2 cgroup rw\n";
        assert_eq!(
            memory_group("0::/service/job\n", subtree).unwrap(),
            Some(("/custom mount".into(), "/custom mount/job".into(), true))
        );
        assert!(memory_group("0::/../../job\n", mounts).is_err());
        assert!(memory_group("0::/job\n", "").is_err());
        assert!(memory_group("broken", mounts).is_err());
    }

    #[test]
    fn counters_and_budget_reject_bad_data_and_include_shared_cgroup_pressure() {
        assert_eq!(
            kib_value("MemAvailable: 42 kB\n", "MemAvailable:").unwrap(),
            42 * 1024
        );
        assert!(kib_value("MemAvailable: 42 B\n", "MemAvailable:").is_err());
        assert!(kib_value("", "MemAvailable:").is_err());
        let resources = SetupResources {
            cpu_limit: 96,
            process_anon_bytes: 200,
            host_available_bytes: Some(1000),
            cgroup_limit_bytes: Some(900),
            cgroup_available_bytes: Some(100),
        };
        assert_eq!(resources.budget(2000), 300);
        assert_eq!(resources.budget(250), 250);
    }
}

fn memory_in_hierarchy(
    mount: &Path,
    current: &Path,
    v2: bool,
) -> Result<(Option<u64>, Option<u64>)> {
    let mut current = current.to_owned();
    let mut limit: Option<u64> = None;
    let mut available: Option<u64> = None;
    ensure!(current.starts_with(mount), "memory group outside its mount");
    loop {
        let limit_file = current.join(if v2 {
            "memory.max"
        } else {
            "memory.limit_in_bytes"
        });
        let raw = match fs::read_to_string(&limit_file) {
            Ok(raw) => raw,
            // A genuine v2 root has no controller limit. A namespaced root
            // may have one, and must be read just like any other ancestor.
            Err(e) if v2 && current == mount && e.kind() == std::io::ErrorKind::NotFound => break,
            Err(e) => return Err(e).with_context(|| format!("read {}", limit_file.display())),
        };
        let bound = if raw.trim() == "max" {
            None
        } else {
            Some(raw.trim().parse::<u64>()?).filter(|value| *value < (1 << 60))
        };
        if let Some(bound) = bound {
            let usage: u64 = fs::read_to_string(current.join(if v2 {
                "memory.current"
            } else {
                "memory.usage_in_bytes"
            }))?
            .trim()
            .parse()?;
            // Only clean inactive file pages are considered reclaimable. Dirty
            // and writeback pages retain their charge during admission.
            let clean = fs::read_to_string(current.join("memory.stat"))
                .ok()
                .and_then(|text| {
                    let counter = |key: &str| {
                        text.lines().find_map(|line| {
                            let mut fields = line.split_whitespace();
                            (fields.next()? == key)
                                .then(|| fields.next()?.parse::<u64>().ok())
                                .flatten()
                        })
                    };
                    let (inactive, dirty, writeback) = if v2 {
                        ("inactive_file", "file_dirty", "file_writeback")
                    } else {
                        ("total_inactive_file", "total_dirty", "total_writeback")
                    };
                    Some(
                        counter(inactive)?
                            .saturating_sub(counter(dirty)?)
                            .saturating_sub(counter(writeback)?),
                    )
                })
                .unwrap_or(0);
            let headroom = bound.saturating_sub(usage.saturating_sub(clean));
            limit = Some(limit.map_or(bound, |old| old.min(bound)));
            available = Some(available.map_or(headroom, |old| old.min(headroom)));
        }
        if current == mount {
            break;
        }
        ensure!(current.pop(), "invalid memory cgroup ancestry");
    }
    Ok((limit, available))
}
