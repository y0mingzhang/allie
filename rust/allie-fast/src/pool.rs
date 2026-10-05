//! The C++ `Pool`: workers pinned to CPUs stay alive between steps, spin briefly for work, then sleep on a
//! condvar until woken; a barrier whose arrivals are counted per group of threads sharing a last-level
//! cache, then once per group.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::missing_safety_doc)]

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::*};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[repr(align(64))]
pub struct Padded<T>(pub T);

#[repr(align(64))]
struct Group {
    count: AtomicI32,
    size: i32,
}

type Job<'a> = dyn Fn(usize) + Sync + 'a;

struct Inner {
    n: usize,
    cpus: Option<Vec<i32>>,
    spin: f64,
    group: Vec<usize>,
    groups: Vec<Group>,
    epoch: Padded<AtomicU32>,
    left: Padded<AtomicI32>,
    top: Padded<AtomicI32>,
    gen: Padded<AtomicU32>,
    pokes: AtomicU32,
    stop: AtomicBool,
    mu: Mutex<()>,
    cv: Condvar,
    job: Padded<std::cell::UnsafeCell<Option<*const Job<'static>>>>,
}

unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct Pool {
    inner: Arc<Inner>,
    threads: Vec<std::thread::JoinHandle<()>>,
    pid: libc::pid_t,
}

/// The CPUs this process may run on (none known: every CPU).
pub fn affinity() -> Vec<i32> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return (0..libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as i32).collect();
        }
        (0..libc::CPU_SETSIZE).filter(|&c| libc::CPU_ISSET(c as usize, &set)).collect()
    }
}

fn bind(cpu: i32) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu as usize, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

impl Inner {
    fn bind(&self, t: usize) {
        if let Some(c) = &self.cpus {
            bind(c[t]);
        }
    }

    fn work(&self, t: usize) {
        self.bind(t);
        let mut seen = 0u32;
        loop {
            let mut t0 = Instant::now();
            let mut k = 1u32;
            while self.epoch.0.load(Acquire) == seen {
                std::hint::spin_loop();
                if k.is_multiple_of(1024) && t0.elapsed().as_secs_f64() > self.spin {
                    let p = self.pokes.load(SeqCst);
                    let guard = self.mu.lock().unwrap();
                    let _g = self.cv.wait_while(guard, |_| self.epoch.0.load(Acquire) == seen && self.pokes.load(SeqCst) == p).unwrap();
                    t0 = Instant::now(); // poked: spin again, work is coming
                }
                k = k.wrapping_add(1);
            }
            seen = self.epoch.0.load(Acquire);
            if self.stop.load(Acquire) {
                return;
            }
            let job = unsafe { (*self.job.0.get()).unwrap() };
            unsafe { (*job)(t) };
            self.left.0.fetch_sub(1, Release);
        }
    }
}

impl Pool {
    /// cpus: thread t's CPU (None: unpinned); groups: thread t's group, numbered from 0 (None: one group).
    /// Threads never outnumber the CPUs of the process's affinity mask.
    pub fn new(n: usize, cpus: Option<Vec<i32>>, groups: Option<Vec<i32>>, spin: f64) -> Pool {
        let n = n.clamp(1, affinity().len().max(1));
        let cpus = cpus.filter(|c| c.len() >= n).map(|c| c[..n].to_vec());
        let group: Vec<usize> = match groups {
            Some(g) if g.len() >= n => g[..n].iter().map(|&x| x.max(0) as usize).collect(),
            _ => vec![0; n],
        };
        // the clamp may drop every member of a group: renumber the groups that remain (an empty group would
        // never arrive at the barrier)
        let mut ids = group.clone();
        ids.sort_unstable();
        ids.dedup();
        let group: Vec<usize> = group.iter().map(|g| ids.binary_search(g).unwrap()).collect();
        let ngroups = group.iter().max().unwrap() + 1;
        let groups: Vec<Group> = (0..ngroups).map(|g| Group { count: AtomicI32::new(0), size: group.iter().filter(|&&x| x == g).count() as i32 }).collect();
        let inner = Arc::new(Inner {
            n,
            cpus,
            spin,
            group,
            groups,
            epoch: Padded(AtomicU32::new(0)),
            left: Padded(AtomicI32::new(0)),
            top: Padded(AtomicI32::new(0)),
            gen: Padded(AtomicU32::new(0)),
            pokes: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            mu: Mutex::new(()),
            cv: Condvar::new(),
            job: Padded(std::cell::UnsafeCell::new(None)),
        });
        let threads = (1..n)
            .map(|t| {
                let me = inner.clone();
                std::thread::Builder::new().name(format!("allie-{t}")).spawn(move || me.work(t)).expect("spawn")
            })
            .collect();
        Pool { inner, threads, pid: unsafe { libc::getpid() } }
    }

    pub fn n(&self) -> usize {
        self.inner.n
    }

    /// f(t) on thread 0 (the caller, pinned for the call) and on every worker; returns when all are done.
    pub fn run(&self, f: &Job<'_>) {
        let p = &*self.inner;
        let old = p.cpus.as_ref().and_then(|_| unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            (libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0).then_some(set)
        });
        if old.is_some() {
            p.bind(0);
        }
        // the workers only run the job between the epoch bump and `left` reaching 0, within this call
        let job: *const Job<'static> = unsafe { std::mem::transmute(f as *const Job<'_>) };
        unsafe { *p.job.0.get() = Some(job) };
        p.left.0.store(p.n as i32 - 1, Relaxed);
        {
            let _g = p.mu.lock().unwrap();
            p.epoch.0.fetch_add(1, Release);
        }
        p.cv.notify_all();
        f(0);
        while p.left.0.load(Acquire) > 0 {
            std::hint::spin_loop();
        }
        if let Some(set) = old {
            unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
        }
    }

    /// Sleeping workers spin again, ready for a step.
    pub fn wake(&self) {
        {
            let _g = self.inner.mu.lock().unwrap();
            self.inner.pokes.fetch_add(1, SeqCst);
        }
        self.inner.cv.notify_all();
    }

    /// The last of a group arrives for it at the top.
    #[inline]
    pub fn barrier(&self, t: usize) {
        let p = &*self.inner;
        if p.n == 1 {
            return;
        }
        let g = p.gen.0.load(Acquire);
        let c = &p.groups[p.group[t]];
        if c.count.fetch_add(1, AcqRel) == c.size - 1 {
            c.count.store(0, Relaxed);
            if p.top.0.fetch_add(1, AcqRel) == p.groups.len() as i32 - 1 {
                p.top.0.store(0, Relaxed);
                p.gen.0.store(g.wrapping_add(1), Release);
                return;
            }
        }
        while p.gen.0.load(Acquire) == g {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        if unsafe { libc::getpid() } != self.pid {
            // a fork's copy: the workers exist only in the parent, and its mutex may be held there
            self.threads.drain(..).for_each(std::mem::forget);
            return;
        }
        {
            let _g = self.inner.mu.lock().unwrap();
            self.inner.stop.store(true, Release);
            self.inner.epoch.0.fetch_add(1, Release);
        }
        self.inner.cv.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn clamped_groups_are_renumbered() {
        // three threads asked for with groups [0, 2, 1], two granted: the groups in use are 0 and 2 -> 0 and 1
        let pool = Pool::new(2, None, Some(vec![0, 2, 1]), 0.001);
        assert!(pool.n() <= 2);
        let (tx, rx) = std::sync::mpsc::channel();
        let p = std::sync::Arc::new(pool);
        let q = p.clone();
        std::thread::spawn(move || {
            for _ in 0..20 {
                q.run(&|t| {
                    q.barrier(t);
                    q.barrier(t);
                });
            }
            tx.send(()).unwrap();
        });
        rx.recv_timeout(std::time::Duration::from_secs(10)).expect("the barrier must not hang on a renumbered group");
    }

    #[test]
    fn runs_and_barriers() {
        let pool = Pool::new(4, None, Some(vec![0, 0, 1, 1]), 0.001);
        let n = pool.n();
        let hits = AtomicUsize::new(0);
        let phase = AtomicUsize::new(0);
        for _ in 0..50 {
            pool.run(&|t| {
                hits.fetch_add(1, SeqCst);
                pool.barrier(t);
                if t == 0 {
                    phase.fetch_add(1, SeqCst);
                }
                pool.barrier(t);
                assert_eq!(phase.load(SeqCst) % 1, 0);
            });
        }
        assert_eq!(hits.load(SeqCst), 50 * n);
        assert_eq!(phase.load(SeqCst), 50);
        pool.wake();
    }
}
