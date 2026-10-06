//! kl.py's KL-regularized search (piKL) for one root: `Forest`, `grow` (one call a step), `backup` and `root_q`,
//! with numpy's arithmetic reproduced where it shows in the bits: sums are numpy's pairwise sums, the
//! `** kappa` tilt is sqrt at kappa 0.5, arctanh is libm's (Rust's own formula differs), and every
//! accumulation runs in the order numpy's `add.at` and level loops run. Nodes carry the boards and clocks tree.py's
//! Nodes keeps (clock rule "zero" by default: the clocks do not advance). Python sees `allie_fast.KL`, driven by
//! handles (`select` / `update`) or natively (`run`: every call's leaves evaluated through the `Server` on the
//! game's cache and the forest's own slot buffers). Tree reuse (`reroot`) keeps the grandchild's subtree as
//! coverage.rs describes, with the same approximation under either clock rule: the kept nodes' clocks (and the
//! evaluations made under them) descend from the old root's, which assumed the two plies' think times (zero
//! under "zero", predicted under "predicted"), not the real ones.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::clocks::{predicted_seconds, Clocks};
use super::coverage::{CONTEXT, VOCAB};
use super::native::{monotonic, Batch, CacheRef, Slots, NONE};
use crate::chess::{self, Position, HEADER};
use crate::server::{Inner, PyServer};

const WDL: usize = 2413;
/// unexpanded prior mass below this is rounding: the node is fully expanded
const TINY: f64 = 1e-5;

#[link(name = "m")]
extern "C" {
    fn atanh(x: f64) -> f64;
}

/// numpy's pairwise summation of a contiguous float64 array (`a.sum()`): blocks of 8 accumulators up to 128
/// elements, halves above, from 0.
pub fn np_sum(a: &[f64]) -> f64 {
    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            a.iter().fold(0., |s, &x| s + x)
        } else if n <= 128 {
            let mut r: [f64; 8] = a[..8].try_into().unwrap();
            let mut i = 8;
            while i < n - n % 8 {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for &x in &a[i..] {
                res += x;
            }
            res
        } else {
            let n2 = n / 2 - (n / 2) % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    0. + pairwise(a)
}

/// kl.softmax: exp(z - max) / sum.
fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - max).exp()).collect();
    let s = np_sum(&e);
    e.iter().map(|x| x / s).collect()
}

fn squashed(v: f64, squash: f64) -> f64 {
    if squash != 0. { unsafe { atanh(squash * v.clamp(-1., 1.)) } } else { v }
}

/// (1 + (sub - 1) / 16) ** kappa as numpy computes it (sqrt at 0.5, ones at 0).
fn tilt(sub: f64, kappa: f64) -> f64 {
    let x = 1. + (sub - 1.) / 16.;
    if kappa == 0. { 1. } else if kappa == 0.5 { x.sqrt() } else { x.powf(kappa) }
}

/// kl.root_weights: sqrt(p (1 - p)) normalized.
fn root_weights(p: &[f64]) -> Vec<f64> {
    let w: Vec<f64> = p.iter().map(|&p| (p * (1. - p)).sqrt()).collect();
    let s = np_sum(&w).max(1e-300);
    w.iter().map(|x| x / s).collect()
}

struct Edge {
    token: i64,
    p: f64,
    kid: i32,
}

struct Node {
    parent: i32,
    token: i64,
    depth: i32,
    length: i32,
    /// the edge it hangs from
    edge: i64,
    start: usize,
    count: usize,
    done: usize,
    prior: f64,
    value: f64,
    wdl: [f64; 3],
    wdlv: Vec<[f64; 3]>,
    terminal: bool,
    pos: Position,
    board: [u8; 68],
    elapsed: f64,
    /// the native loop's slot (NONE: the root and terminal nodes, never evaluated)
    slot: u32,
}

#[derive(Clone, Copy)]
pub struct Read {
    pub own: f64,
    pub opp: f64,
    pub soft: bool,
    pub kappa: f64,
    pub squash: f64,
}

#[derive(Clone, Copy)]
pub struct Grow {
    pub read: Read,
    pub k: i64,
    pub g: f64,
    pub width: usize,
    pub full: bool,
    pub floor: f64,
}

#[pyclass(name = "KL", module = "allie_fast")]
pub struct Forest {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    views: usize,
    cap: usize,
    calls: i64,
    spent: i64,
    rootw: Vec<f64>,
    pending: Vec<usize>,
    clocks: Vec<Clocks>,
    predicted: bool,
    slots: Vec<Slots>,
    free: Vec<u32>,
    top: u32,
    /// re-root: the evaluated leaves kept
    reused: i64,
    /// the capacity cut a call's quota: the budget may not be reached
    full: bool,
    /// driven by `select` / `update`: its nodes have no slot rows, so `run` refuses it
    by_handles: bool,
}

impl Forest {
    /// roots: each view's root logits; prior: the root's logits for its policy when they are not roots[0]'s
    /// (the game's own under views).
    pub fn build(prefix: &[i64], roots: &[&[f64]], prior: Option<&[f64]>, features: &[[f32; 3]], increment: i64, cap: usize, predicted: bool) -> Result<Forest, &'static str> {
        if roots.is_empty() || roots.iter().chain(prior.iter()).any(|z| z.len() != VOCAB) {
            return Err("root dimensions");
        }
        if prefix.len() < HEADER || prefix.len() >= CONTEXT || features.len() != prefix.len() || cap < 1 {
            return Err("no search context, features not covering the prefix, or no capacity");
        }
        let mut pos = Position::default();
        for &t in &prefix[HEADER..] {
            pos.push_token(t)?;
        }
        let boards = chess::encode(prefix)?;
        let root = Node {
            parent: -1,
            token: 0,
            depth: 0,
            length: prefix.len() as i32,
            edge: -1,
            start: 0,
            count: 0,
            done: 0,
            prior: 1.,
            value: 0.,
            wdl: [0.; 3],
            wdlv: vec![[0.; 3]; roots.len()],
            terminal: false,
            pos,
            board: boards[boards.len() - 68..].try_into().unwrap(),
            elapsed: 0.,
            slot: NONE,
        };
        let mut f = Forest { nodes: vec![root], edges: Vec::new(), views: roots.len(), cap, calls: 0, spent: 0, rootw: Vec::new(), pending: Vec::new(), clocks: vec![Clocks::new(features, increment)], predicted, slots: Vec::new(), free: Vec::new(), top: 1, reused: 0, full: false, by_handles: false };
        f.set(0, prior.unwrap_or(roots[0]), roots);
        f.rootw = root_weights(&f.edges.iter().map(|e| e.p).collect::<Vec<_>>());
        Ok(f)
    }

    /// kl.Forest._set for one node: its edges by falling prior under `policy`, its W/D/L the views' mean.
    fn set(&mut self, i: usize, policy: &[f64], zs: &[&[f64]]) {
        let legal = self.nodes[i].pos.legal_tokens();
        let p = softmax(&legal.iter().map(|&t| policy[t as usize]).collect::<Vec<_>>());
        let mut order: Vec<usize> = (0..legal.len()).collect();
        order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
        let node = &mut self.nodes[i];
        node.start = self.edges.len();
        node.count = legal.len();
        self.edges.extend(order.iter().map(|&j| Edge { token: legal[j], p: p[j], kid: -1 }));
        let mut w = [0.; 3];
        for (v, z) in zs.iter().enumerate() {
            let s = softmax(&z[WDL..WDL + 3]);
            node.wdlv[v] = [s[0], s[1], s[2]];
            for k in 0..3 {
                w[k] += s[k];
            }
        }
        node.wdl = w.map(|x| x / zs.len() as f64);
        node.value = node.wdl[0] - node.wdl[2];
        if self.predicted {
            node.elapsed = predicted_seconds(zs[0]);
        }
    }

    /// kl.backup over the nodes so far: (V, sigma, rest).
    pub fn backup(&self, r: Read) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.nodes.len();
        let par: Vec<i32> = self.nodes.iter().map(|x| x.parent).collect();
        let prior: Vec<f64> = self.nodes.iter().map(|x| x.prior).collect();
        let v: Vec<f64> = self.nodes.iter().map(|x| squashed(x.wdl[0] - x.wdl[2], r.squash)).collect();
        let deepest = self.nodes.iter().map(|x| x.depth).max().unwrap_or(0) as usize;
        let mut levels = vec![Vec::new(); deepest + 1];
        for (i, x) in self.nodes.iter().enumerate() {
            levels[x.depth as usize].push(i);
        }
        let (mut seen, mut sub) = (vec![0.; n], vec![1.; n]);
        for level in levels[1..].iter().rev() {
            for &c in level {
                seen[par[c] as usize] += prior[c];
            }
            for &c in level {
                sub[par[c] as usize] += sub[c];
            }
        }
        let beta: Vec<f64> = (0..n).map(|i| if self.nodes[i].depth % 2 == 0 { r.own } else { r.opp } * tilt(sub[i], r.kappa)).collect();
        let left: Vec<f64> = seen.iter().map(|&s| if s < 1. - TINY { 1. - s } else { 0. }).collect();
        let (mut big_v, mut hi, mut z) = (v.clone(), v.clone(), vec![1.; n]);
        for level in levels[1..].iter().rev() {
            let mut parents: Vec<usize> = level.iter().map(|&c| par[c] as usize).collect();
            for &p in &parents {
                hi[p] = if left[p] > 0. { v[p] } else { f64::NEG_INFINITY };
            }
            for &c in level {
                let p = par[c] as usize;
                hi[p] = hi[p].max(-big_v[c]);
            }
            let e: Vec<f64> = level.iter().map(|&c| prior[c] * (beta[par[c] as usize] * (-big_v[c] - hi[par[c] as usize])).exp()).collect();
            parents.sort_unstable();
            parents.dedup();
            for &p in &parents {
                z[p] = left[p] * (beta[p] * (v[p] - hi[p]).min(0.)).exp();
            }
            let mut num: Vec<f64> = parents.iter().map(|&p| z[p] * v[p]).collect();
            for (&c, &e) in level.iter().zip(&e) {
                z[par[c] as usize] += e;
            }
            for (&c, &e) in level.iter().zip(&e) {
                let p = par[c] as usize;
                num[parents.binary_search(&p).unwrap()] += e * -big_v[c];
            }
            for (&p, &num) in parents.iter().zip(&num) {
                big_v[p] = if r.soft && beta[p] > 0. { hi[p] + z[p].ln() / beta[p] } else { num / z[p] };
            }
        }
        let mut sigma = vec![0.; n];
        for level in &levels[1..] {
            for &c in level {
                let p = par[c] as usize;
                sigma[c] = prior[c] * (beta[p] * (-big_v[c] - hi[p])).exp() / z[p];
            }
        }
        let rest = (0..n).map(|i| (beta[i] * (v[i] - hi[i]).min(0.)).exp() / z[i]).collect();
        (big_v, sigma, rest)
    }

    /// One call of kl.grow: the moves of largest reach (up to the quota) expanded; their new nonterminal
    /// children await `apply`. None when nothing is left to expand (the budget spent, or the forest full).
    pub fn step(&mut self, budget: i64, g: Grow) -> Result<Option<&[usize]>, &'static str> {
        if !self.pending.is_empty() {
            return Err("update pending predictions first");
        }
        let root = &self.nodes[0];
        let need = if root.count > 1 { budget - self.spent } else { 0 };
        let (size, n) = (self.nodes.len(), 1);
        let (_, mut sigma, mut rest) = self.backup(g.read);
        for i in 0..size {
            sigma[i] = (1. - g.floor) * sigma[i] + g.floor * self.nodes[i].prior;
            rest[i] = (1. - g.floor) * rest[i] + g.floor;
        }
        let mut reach = vec![1.; size];
        for i in n..size {
            let x = &self.nodes[i];
            reach[i] = if x.depth == 1 { self.rootw[x.edge as usize] } else { reach[x.parent as usize] * sigma[i] };
        }
        // candidates: the root's unexpanded moves, then each expandable node's next `width` moves
        let mut cand: Vec<(usize, usize, f64)> = Vec::new();
        if need > 0 && (root.length as usize) < CONTEXT {
            cand.extend((0..root.count).filter(|&e| self.edges[root.start + e].kid < 0).map(|e| (0, e, self.rootw[e] + g.full as i64 as f64)));
        }
        for i in n..size {
            let x = &self.nodes[i];
            if x.terminal || x.done >= x.count || x.length as usize >= CONTEXT || need <= 0 {
                continue;
            }
            for j in x.done..(x.done + g.width).min(x.count) {
                cand.push((i, j, reach[i] * rest[i] * self.edges[x.start + j].p));
            }
        }
        cand.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap().then(a.1.cmp(&b.1)));
        let quota = ((g.g * self.spent as f64).ceil() as i64).max(g.k).min(need).max(0) as usize;
        let room = self.cap.saturating_sub(size);
        self.full |= quota.min(cand.len()) > room;
        let take = quota.min(room);
        if take == 0 || cand.is_empty() {
            return Ok(None);
        }
        let pairs: Vec<(usize, usize)> = cand.iter().take(take).map(|c| (c.0, c.1)).collect();
        self.expand(&pairs);
        Ok(Some(&self.pending))
    }

    /// kl.Forest.expand: one child per (node, j) pair, in order; the nonterminal ones pending.
    fn expand(&mut self, pairs: &[(usize, usize)]) {
        for &(i, j) in pairs {
            let c = self.nodes.len();
            let e = self.nodes[i].start + j;
            assert!(j < self.nodes[i].count && self.edges[e].kid < 0 && c < self.cap);
            self.nodes[i].done += 1;
            self.edges[e].kid = c as i32;
            let (token, p) = (self.edges[e].token, self.edges[e].p);
            let parent = &self.nodes[i];
            let mut pos = parent.pos.clone();
            pos.push_token(token).expect("legal move");
            let mut board = parent.board;
            chess::advance(&mut board, token).expect("legal move");
            let (depth, length, elapsed) = (parent.depth + 1, parent.length + 1, parent.elapsed);
            for cl in &mut self.clocks {
                cl.push(i, length as usize - 1, elapsed);
            }
            let out = pos.outcome();
            let terminal = out >= 0.;
            let wdl = if out == 0.5 { [0., 1., 0.] } else { [0., 0., 1.] };
            let slot = if terminal { NONE } else { self.alloc_slot() };
            if !terminal {
                self.spent += 1;
                self.pending.push(c);
            }
            self.nodes.push(Node {
                parent: i as i32,
                token,
                depth,
                length,
                edge: e as i64,
                start: 0,
                count: 0,
                done: 0,
                prior: p,
                value: if terminal { if out == 0.5 { 0. } else { -1. } } else { 0. },
                wdl: if terminal { wdl } else { [0.; 3] },
                wdlv: vec![if terminal { wdl } else { [0.; 3] }; self.views],
                terminal,
                pos,
                board,
                elapsed: 0.,
                slot,
            });
        }
        if !self.pending.is_empty() {
            self.calls += 1;
        }
    }

    fn alloc_slot(&mut self) -> u32 {
        self.free.pop().unwrap_or_else(|| {
            self.top += 1;
            self.top - 1
        })
    }

    /// The node's ancestors' slots below the root, in order.
    fn path_slots(&self, id: usize) -> Vec<u32> {
        let mut p = Vec::new();
        let mut j = self.nodes[id].parent;
        while j > 0 {
            p.push(self.nodes[j as usize].slot);
            j = self.nodes[j as usize].parent;
        }
        p.reverse();
        p
    }

    /// The native loop: kl.grow's calls toward `budget` evaluations, each call's new nodes evaluated through
    /// the server under every view (one request when `merged`, else one per view), until nothing is left to
    /// expand. false: the deadline (time.monotonic seconds) passed before a call with network evaluations (that
    /// call's nodes stay pending, uncounted in `spent`).
    pub fn native(&mut self, server: &Inner, caches: &[CacheRef], deadline: f64, budget: i64, g: Grow, merged: bool) -> Result<bool, String> {
        if caches.len() != self.clocks.len() {
            return Err("one cache per view".into());
        }
        if server.dims[4] != VOCAB {
            return Err("the engine's vocabulary".into());
        }
        if self.slots.is_empty() {
            let cap = (self.top as usize).max(budget.max(0) as usize + 2);
            self.slots = caches.iter().map(|_| Slots::new(server.dims, cap)).collect();
        }
        loop {
            let Some(pending) = self.step(budget, g)?.map(|p| p.to_vec()) else { break };
            if pending.is_empty() {
                continue;
            }
            if monotonic() > deadline {
                self.spent -= pending.len() as i64;
                return Ok(false);
            }
            let zs = self.evaluate(server, caches, &pending, merged)?;
            self.apply(&zs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
        }
        Ok(true)
    }

    /// The pending nodes' logits under every view: f64 rows [n, 2432] per view, `pending`'s order.
    fn evaluate(&mut self, server: &Inner, caches: &[CacheRef], pending: &[usize], merged: bool) -> Result<Vec<Vec<f64>>, String> {
        let (views, n, top) = (caches.len(), pending.len(), self.top as usize);
        for s in &mut self.slots {
            s.ensure(top);
        }
        let mut batches: Vec<Batch> = (0..if merged { 1 } else { views }).map(|_| Batch::new(VOCAB)).collect();
        for v in 0..views {
            for &id in pending {
                let (node, path) = (&self.nodes[id], self.path_slots(id));
                let feats = self.clocks[v].feats[id].map(|x| x as f32);
                batches[if merged { 0 } else { v }].leaf(&caches[v], &self.slots[v], &path, node.slot, node.token, feats, &node.board);
            }
        }
        let outs = batches.into_iter().map(|b| b.submit(server)).collect::<Result<Vec<_>, _>>()?;
        Ok((0..views)
            .map(|v| {
                let (o, base) = if merged { (&outs[0], v * n) } else { (&outs[v], 0) };
                o[base * VOCAB..(base + n) * VOCAB].iter().map(|&x| x as f64).collect()
            })
            .collect())
    }

    /// Re-root at the grandchild reached by `moves` (the two plies played since the search), if it was
    /// evaluated: its subtree kept (ids remapped in order, depths less two), the root set from the game's new
    /// logits (`roots` per view, `prior` as `build`) with its kept children by token (their priors the new
    /// ones), every view's clocks restarted from `features` (the game's, two plies longer) for the root, the
    /// evaluations spent those of the kept leaves, the capacity `cap` nodes (as `build`'s, for the next budget).
    /// The kept nodes' old ids, or None when nothing can be kept.
    pub fn rebase(&mut self, moves: &[i64], roots: &[&[f64]], prior: Option<&[f64]>, features: &[Vec<[f32; 3]>], cap: usize) -> Result<Option<Vec<usize>>, &'static str> {
        if !self.pending.is_empty() {
            return Err("update pending predictions first");
        }
        if moves.len() != 2 || roots.len() != self.views || roots.iter().chain(prior.iter()).any(|z| z.len() != VOCAB) {
            return Err("two moves, one root per view [2432]");
        }
        let len = self.nodes[0].length as usize + 2;
        if features.len() != self.clocks.len() || features.iter().any(|f| f.len() != len) {
            return Err("features must cover the new prefix under every view");
        }
        let find = |f: &Forest, i: usize, t: i64| {
            let x = &f.nodes[i];
            (x.start..x.start + x.count).find(|&e| f.edges[e].token == t && f.edges[e].kid >= 0).map(|e| f.edges[e].kid as usize)
        };
        let Some(c1) = find(self, 0, moves[0]) else { return Ok(None) };
        let Some(g) = find(self, c1, moves[1]) else { return Ok(None) };
        if self.nodes[g].terminal || self.nodes[g].count == 0 {
            return Ok(None);
        }
        let n = self.nodes.len();
        let (mut map, mut kept) = (vec![-1i32; n], Vec::new());
        for i in 0..n {
            let p = self.nodes[i].parent;
            if i == g || (p >= 0 && map[p as usize] >= 0) {
                map[i] = kept.len() as i32;
                kept.push(i);
            }
        }
        let old_start: Vec<usize> = self.nodes.iter().map(|x| x.start).collect();
        let mut old: Vec<Option<Node>> = std::mem::take(&mut self.nodes).into_iter().map(Some).collect();
        let old_edges = std::mem::take(&mut self.edges);
        let gn = old[g].take().unwrap();
        self.nodes.push(Node { parent: -1, token: 0, depth: 0, length: gn.length, edge: -1, start: 0, count: 0, done: 0, prior: 1., value: 0., wdl: [0.; 3], wdlv: vec![[0.; 3]; self.views], terminal: false, pos: gn.pos, board: gn.board, elapsed: 0., slot: NONE });
        self.set(0, prior.unwrap_or(roots[0]), roots);
        let count = self.nodes[0].count;
        for e in &old_edges[gn.start..gn.start + gn.count] {
            if e.kid >= 0 {
                let j = (0..count).find(|&j| self.edges[j].token == e.token).ok_or("legal moves changed")?;
                self.edges[j].kid = map[e.kid as usize];
            }
        }
        self.nodes[0].done = self.edges[..count].iter().filter(|e| e.kid >= 0).count();
        let mut used = vec![false; self.top as usize];
        for &i in &kept[1..] {
            let mut x = old[i].take().unwrap();
            let start = self.edges.len();
            self.edges.extend(old_edges[x.start..x.start + x.count].iter().map(|e| Edge { token: e.token, p: e.p, kid: if e.kid >= 0 { map[e.kid as usize] } else { -1 } }));
            let p = x.parent as usize;
            x.edge = if p == g {
                let j = (0..count).find(|&j| self.edges[j].kid == map[i]).unwrap();
                x.prior = self.edges[j].p;
                j as i64
            } else {
                (self.nodes[map[p] as usize].start + (x.edge as usize - old_start[p])) as i64
            };
            x.parent = map[p];
            x.depth -= 2;
            x.start = start;
            if !x.terminal {
                used[x.slot as usize] = true;
            }
            self.nodes.push(x);
        }
        self.rootw = root_weights(&self.edges[..count].iter().map(|e| e.p).collect::<Vec<_>>());
        self.free = (1..self.top).rev().filter(|&s| !used[s as usize]).collect();
        for (c, f) in self.clocks.iter_mut().zip(features) {
            c.rebase(f, &kept);
        }
        self.spent = self.nodes[1..].iter().filter(|x| !x.terminal).count() as i64;
        self.reused = self.spent;
        self.calls = 0;
        self.cap = cap.max(self.nodes.len());
        self.full = false;
        Ok(Some(kept))
    }

    /// The pending nodes' logits under every view ([n, 2432] each, pending order).
    pub fn apply(&mut self, zs: &[&[f64]]) -> Result<(), &'static str> {
        if zs.len() != self.views || zs.iter().any(|z| z.len() != self.pending.len() * VOCAB) {
            return Err("leaf dimensions");
        }
        for (k, &c) in std::mem::take(&mut self.pending).iter().enumerate() {
            let rows: Vec<&[f64]> = zs.iter().map(|z| &z[k * VOCAB..(k + 1) * VOCAB]).collect();
            self.set(c, rows[0], &rows);
        }
        Ok(())
    }

    pub fn handles(&self) -> Vec<[i64; 4]> {
        self.pending.iter().map(|&c| [c as i64, self.nodes[c].parent as i64, self.nodes[c].token, self.nodes[c].length as i64]).collect()
    }

    /// kl.root_q with tree.KL's fill: (moves by falling prior, prior, Q for the mover; unexpanded moves at the
    /// root's own value, squashed as the values are).
    pub fn root_values(&self, r: Read) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
        let v = self.backup(r).0;
        let root = &self.nodes[0];
        let own = squashed(root.value, r.squash);
        let edges = &self.edges[root.start..root.start + root.count];
        (edges.iter().map(|e| e.token).collect(), edges.iter().map(|e| e.p).collect(), edges.iter().map(|e| if e.kid >= 0 { -v[e.kid as usize] } else { own }).collect())
    }
}

fn invalid(e: &'static str) -> PyErr {
    PyValueError::new_err(e)
}

fn rows3(a: &PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 3]>> {
    let a = a.as_array();
    if a.shape()[1] != 3 {
        return Err(invalid("features must be [n, 3]"));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[pymethods]
impl Forest {
    /// prefix: the root's tokens; roots: the root's logits under each view [2432]; prior: its logits for the
    /// policy when not roots[0]'s (the game's own under views); features, increment: the game's clocks (as
    /// Coverage); cap: the most nodes; clock_rule: "zero" (clocks do not advance) or "predicted".
    #[new]
    #[pyo3(signature = (prefix, roots, prior, features, increment, cap, clock_rule="zero"))]
    fn new(prefix: Vec<i64>, roots: Vec<PyReadonlyArray1<f64>>, prior: Option<PyReadonlyArray1<f64>>, features: PyReadonlyArray2<f32>, increment: i64, cap: usize, clock_rule: &str) -> PyResult<Self> {
        let predicted = match clock_rule {
            "predicted" => true,
            "zero" => false,
            _ => return Err(invalid("unknown clock rule")),
        };
        let roots: Vec<Vec<f64>> = roots.iter().map(|z| z.as_array().iter().copied().collect()).collect();
        let prior: Option<Vec<f64>> = prior.map(|z| z.as_array().iter().copied().collect());
        let feats = rows3(&features)?;
        Forest::build(&prefix, &roots.iter().map(Vec::as_slice).collect::<Vec<_>>(), prior.as_deref(), &feats, increment, cap, predicted).map_err(invalid)
    }

    fn add_view(&mut self, features: PyReadonlyArray2<f32>, increment: i64) -> PyResult<usize> {
        if self.nodes.len() != 1 {
            return Err(PyRuntimeError::new_err("views join before the search"));
        }
        let feats = rows3(&features)?;
        if feats.len() != self.nodes[0].length as usize {
            return Err(invalid("features must cover the prefix"));
        }
        self.clocks.push(Clocks::new(&feats, increment));
        Ok(self.clocks.len() - 1)
    }

    /// One kl.grow call toward `budget` network evaluations (kl.grow's keywords): the new nonterminal nodes'
    /// handles [n, 4] (id, parent, token, prefix length), or None when nothing is left to expand. root (the
    /// output tilt of the root's weights) is not ported.
    #[pyo3(signature = (budget, own=0., opp=0., soft=false, kappa=0., k=8, g=0.125, width=4, root=0., full=false, floor=0., squash=0.))]
    #[allow(clippy::too_many_arguments)]
    fn select<'py>(&mut self, py: Python<'py>, budget: i64, own: f64, opp: f64, soft: bool, kappa: f64, k: i64, g: f64, width: usize, root: f64, full: bool, floor: f64, squash: f64) -> PyResult<Option<Bound<'py, PyArray2<i64>>>> {
        if root != 0. {
            return Err(invalid("the root tilt is not ported"));
        }
        if !self.slots.is_empty() {
            return Err(PyRuntimeError::new_err("a forest run natively continues with run()"));
        }
        self.by_handles = true;
        let grow = Grow { read: Read { own, opp, soft, kappa, squash }, k, g, width, full, floor };
        if self.step(budget, grow).map_err(PyRuntimeError::new_err)?.is_none() {
            return Ok(None);
        }
        let rows: Vec<i64> = self.handles().into_iter().flatten().collect();
        Ok(Some(PyArray1::from_vec_bound(py, rows).reshape([self.pending.len(), 4])?))
    }

    /// The whole search natively through `server` toward `budget` evaluations (kl.grow's keywords as `select`):
    /// caches, one per view, as (k, v, e pointers, capacity, rows) of the game's (its views') cache, kept alive
    /// and untouched by the caller until this returns; deadline: time.monotonic() seconds past which the search
    /// stops before its next call, giving False; merged: all views' items of a call in one request. The GIL is
    /// released.
    #[pyo3(signature = (server, caches, budget, deadline=f64::INFINITY, merged=true, own=0., opp=0., soft=false, kappa=0., k=8, g=0.125, width=4, root=0., full=false, floor=0., squash=0.))]
    #[allow(clippy::too_many_arguments)]
    fn run(&mut self, py: Python<'_>, server: PyRef<'_, PyServer>, caches: Vec<(usize, usize, usize, usize, usize)>, budget: i64, deadline: f64, merged: bool, own: f64, opp: f64, soft: bool, kappa: f64, k: i64, g: f64, width: usize, root: f64, full: bool, floor: f64, squash: f64) -> PyResult<bool> {
        if root != 0. {
            return Err(invalid("the root tilt is not ported"));
        }
        if self.by_handles {
            return Err(PyRuntimeError::new_err("a forest driven by select() has no slot rows to run natively"));
        }
        let grow = Grow { read: Read { own, opp, soft, kappa, squash }, k, g, width, full, floor };
        let inner = server.inner.clone();
        let caches: Vec<CacheRef> = caches.into_iter().map(CacheRef::from).collect();
        py.allow_threads(move || self.native(&inner, &caches, deadline, budget, grow, merged)).map_err(PyRuntimeError::new_err)
    }

    /// Tree reuse: re-root at the grandchild reached by `moves` (two move tokens) with the game's new logits
    /// (roots per view, prior as the constructor's) and clock features (one [n, 3] per view, n the new prefix
    /// length), at most `cap` nodes from now on (the constructor's for the next budget); the kept nodes' old ids,
    /// or None when the grandchild was not evaluated (start fresh).
    #[pyo3(signature = (moves, roots, prior, features, cap))]
    fn reroot(&mut self, moves: Vec<i64>, roots: Vec<PyReadonlyArray1<f64>>, prior: Option<PyReadonlyArray1<f64>>, features: Vec<PyReadonlyArray2<f32>>, cap: usize) -> PyResult<Option<Vec<usize>>> {
        let roots: Vec<Vec<f64>> = roots.iter().map(|z| z.as_array().iter().copied().collect()).collect();
        let prior: Option<Vec<f64>> = prior.map(|z| z.as_array().iter().copied().collect());
        let feats = features.iter().map(rows3).collect::<PyResult<Vec<_>>>()?;
        self.rebase(&moves, &roots.iter().map(Vec::as_slice).collect::<Vec<_>>(), prior.as_deref(), &feats, cap).map_err(invalid)
    }

    /// Evaluated leaves kept by the last re-root.
    #[getter]
    fn reused(&self) -> i64 {
        self.reused
    }

    /// The node capacity cut a call short since the forest was built or re-rooted: the budget may be unmet.
    #[getter]
    fn full(&self) -> bool {
        self.full
    }

    /// The pending nodes' logits under each view, [n, 2432] float64 each.
    fn update(&mut self, zs: Vec<PyReadonlyArray2<f64>>) -> PyResult<()> {
        let zs: Vec<Vec<f64>> = zs.iter().map(|z| z.as_array().iter().copied().collect()).collect();
        self.apply(&zs.iter().map(Vec::as_slice).collect::<Vec<_>>()).map_err(invalid)
    }

    /// (moves by falling prior, prior, Q for the mover) read by kl.backup's keywords, unexpanded moves at the
    /// root's own value.
    #[pyo3(signature = (own=0., opp=0., soft=false, kappa=0., squash=0.))]
    fn root_q<'py>(&self, py: Python<'py>, own: f64, opp: f64, soft: bool, kappa: f64, squash: f64) -> PyResult<(Vec<i64>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        if !self.pending.is_empty() {
            return Err(PyRuntimeError::new_err("pending update"));
        }
        let (moves, prior, q) = self.root_values(Read { own, opp, soft, kappa, squash });
        Ok((moves, PyArray1::from_vec_bound(py, prior), PyArray1::from_vec_bound(py, q)))
    }

    /// kl.backup over the nodes so far: (V, sigma, rest).
    #[pyo3(signature = (own=0., opp=0., soft=false, kappa=0., squash=0.))]
    fn values<'py>(&self, py: Python<'py>, own: f64, opp: f64, soft: bool, kappa: f64, squash: f64) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
        let (v, s, r) = self.backup(Read { own, opp, soft, kappa, squash });
        (PyArray1::from_vec_bound(py, v), PyArray1::from_vec_bound(py, s), PyArray1::from_vec_bound(py, r))
    }

    /// Network evaluations so far.
    #[getter]
    fn spent(&self) -> i64 {
        self.spent
    }

    #[getter]
    fn calls(&self) -> i64 {
        self.calls
    }

    #[getter]
    fn size(&self) -> usize {
        self.nodes.len()
    }

    fn boards<'py>(&self, py: Python<'py>, ids: Vec<usize>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let mut out = Vec::with_capacity(ids.len() * 68);
        for id in &ids {
            out.extend_from_slice(&self.nodes.get(*id).ok_or_else(|| invalid("node id"))?.board);
        }
        PyArray1::from_vec_bound(py, out).reshape([ids.len(), 68])
    }

    #[pyo3(signature = (ids, view=0))]
    fn feats<'py>(&self, py: Python<'py>, ids: Vec<usize>, view: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let clocks = self.clocks.get(view).ok_or_else(|| invalid("view"))?;
        let mut out = Vec::with_capacity(ids.len() * 3);
        for id in &ids {
            out.extend(clocks.feats.get(*id).ok_or_else(|| invalid("node id"))?.map(|x| x as f32));
        }
        PyArray1::from_vec_bound(py, out).reshape([ids.len(), 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum() {
        // 0.1 added a thousand times: numpy's blocks give a different last bit from a running sum
        let a = vec![0.1; 1000];
        let seq = a.iter().fold(0., |s, x| s + x);
        let np = np_sum(&a);
        assert!((np - 100.).abs() < 1e-12 && np != seq);
        assert_eq!(np_sum(&[1., 2., 3.]), 6.);
        assert_eq!(np_sum(&[]), 0.);
        let b: Vec<f64> = (0..37).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut r = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
        for i in (8..32).step_by(8) {
            for j in 0..8 {
                r[j] += b[i + j];
            }
        }
        let mut want = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for x in &b[32..] {
            want += x;
        }
        assert_eq!(np_sum(&b), want);
    }

    fn token(uci: &str) -> i64 {
        crate::chess::MOVE_START + crate::chess::vocab().uci.iter().position(|m| m == uci).unwrap() as i64
    }

    /// Deterministic logits from a hash of the token sequence.
    fn fake(prefix: &[i64]) -> Vec<f64> {
        let mut s = prefix.iter().fold(0x9E3779B97F4A7C15u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100000001B3));
        (0..VOCAB)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                6. * ((s >> 11) as f64 / (1u64 << 53) as f64) - 3.
            })
            .collect()
    }

    fn line(f: &Forest, root: &[i64], mut c: usize) -> Vec<i64> {
        let mut t = Vec::new();
        while c != 0 {
            t.push(f.nodes[c].token);
            c = f.nodes[c].parent as usize;
        }
        [root, &t.into_iter().rev().collect::<Vec<_>>()].concat()
    }

    fn grow(f: &mut Forest, root: &[i64], budget: i64) {
        let g = Grow { read: Read { own: 5., opp: 5., soft: true, kappa: 0.5, squash: 0. }, k: 8, g: 0.125, width: 4, full: false, floor: 0. };
        while let Some(p) = f.step(budget, g).unwrap().map(|p| p.to_vec()) {
            let z: Vec<f64> = p.iter().flat_map(|&c| fake(&line(f, root, c))).collect();
            f.apply(&[&z]).unwrap();
        }
    }

    #[test]
    fn reroot_keeps_the_subtree_values() {
        let p: Vec<i64> = [vec![2348, 199, 12, 1, 5, 0, 0, 1, 5, 0, 0], ["e2e4", "e7e5", "g1f3"].iter().map(|m| token(m)).collect()].concat();
        let feats: Vec<[f32; 3]> = (0..p.len()).map(|i| if i < 11 { [-1.; 3] } else { [180. - i as f32, 181. - i as f32, -1.] }).collect();
        let mut f = Forest::build(&p, &[&fake(&p)], None, &feats, 2, 400, false).unwrap();
        grow(&mut f, &p, 96);
        assert_eq!(f.spent, 96);
        let n = f.nodes.len();
        let mut size = vec![1usize; n];
        for i in (1..n).rev() {
            size[f.nodes[i].parent as usize] += size[i];
        }
        let g = (1..n).filter(|&i| f.nodes[i].depth == 2 && f.nodes[i].count > 0).max_by_key(|&i| size[i]).unwrap();
        assert!(size[g] > 3, "a grandchild with a subtree");
        let kept: Vec<usize> = (0..n).filter(|&i| { let mut j = i as i32; while j > 0 && j != g as i32 { j = f.nodes[j as usize].parent; } j == g as i32 }).collect();
        let read = Read { own: 12., opp: 12., soft: true, kappa: 0., squash: 0.95 };
        let (v, sigma, _) = f.backup(read);
        let old: Vec<([u8; 68], i32, [f64; 3], u32)> = kept.iter().map(|&i| (f.nodes[i].board, f.nodes[i].depth, f.clocks[0].feats[i], f.nodes[i].slot)).collect();
        let moves = [f.nodes[f.nodes[g].parent as usize].token, f.nodes[g].token];
        let q = [p.clone(), moves.to_vec()].concat();
        let longer: Vec<[f32; 3]> = feats.iter().copied().chain([[170., 160., 3.], [150., 160., 8.]]).collect();
        assert_eq!(f.rebase(&moves, &[&fake(&q)], None, &[longer.clone()], 400), Ok(Some(kept.clone())));
        assert_eq!((f.nodes.len(), f.nodes[0].length as usize, f.calls), (kept.len(), q.len(), 0));
        let (v2, sigma2, _) = f.backup(read);
        for (k, &i) in kept.iter().enumerate().skip(1) {
            assert_eq!(v2[k], v[i], "V of kept node {i}");
            assert_eq!((f.nodes[k].board, f.nodes[k].depth + 2, f.clocks[0].feats[k], f.nodes[k].slot), old[k]);
            if f.nodes[k].depth >= 2 {
                assert_eq!(sigma2[k], sigma[i]);
            }
            let p = f.nodes[k].parent as usize;
            assert!(p < k && f.edges[f.nodes[k].edge as usize].kid == k as i32 && f.edges[f.nodes[k].edge as usize].token == f.nodes[k].token);
            assert!(f.nodes[p].start <= f.nodes[k].edge as usize && (f.nodes[k].edge as usize) < f.nodes[p].start + f.nodes[p].count);
        }
        let live = f.nodes[1..].iter().filter(|x| !x.terminal).count() as i64;
        assert_eq!((f.spent, f.reused, f.clocks[0].feats[0]), (live, live, [150., 160., 8.]));
        let legal = f.nodes[0].pos.legal_tokens();
        assert_eq!(f.nodes[0].count, legal.len());
        grow(&mut f, &q, 96);
        assert_eq!(f.spent, 96);
        let (_, _, qv) = f.root_values(read);
        assert!(qv.iter().all(|x| x.is_finite()));
        // a reply the forest never expanded: nothing kept
        let c1 = f.edges[f.nodes[0].start].kid;
        if c1 > 0 {
            let x = &f.nodes[c1 as usize];
            if let Some(e) = (x.start..x.start + x.count).find(|&e| f.edges[e].kid < 0) {
                let t = [f.nodes[c1 as usize].token, f.edges[e].token];
                let lon: Vec<[f32; 3]> = longer.iter().copied().chain([[1.; 3], [1.; 3]]).collect();
                assert_eq!(f.rebase(&t, &[&fake(&q)], None, &[lon], 400), Ok(None));
            }
        }
    }

    #[test]
    fn reroot_grows_the_capacity() {
        // a forest built for 16 leaves (288 nodes), re-rooted for 512: the capacity follows the new budget;
        // re-rooted at the old capacity, the cut is flagged
        let p: Vec<i64> = [vec![2348, 199, 12, 1, 5, 0, 0, 1, 5, 0, 0], ["d2d4", "d7d5"].iter().map(|m| token(m)).collect()].concat();
        let feats: Vec<[f32; 3]> = vec![[-1.; 3]; p.len()];
        let longer: Vec<[f32; 3]> = vec![[-1.; 3]; p.len() + 2];
        for (cap, short) in [(2 * 512 + 256, false), (2 * 16 + 256, true)] {
            let mut f = Forest::build(&p, &[&fake(&p)], None, &feats, -1, 2 * 16 + 256, false).unwrap();
            grow(&mut f, &p, 16);
            let g = (1..f.nodes.len()).find(|&i| f.nodes[i].depth == 2 && !f.nodes[i].terminal).unwrap();
            let moves = [f.nodes[f.nodes[g].parent as usize].token, f.nodes[g].token];
            let q = [p.clone(), moves.to_vec()].concat();
            assert!(f.rebase(&moves, &[&fake(&q)], None, &[longer.clone()], cap).unwrap().is_some());
            assert!(!f.full);
            grow(&mut f, &q, 512);
            assert_eq!((f.spent == 512, f.full), (!short, short), "capacity {cap}");
        }
    }

    #[test]
    fn tilts() {
        assert_eq!(tilt(17., 0.5), 2f64.sqrt());
        assert_eq!(tilt(5., 0.), 1.);
        assert_eq!(squashed(2., 0.5), unsafe { atanh(0.5) });
        assert_eq!(softmax(&[0., 0.]), [0.5, 0.5]);
        assert_eq!(root_weights(&[1.]), [0.]);
    }
}
