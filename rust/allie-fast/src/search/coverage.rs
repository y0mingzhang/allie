//! tree.cpp's `GrowForest` for one root, the coverage search's tree: the root's branches pulled in turn by
//! sqrt(p (1 - p)) / (1 + pulls), PUCT below, expansion from the logits, terminal and depth-limited backups,
//! `compact` for value.cpp's backup (`reduce`), `grow` for a larger budget; with tree.py `Nodes`' per-node
//! board states and clocks. Python sees it as `allie_fast.Coverage`, driven as the C++ one (`select` gives
//! the pending nodes' handles (id, parent, token, prefix length), `update` takes their logits) or natively:
//! `run(server, caches, deadline)` evaluates every call's leaves itself through the `Server` as path items on
//! the game's cache and the tree's own slot buffers, with no Python per leaf.
//!
//! Values are f64 wherever the C++ and numpy use double; logits pass through f32 as the C++ bindings'
//! forcecast rounds them. Node ids are creation order from the root's 0, so they index the Python side's slots;
//! the native loop's slots are the tree's own (`Node::slot`, equal to the id until a re-root frees some).
//!
//! Tree reuse (`reroot`): after the two plies since the search (the bot's move, the reply) the grandchild's
//! subtree becomes the next search's start: its nodes, visits, bootstrap values and slot rows are kept (slot
//! positions are absolute: the two plies' rows are now the cache's), the pull schedule is rebuilt with the
//! kept visits as pulls already made, and the root is re-expanded from the game's real logits. The kept
//! leaves were evaluated under the clocks the search predicted for those two plies while the cache rows now
//! carry the real clocks: an approximation inherent to reuse (the calibration refits for it); KL's "zero"
//! clock rule has no such gap.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::backup::{Backup, Compact};
use super::clocks::{predicted_seconds, Clocks};
use super::native::{mixed, monotonic, Batch, CacheRef, Slots, NONE};
use crate::chess::{self, Position, HEADER, MOVE_START};
use crate::server::{Inner, PyServer};

pub const CONTEXT: usize = 1025;
pub const VOCAB: usize = 2432;
const MAX_SEARCH_DEPTH: i32 = 100;
const WDL: usize = 2413;

#[derive(Clone, Copy)]
struct Edge {
    token: i64,
    child: i32,
    prior: f64,
}

struct Node {
    parent: i32,
    token: i64,
    depth: i32,
    born: i32,
    n: i32,
    /// the sum of backed-up values for the player who moved into the node
    w: f64,
    prior: f64,
    /// loss minus win for the side to move, from the W/D/L head
    boot: f64,
    children: Vec<Edge>,
    pos: Position,
    terminal: f64,
    board: [u8; 68],
    /// the think time predicted for the node's mover (0 under the "zero" clock rule)
    elapsed: f64,
    /// the native loop's slot for the node's keys, values and embedding (NONE: the root)
    slot: u32,
}

struct Branch {
    edge: usize,
    next: usize,
    pulls: Vec<i32>,
}

#[derive(Default)]
struct Stats {
    evaluated: i64,
    terminal_visits: i64,
    depth_visits: i64,
    prefix_tokens: i64,
    requests: i64,
    max_depth: i32,
    /// re-root: the evaluated leaves kept and the pulls they stand for
    reused: i64,
    kept_pulls: i32,
}

/// Pull k (from + 1..=budget) goes to the branch of largest weight / (1 + its pulls so far), the lowest on
/// ties; `counts`: pulls already made per branch (a re-rooted tree's kept visits).
fn schedule(weights: &[f64], counts: &[i32], from: i32, budget: i32) -> Vec<Vec<i32>> {
    let mut pulls = vec![Vec::new(); weights.len()];
    let mut counts = counts.to_vec();
    for pull in from + 1..=budget {
        let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
        for (j, &w) in weights.iter().enumerate() {
            let u = w / (1 + counts[j]) as f64;
            if u > best_u {
                best_u = u;
                best = j;
            }
        }
        pulls[best].push(pull);
        counts[best] += 1;
    }
    pulls
}

fn rounded<'a>(z: impl Iterator<Item = &'a f64>) -> Vec<f64> {
    z.map(|&x| x as f32 as f64).collect()
}

fn invalid(e: &'static str) -> PyErr {
    PyValueError::new_err(e)
}

#[pyclass(module = "allie_fast")]
pub struct Coverage {
    nodes: Vec<Node>,
    branches: Vec<Branch>,
    pending: Vec<usize>,
    clocks: Vec<Clocks>,
    root_len: usize,
    budget: i32,
    cpuct: f64,
    predicted: bool,
    remaining: i32,
    rounds: i32,
    evals: i64,
    stats: Stats,
    /// the native loop's slot buffers, one per view, allocated at its first run
    slots: Vec<Slots>,
    /// slots freed by a re-root, and the slots ever allocated
    free: Vec<u32>,
    top: u32,
}

impl Coverage {
    pub fn build(prefix: &[i64], root: &[f64], features: &[[f32; 3]], increment: i64, budget: i32, cpuct: f64, predicted: bool) -> Result<Coverage, &'static str> {
        if root.len() != VOCAB {
            return Err("root dimensions");
        }
        if prefix.len() < HEADER || prefix.len() >= CONTEXT || budget < 0 {
            return Err("no search context or negative budget");
        }
        if !cpuct.is_finite() || cpuct < 0. {
            return Err("coverage cpuct must be finite and nonnegative");
        }
        if features.len() != prefix.len() {
            return Err("features must cover the prefix");
        }
        let mut pos = Position::default();
        for &t in &prefix[HEADER..] {
            pos.push_token(t)?;
        }
        if pos.outcome() >= 0. {
            return Err("terminal root");
        }
        let boards = chess::encode(prefix)?;
        let board: [u8; 68] = boards[boards.len() - 68..].try_into().unwrap();
        let z = rounded(root.iter());
        let root = Node { parent: -1, token: -1, depth: 0, born: 0, n: 0, w: 0., prior: 0., boot: 0., children: Vec::new(), pos, terminal: -1., board, elapsed: 0., slot: NONE };
        let mut cov = Coverage {
            nodes: vec![root],
            branches: Vec::new(),
            pending: Vec::new(),
            clocks: vec![Clocks::new(features, increment)],
            root_len: prefix.len(),
            budget,
            cpuct,
            predicted,
            remaining: budget,
            rounds: 0,
            evals: 0,
            stats: Stats::default(),
            slots: Vec::new(),
            free: Vec::new(),
            top: 1,
        };
        cov.expand(0, &z)?;
        cov.nodes[0].elapsed = if predicted { predicted_seconds(&z) } else { 0. };
        let counts = vec![0; cov.nodes[0].children.len()];
        cov.branches = schedule(&cov.weights(), &counts, 0, budget).into_iter().enumerate().map(|(edge, pulls)| Branch { edge, next: 0, pulls }).collect();
        Ok(cov)
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

    fn weights(&self) -> Vec<f64> {
        self.nodes[0].children.iter().map(|e| (e.prior * (1. - e.prior)).sqrt()).collect()
    }

    pub fn is_done(&self) -> bool {
        self.remaining == 0 && self.pending.is_empty()
    }

    /// The node's children from its legal moves (a softmax of their logits in double) and its bootstrap
    /// value; z is one row of logits already rounded through f32.
    fn expand(&mut self, id: usize, z: &[f64]) -> Result<f64, &'static str> {
        let legal = self.nodes[id].pos.legal_tokens();
        if legal.is_empty() {
            return Err("terminal expansion");
        }
        let max = legal.iter().map(|&t| z[t as usize]).fold(f64::NEG_INFINITY, f64::max);
        let (mut sum, mut p) = (0., Vec::with_capacity(legal.len()));
        for &t in &legal {
            let e = (z[t as usize] - max).exp();
            p.push(e);
            sum += e;
        }
        let node = &mut self.nodes[id];
        node.children = legal.iter().zip(&p).map(|(&token, &e)| Edge { token, child: -1, prior: e / sum }).collect();
        let (w, d, l) = (z[WDL], z[WDL + 1], z[WDL + 2]);
        let max = w.max(d).max(l);
        let (win, draw, loss) = ((w - max).exp(), (d - max).exp(), (l - max).exp());
        node.boot = loss / (win + draw + loss) - win / (win + draw + loss);
        Ok(node.boot)
    }

    fn backup(&mut self, id: usize, mut value: f64) {
        let mut id = id as i32;
        while id >= 0 {
            let node = &mut self.nodes[id as usize];
            node.n += 1;
            node.w += value;
            value = -value;
            id = node.parent;
        }
    }

    /// The child along an edge, created (born at this pull, its board and clocks advanced) if new.
    fn child(&mut self, parent: usize, edge: usize, birth: i32) -> usize {
        if self.nodes[parent].children[edge].child < 0 {
            let id = self.nodes.len();
            let p = &self.nodes[parent];
            let Edge { token, prior, .. } = p.children[edge];
            let mut pos = p.pos.clone();
            pos.push_token(token).expect("legal move");
            let mut board = p.board;
            chess::advance(&mut board, token).expect("legal move");
            let (depth, length, elapsed) = (p.depth + 1, self.root_len + p.depth as usize, p.elapsed);
            for c in &mut self.clocks {
                c.push(parent, length, elapsed);
            }
            let terminal = pos.outcome();
            let slot = self.alloc_slot();
            self.nodes.push(Node { parent: parent as i32, token, depth, born: birth, n: 0, w: 0., prior, boot: 0., children: Vec::new(), pos, terminal, board, elapsed: 0., slot });
            self.nodes[parent].children[edge].child = id as i32;
        }
        self.nodes[parent].children[edge].child as usize
    }

    /// PUCT from `id` down to an unexpanded node or the depth limit; the first child wins ties.
    fn descend(&mut self, mut id: usize, birth: i32, depth_limit: i32) -> usize {
        while !self.nodes[id].children.is_empty() && self.nodes[id].depth < depth_limit {
            let node = &self.nodes[id];
            let factor = ((((node.n as f64 + 19652.) + 1.) / 19652.).ln() + self.cpuct) * (node.n.max(1) as f64).sqrt();
            let (mut best, mut best_u) = (0, f64::NEG_INFINITY);
            for (j, e) in node.children.iter().enumerate() {
                let (n, q) = match e.child {
                    c if c >= 0 && self.nodes[c as usize].n > 0 => {
                        let c = &self.nodes[c as usize];
                        (c.n, c.w / c.n as f64)
                    }
                    _ => (0, 0.),
                };
                let u = q + factor * e.prior / (1 + n) as f64;
                if u > best_u {
                    best_u = u;
                    best = j;
                }
            }
            id = self.child(id, best, birth);
        }
        id
    }

    /// One pull of every branch with pulls left: the pending nodes to evaluate, in branch order.
    pub fn next(&mut self) -> Result<&[usize], &'static str> {
        if !self.pending.is_empty() {
            return Err("update pending predictions first");
        }
        if self.is_done() {
            return Err("already finished");
        }
        let depth_limit = MAX_SEARCH_DEPTH.min((CONTEXT - self.root_len) as i32);
        for b in 0..self.branches.len() {
            let branch = &mut self.branches[b];
            if branch.next == branch.pulls.len() {
                continue;
            }
            let (edge, birth) = (branch.edge, branch.pulls[branch.next]);
            branch.next += 1;
            let id = self.child(0, edge, birth);
            let id = self.descend(id, birth, depth_limit);
            let node = &self.nodes[id];
            self.stats.max_depth = self.stats.max_depth.max(node.depth);
            if node.terminal >= 0. {
                let value = if node.terminal == 0.5 { 0. } else { 1. }; // a decisive end is mate: the player who moved in won
                self.backup(id, value);
                self.stats.terminal_visits += 1;
            } else if !node.children.is_empty() && node.depth >= depth_limit {
                let value = node.boot;
                self.backup(id, value);
                self.stats.depth_visits += 1;
            } else {
                self.stats.prefix_tokens += (self.root_len + node.depth as usize) as i64;
                self.evals += 1;
                self.pending.push(id);
            }
            self.remaining -= 1;
        }
        self.rounds += 1;
        if !self.pending.is_empty() {
            self.stats.requests += 1;
        }
        self.stats.evaluated += self.pending.len() as i64;
        Ok(&self.pending)
    }

    /// The pending nodes' logits (f64 rows of 2432, in `next`'s order): expansions first, then the backups in
    /// the same order, as the C++ update_fast.
    pub fn apply(&mut self, z: &[f64]) -> Result<(), &'static str> {
        if z.len() != self.pending.len() * VOCAB {
            return Err("leaf dimensions");
        }
        let mut values = Vec::with_capacity(self.pending.len());
        for (i, &id) in self.pending.clone().iter().enumerate() {
            let row = &z[i * VOCAB..(i + 1) * VOCAB];
            values.push(self.expand(id, row)?);
            if self.predicted {
                self.nodes[id].elapsed = predicted_seconds(row);
            }
        }
        for (&id, &v) in std::mem::take(&mut self.pending).iter().zip(&values) {
            self.backup(id, v);
        }
        Ok(())
    }

    pub fn handles(&self) -> Vec<[i64; 4]> {
        self.pending.iter().map(|&id| [id as i64, self.nodes[id].parent as i64, self.nodes[id].token, (self.root_len + self.nodes[id].depth as usize) as i64]).collect()
    }

    pub fn compact_arrays(&self) -> Compact {
        let n = &self.nodes;
        Compact {
            parent: n.iter().map(|x| x.parent).collect(),
            mv: n.iter().map(|x| (x.token - MOVE_START) as i32).collect(),
            depth: n.iter().map(|x| x.depth).collect(),
            born: n.iter().map(|x| x.born).collect(),
            degree: n.iter().map(|x| x.children.len() as i32).collect(),
            prior: n.iter().map(|x| x.prior).collect(),
            boot: n.iter().map(|x| x.boot).collect(),
            mass: n.iter().map(|x| x.children.iter().fold(0., |s, e| s + e.prior)).collect(),
            terminal: n.iter().map(|x| x.terminal).collect(),
            roots: vec![0],
        }
    }

    /// Continue the finished search to a larger budget: the schedule's first `budget` pulls are unchanged.
    pub fn extend(&mut self, budget: i32) -> Result<(), &'static str> {
        if self.stats.kept_pulls > 0 {
            return Err("cannot grow a re-rooted tree");
        }
        if !self.is_done() {
            return Err("finish current phase before grow");
        }
        if budget < self.budget {
            return Err("cannot shrink");
        }
        if self.branches.iter().any(|b| b.next != b.pulls.len()) {
            return Err("unfinished branch");
        }
        let counts = vec![0; self.branches.len()];
        let pulls = schedule(&self.weights(), &counts, 0, budget);
        for (branch, pulls) in self.branches.iter_mut().zip(pulls) {
            if !pulls.iter().filter(|&&p| p <= self.budget).eq(branch.pulls.iter()) {
                return Err("quota prefix changed");
            }
            self.remaining += (pulls.len() - branch.pulls.len()) as i32;
            branch.pulls = pulls;
        }
        self.budget = budget;
        Ok(())
    }

    /// The native loop: every call's pending leaves evaluated through the server (all views' items in one
    /// request when `merged`, else one request per view), their logits applied, until done. false: the
    /// deadline (time.monotonic seconds) passed before a call.
    pub fn native(&mut self, server: &Inner, caches: &[CacheRef], deadline: f64, merged: bool) -> Result<bool, String> {
        if caches.len() != self.clocks.len() {
            return Err("one cache per view".into());
        }
        if server.dims[4] != VOCAB {
            return Err("the engine's vocabulary".into());
        }
        if self.slots.is_empty() {
            let cap = (self.top as usize).max(self.budget as usize + 2);
            self.slots = caches.iter().map(|_| Slots::new(server.dims, cap)).collect();
        }
        while !self.is_done() {
            if monotonic() > deadline {
                return Ok(false);
            }
            let pending = self.next()?.to_vec();
            if pending.is_empty() {
                continue;
            }
            let z = self.evaluate(server, caches, &pending, merged)?;
            self.apply(&z)?;
        }
        Ok(true)
    }

    /// The pending nodes' logits under every view, mixed (tree.py `mixed`) and rounded through f32 as
    /// `update` takes them: f64 rows in `pending`'s order.
    fn evaluate(&mut self, server: &Inner, caches: &[CacheRef], pending: &[usize], merged: bool) -> Result<Vec<f64>, String> {
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
        let row = |v: usize, k: usize| -> &[f32] {
            let (o, i) = if merged { (&outs[0], v * n + k) } else { (&outs[v], k) };
            &o[i * VOCAB..(i + 1) * VOCAB]
        };
        let mut z = Vec::with_capacity(n * VOCAB);
        for k in 0..n {
            if views == 1 {
                z.extend(row(0, k).iter().map(|&x| x as f64));
            } else {
                let rows: Vec<&[f32]> = (0..views).map(|v| row(v, k)).collect();
                z.extend(rounded(mixed(&rows).iter()));
            }
        }
        Ok(z)
    }

    /// Re-root at the grandchild reached by `moves` (the two plies played since the search), if it was
    /// evaluated: its subtree kept (ids remapped in order, depths less two, born reset), the root re-expanded
    /// from the game's logits `root` (rounded through f32) with the kept children's visits, every view's
    /// clocks restarted from `features` (the game's, now two plies longer) for the root, the pull schedule
    /// rebuilt toward `budget` after the kept visits. Returns the kept nodes' old ids (new id i was old id
    /// kept[i]), or None when nothing can be kept: the caller starts fresh.
    pub fn rebase(&mut self, moves: &[i64], root: &[f64], features: &[Vec<[f32; 3]>], budget: i32) -> Result<Option<Vec<usize>>, &'static str> {
        if !self.is_done() {
            return Err("unfinished search");
        }
        if root.len() != VOCAB || moves.len() != 2 || budget < 0 {
            return Err("two moves, root logits [2432], a nonnegative budget");
        }
        if features.len() != self.clocks.len() || features.iter().any(|f| f.len() != self.root_len + 2) {
            return Err("features must cover the new prefix under every view");
        }
        let find = |nodes: &[Node], id: usize, t: i64| nodes[id].children.iter().find(|e| e.token == t && e.child >= 0).map(|e| e.child as usize);
        let Some(c1) = find(&self.nodes, 0, moves[0]) else { return Ok(None) };
        let Some(g) = find(&self.nodes, c1, moves[1]) else { return Ok(None) };
        if self.nodes[g].children.is_empty() || self.nodes[g].terminal >= 0. {
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
        let mut used = vec![false; self.top as usize];
        let old = std::mem::take(&mut self.nodes);
        for (i, mut x) in old.into_iter().enumerate() {
            if map[i] < 0 {
                continue;
            }
            x.parent = if i == g { -1 } else { map[x.parent as usize] };
            x.depth -= 2;
            x.born = 0;
            for e in &mut x.children {
                if e.child >= 0 {
                    e.child = map[e.child as usize];
                }
            }
            if i == g {
                x.slot = NONE;
                x.token = -1;
            } else {
                used[x.slot as usize] = true;
            }
            self.nodes.push(x);
        }
        self.free = (1..self.top).rev().filter(|&s| !used[s as usize]).collect();
        let saved: Vec<(i64, i32)> = self.nodes[0].children.iter().map(|e| (e.token, e.child)).collect();
        self.expand(0, root)?;
        for e in &mut self.nodes[0].children {
            e.child = saved.iter().find(|s| s.0 == e.token).map_or(-1, |s| s.1);
        }
        let edges: Vec<Edge> = self.nodes[0].children.clone();
        let mut counts = Vec::with_capacity(edges.len());
        for e in &edges {
            counts.push(if e.child >= 0 {
                self.nodes[e.child as usize].prior = e.prior;
                self.nodes[e.child as usize].n
            } else {
                0
            });
        }
        let kept_pulls: i32 = counts.iter().sum();
        let r = &mut self.nodes[0];
        r.n = kept_pulls;
        r.w = 0.;
        r.elapsed = if self.predicted { predicted_seconds(root) } else { 0. };
        for (c, f) in self.clocks.iter_mut().zip(features) {
            c.rebase(f, &kept);
        }
        self.root_len += 2;
        self.budget = budget;
        self.remaining = (budget - kept_pulls).max(0);
        self.branches = schedule(&self.weights(), &counts, kept_pulls, budget).into_iter().enumerate().map(|(edge, pulls)| Branch { edge, next: 0, pulls }).collect();
        self.rounds = 0;
        self.evals = 0;
        self.stats = Stats { reused: self.nodes[1..].iter().filter(|x| !x.children.is_empty()).count() as i64, kept_pulls, ..Stats::default() };
        Ok(Some(kept))
    }
}

#[pymethods]
impl Coverage {
    /// prefix: the root's tokens (11 header tokens, then moves); root: its logits [2432]; features: the
    /// game's clock features at every prefix token [len(prefix), 3]; increment: seconds, -1 without a clock;
    /// clock_rule: "predicted" (nodes' clocks advance by their parents' predicted think times) or "zero".
    #[new]
    #[pyo3(signature = (prefix, root, features, increment, budget, cpuct, clock_rule="predicted"))]
    fn new(prefix: Vec<i64>, root: PyReadonlyArray1<f64>, features: PyReadonlyArray2<f32>, increment: i64, budget: i32, cpuct: f64, clock_rule: &str) -> PyResult<Self> {
        let predicted = match clock_rule {
            "predicted" => true,
            "zero" => false,
            _ => return Err(invalid("unknown clock rule")),
        };
        let feats = rows3(&features)?;
        let root = root.as_array();
        Coverage::build(&prefix, &root.iter().copied().collect::<Vec<_>>(), &feats, increment, budget, cpuct, predicted).map_err(invalid)
    }

    /// Another view's clocks (its features and increment), before the search starts; `feats(ids, view)`.
    fn add_view(&mut self, features: PyReadonlyArray2<f32>, increment: i64) -> PyResult<usize> {
        if self.nodes.len() != 1 {
            return Err(PyRuntimeError::new_err("views join before the search"));
        }
        let feats = rows3(&features)?;
        if feats.len() != self.root_len {
            return Err(invalid("features must cover the prefix"));
        }
        self.clocks.push(Clocks::new(&feats, increment));
        Ok(self.clocks.len() - 1)
    }

    /// The next pending nodes' handles [n, 4]: id, parent, move token, prefix length.
    fn select<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.next().map_err(PyRuntimeError::new_err)?;
        let rows: Vec<i64> = self.handles().into_iter().flatten().collect();
        PyArray1::from_vec_bound(py, rows).reshape([self.pending.len(), 4])
    }

    /// The pending nodes' logits [n, 2432] (float64; rounded through float32 as the C++ bindings do).
    fn update(&mut self, z: PyReadonlyArray2<f64>) -> PyResult<()> {
        let z = z.as_array();
        if z.shape() != [self.pending.len(), VOCAB] {
            return Err(invalid("leaf dimensions"));
        }
        self.apply(&rounded(z.iter())).map_err(PyRuntimeError::new_err)
    }

    fn grow(&mut self, budget: i32) -> PyResult<()> {
        self.extend(budget).map_err(invalid)
    }

    /// The whole search natively through `server`: caches, one per view, as (k, v, e pointers, capacity,
    /// rows) of the game's (its views') cache, kept alive and untouched by the caller until this returns;
    /// deadline: time.monotonic() seconds past which the search stops before its next call, giving False;
    /// merged: all views' items of a call in one request (else one per view). The GIL is released.
    #[pyo3(signature = (server, caches, deadline=f64::INFINITY, merged=true))]
    fn run(&mut self, py: Python<'_>, server: PyRef<'_, PyServer>, caches: Vec<(usize, usize, usize, usize, usize)>, deadline: f64, merged: bool) -> PyResult<bool> {
        let inner = server.inner.clone();
        let caches: Vec<CacheRef> = caches.into_iter().map(CacheRef::from).collect();
        py.allow_threads(move || self.native(&inner, &caches, deadline, merged)).map_err(PyRuntimeError::new_err)
    }

    /// Tree reuse: re-root at the grandchild reached by `moves` (two move tokens) for a search of `budget`
    /// from the game's new logits `root` [2432] and clock features `features` (one [n, 3] per view, n the
    /// new prefix length); the kept nodes' old ids, or None when the grandchild was not evaluated (start
    /// fresh). The native loop's slots stay valid.
    fn reroot(&mut self, moves: Vec<i64>, root: PyReadonlyArray1<f64>, features: Vec<PyReadonlyArray2<f32>>, budget: i32) -> PyResult<Option<Vec<usize>>> {
        let feats = features.iter().map(rows3).collect::<PyResult<Vec<_>>>()?;
        let root = rounded(root.as_array().iter());
        self.rebase(&moves, &root, &feats, budget).map_err(invalid)
    }

    /// Every node's visit count, by id.
    fn visits(&self) -> Vec<i32> {
        self.nodes.iter().map(|x| x.n).collect()
    }

    #[getter]
    fn done(&self) -> bool {
        self.is_done()
    }

    /// Network evaluations per root.
    #[getter]
    fn evals(&self) -> Vec<i64> {
        vec![self.evals]
    }

    /// The nodes' 68-byte board states [n, 68].
    fn boards<'py>(&self, py: Python<'py>, ids: Vec<usize>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let mut out = Vec::with_capacity(ids.len() * 68);
        for id in &ids {
            out.extend_from_slice(&self.nodes.get(*id).ok_or_else(|| invalid("node id"))?.board);
        }
        PyArray1::from_vec_bound(py, out).reshape([ids.len(), 68])
    }

    /// The nodes' clock features [n, 3] under a view (float32, as the model reads them).
    #[pyo3(signature = (ids, view=0))]
    fn feats<'py>(&self, py: Python<'py>, ids: Vec<usize>, view: usize) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let clocks = self.clocks.get(view).ok_or_else(|| invalid("view"))?;
        let mut out = Vec::with_capacity(ids.len() * 3);
        for id in &ids {
            out.extend(clocks.feats.get(*id).ok_or_else(|| invalid("node id"))?.map(|x| x as f32));
        }
        PyArray1::from_vec_bound(py, out).reshape([ids.len(), 3])
    }

    /// tree.cpp's compact(): the arrays value.cpp reads.
    fn compact<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.pending.is_empty() {
            return Err(PyRuntimeError::new_err("pending update"));
        }
        let c = self.compact_arrays();
        let d = PyDict::new_bound(py);
        d.set_item("parent", PyArray1::from_vec_bound(py, c.parent))?;
        d.set_item("move", PyArray1::from_vec_bound(py, c.mv))?;
        d.set_item("depth", PyArray1::from_vec_bound(py, c.depth))?;
        d.set_item("born", PyArray1::from_vec_bound(py, c.born))?;
        d.set_item("degree", PyArray1::from_vec_bound(py, c.degree))?;
        d.set_item("prior", PyArray1::from_vec_bound(py, c.prior))?;
        d.set_item("boot", PyArray1::from_vec_bound(py, c.boot))?;
        d.set_item("mass", PyArray1::from_vec_bound(py, c.mass))?;
        d.set_item("terminal", PyArray1::from_vec_bound(py, c.terminal))?;
        d.set_item("roots", PyArray1::from_vec_bound(py, c.roots))?;
        Ok(d)
    }

    /// value.cpp's ScaledCount Backup(compact, budget, count_scale).reduce(log_tau, exponent, ids): ids
    /// [1, k] move ids (token - 378) -> [3, 1, k] (the value for the mover and its gradients).
    fn reduce<'py>(&self, py: Python<'py>, log_tau: f64, exponent: f64, count_scale: f64, budget: i32, ids: PyReadonlyArray2<i64>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if !self.pending.is_empty() {
            return Err(PyRuntimeError::new_err("pending update"));
        }
        let ids = ids.as_array();
        let (r, k) = (ids.shape()[0], ids.shape()[1]);
        if r != 1 {
            return Err(invalid("root IDs shape"));
        }
        let backup = Backup::new(&self.compact_arrays(), budget, count_scale).map_err(invalid)?;
        let out = backup.reduce(log_tau, exponent, &ids.iter().copied().collect::<Vec<_>>(), k).map_err(invalid)?;
        PyArray1::from_vec_bound(py, out).reshape([3, 1, k])
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        d.set_item("simulations", self.budget as i64)?;
        d.set_item("evaluated_leaves", self.stats.evaluated)?;
        d.set_item("terminal_visits", self.stats.terminal_visits)?;
        d.set_item("useful_prefix_tokens", self.stats.prefix_tokens)?;
        d.set_item("requests", self.stats.requests)?;
        d.set_item("max_depth", self.stats.max_depth)?;
        if self.stats.depth_visits != 0 {
            d.set_item("depth_limited_visits", self.stats.depth_visits)?;
        }
        d.set_item("nodes", self.nodes.len())?;
        d.set_item("rounds", self.rounds)?;
        d.set_item("reused_leaves", self.stats.reused)?;
        d.set_item("kept_pulls", self.stats.kept_pulls)?;
        Ok(d)
    }
}

fn rows3(a: &PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 3]>> {
    let a = a.as_array();
    if a.shape()[1] != 3 {
        return Err(invalid("features must be [n, 3]"));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chess::vocab;

    fn token(uci: &str) -> i64 {
        MOVE_START + vocab().uci.iter().position(|m| m == uci).unwrap() as i64
    }

    fn prefix(moves: &[&str]) -> Vec<i64> {
        [vec![2348, 199, 12, 1, 5, 0, 0, 1, 5, 0, 0], moves.iter().map(|m| token(m)).collect()].concat()
    }

    /// Deterministic logits: a hash of the node's path tokens spreads the move logits, the W/D/L and time heads.
    fn fake(prefix: &[i64]) -> Vec<f64> {
        let mut s = prefix.iter().fold(0x9E3779B97F4A7C15u64, |h, &t| (h ^ t as u64).wrapping_mul(0x100000001B3));
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..VOCAB).map(|_| 6. * next() - 3.).collect()
    }

    fn run(cov: &mut Coverage, root: &[i64]) -> usize {
        let mut calls = 0;
        while !cov.is_done() {
            let pending = cov.next().unwrap().to_vec();
            let mut z = Vec::new();
            for &id in &pending {
                let mut p = root.to_vec();
                let (mut i, mut path) = (id, Vec::new());
                while i != 0 {
                    path.push(cov.nodes[i].token);
                    i = cov.nodes[i].parent as usize;
                }
                p.extend(path.iter().rev());
                z.extend(fake(&p));
            }
            cov.apply(&z).unwrap();
            calls += 1;
        }
        calls
    }

    #[test]
    fn pull_schedule() {
        assert_eq!(schedule(&[0.5, 0.5, 0.1], &[0; 3], 0, 5), [vec![1, 3, 5], vec![2, 4], vec![]]);
        assert_eq!(schedule(&[0., 0.], &[0; 2], 0, 2), [vec![1, 2], vec![]]);
        assert!(schedule(&[0.3], &[0], 0, 0)[0].is_empty());
        // pulls already made: the second branch is behind, so it takes the next pulls until even
        assert_eq!(schedule(&[0.5, 0.5], &[3, 0], 3, 7), [vec![7], vec![4, 5, 6]]);
    }

    #[test]
    fn mate_is_a_win_for_the_mover() {
        let p = prefix(&["f2f3", "e7e5", "g2g4"]);
        let mut z = vec![0.; VOCAB];
        z[token("d8h4") as usize] = 30.;
        let feats = vec![[-1f32; 3]; p.len()];
        let mut cov = Coverage::build(&p, &z, &feats, -1, 1, 2.5, true).unwrap();
        assert!(cov.next().unwrap().is_empty()); // the mate backs up without an evaluation
        assert!(cov.is_done() && cov.evals == 0 && cov.stats.terminal_visits == 1);
        let (mate, root) = (&cov.nodes[1], &cov.nodes[0]);
        assert_eq!((mate.token, mate.n, mate.w, mate.terminal), (token("d8h4"), 1, 1., 0.));
        assert_eq!((root.n, root.w), (1, -1.));
        let c = cov.compact_arrays();
        assert_eq!((c.mv[0], c.born[1], c.degree[1], c.terminal[1]), (-379, 1, 0, 0.));
        let q = Backup::new(&c, 1, 24.).unwrap().reduce(0., 0., &[token("d8h4") - MOVE_START, token("a7a6") - MOVE_START], 2).unwrap();
        assert_eq!(q[0], 1.);
        assert_eq!(q[1], -c.boot[0]);
    }

    #[test]
    fn search_and_grow() {
        let p = prefix(&["e2e4", "e7e5", "g1f3"]);
        let feats: Vec<[f32; 3]> = (0..p.len()).map(|i| if i < 11 { [-1.; 3] } else { [180. - i as f32, 181. - i as f32, -1.] }).collect();
        let mut a = Coverage::build(&p, &fake(&p), &feats, 2, 24, 2.5, true).unwrap();
        let calls = run(&mut a, &p);
        assert!(calls > 1 && a.nodes.len() <= 25 && a.nodes[0].n == 24);
        assert_eq!(a.evals + a.stats.terminal_visits + a.stats.depth_visits, 24);
        let legal = a.nodes[0].pos.legal_tokens();
        let ids: Vec<i64> = legal.iter().map(|t| t - MOVE_START).collect();
        let total: i32 = a.nodes[0].children.iter().map(|e| if e.child < 0 { 0 } else { a.nodes[e.child as usize].n }).sum();
        assert_eq!(total, 24);
        assert!(a.handles().is_empty());
        // clocks: a depth-1 node's mover is the opponent, with the root's opponent seconds, the root's clock less
        // its predicted think time plus the increment, and the opponent's previous think time off the prefix
        let child = a.nodes[0].children.iter().find(|e| e.child >= 0).unwrap().child as usize;
        let spent = a.nodes[0].elapsed.round_ties_even().max(0.).min(feats[p.len() - 1][0] as f64);
        assert_eq!(a.clocks[0].other[0], 2.);
        assert_eq!(a.clocks[0].feats[child], [feats[p.len() - 1][1] as f64, feats[p.len() - 1][0] as f64 - spent + 2., 2.]);
        assert_eq!(a.nodes[child].board, { let mut b = a.nodes[0].board; chess::advance(&mut b, a.nodes[child].token).unwrap(); b });
        // grown from 24 to 64, the tree equals a fresh 64's under the same evaluator (ids differ, values not)
        let mut b = Coverage::build(&p, &fake(&p), &feats, 2, 64, 2.5, true).unwrap();
        run(&mut b, &p);
        a.extend(64).unwrap();
        assert!(!a.is_done());
        run(&mut a, &p);
        let (ca, cb) = (a.compact_arrays(), b.compact_arrays());
        assert_eq!(ca.parent.len(), cb.parent.len());
        let q = |c: &Compact| Backup::new(c, 64, 24.).unwrap().reduce(-1.2, 0.3, &ids, ids.len()).unwrap();
        assert_eq!(q(&ca), q(&cb));
        assert_eq!(a.extend(60), Err("cannot shrink"));
        assert_eq!(b.next(), Err("already finished"));
    }

    #[test]
    fn reroots_at_the_played_grandchild() {
        let p = prefix(&["e2e4", "e7e5", "g1f3"]);
        let feats: Vec<[f32; 3]> = (0..p.len()).map(|i| if i < 11 { [-1.; 3] } else { [180. - i as f32, 181. - i as f32, -1.] }).collect();
        let mut a = Coverage::build(&p, &fake(&p), &feats, 2, 48, 2.5, true).unwrap();
        run(&mut a, &p);
        // the most visited root child, then its most visited expanded child
        let kid = |a: &Coverage, id: usize| a.nodes[id].children.iter().filter(|e| e.child >= 0 && !a.nodes[e.child as usize].children.is_empty()).max_by_key(|e| a.nodes[e.child as usize].n).map(|e| e.child as usize).unwrap();
        let (c1, g) = (kid(&a, 0), kid(&a, kid(&a, 0)));
        let moves = [a.nodes[c1].token, a.nodes[g].token];
        let kept: Vec<usize> = (0..a.nodes.len()).filter(|&i| { let mut j = i as i32; while j >= 0 && j != g as i32 { j = a.nodes[j as usize].parent; } j == g as i32 }).collect();
        let (old_n, old_boot, old_slot): (Vec<i32>, Vec<f64>, Vec<u32>) = (kept.iter().map(|&i| a.nodes[i].n).collect(), kept.iter().map(|&i| a.nodes[i].boot).collect(), kept.iter().map(|&i| a.nodes[i].slot).collect());
        let q = [p.clone(), moves.to_vec()].concat();
        let longer: Vec<[f32; 3]> = feats.iter().copied().chain([[170., 160., 3.], [150., 160., 8.]]).collect();
        assert_eq!(a.rebase(&moves, &fake(&q), &[longer.clone()], 48), Ok(Some(kept.clone())));
        assert_eq!((a.nodes.len(), a.root_len, a.nodes[0].parent, a.nodes[0].slot, a.nodes[0].depth), (kept.len(), p.len() + 2, -1, NONE, 0));
        assert!(a.nodes[1..].iter().all(|x| x.parent >= 0 && x.born == 0 && x.depth >= 1));
        assert_eq!(a.nodes.iter().map(|x| x.n).collect::<Vec<_>>()[1..], old_n[1..]);
        assert_eq!(a.nodes.iter().map(|x| x.boot).collect::<Vec<_>>()[1..], old_boot[1..]);
        assert_eq!(a.nodes.iter().map(|x| x.slot).collect::<Vec<_>>()[1..], old_slot[1..]);
        let kept_pulls: i32 = a.nodes[0].children.iter().map(|e| if e.child >= 0 { a.nodes[e.child as usize].n } else { 0 }).sum();
        assert_eq!((a.nodes[0].n, a.stats.kept_pulls, a.remaining), (kept_pulls, kept_pulls, 48 - kept_pulls));
        assert_eq!(a.stats.reused, kept.len() as i64 - 1 - a.nodes[1..].iter().filter(|x| x.children.is_empty()).count() as i64);
        assert_eq!(a.clocks[0].feats[0], [150., 160., 8.]);
        assert!(a.free.contains(&old_slot[0]) && a.free.iter().all(|&s| !old_slot[1..].contains(&s)) && a.free.len() + old_slot.len() - 1 == a.top as usize - 1);
        assert_eq!(a.extend(64), Err("cannot grow a re-rooted tree"));
        // the kept visits are pulls already made: the schedule continues from them to the budget
        let pulls: Vec<i32> = a.branches.iter().flat_map(|b| b.pulls.iter().copied()).collect();
        assert_eq!(pulls.len() as i32, 48 - kept_pulls);
        assert!(pulls.iter().all(|&k| k > kept_pulls && k <= 48));
        run(&mut a, &q);
        assert!(a.is_done() && a.nodes[0].n == 48);
        let c = a.compact_arrays();
        assert!(Backup::new(&c, 48, 24.).unwrap().reduce(0., 0., &[a.nodes[0].children[0].token - MOVE_START], 1).unwrap()[0].is_finite());
        // a reply missing from the tree: nothing kept; a feature length off: an error
        let miss = a.nodes[0].children.iter().find(|e| e.child < 0).map(|e| e.token);
        if let Some(t) = miss {
            let mut b = Coverage::build(&q, &fake(&q), &longer, 2, 8, 2.5, true).unwrap();
            run(&mut b, &q);
            let lon: Vec<[f32; 3]> = longer.iter().copied().chain([[1.; 3], [1.; 3]]).collect();
            assert_eq!(b.rebase(&[t, t], &fake(&q), &[lon], 8), Ok(None));
            assert_eq!(b.rebase(&[t, t], &fake(&q), &[longer.clone()], 8), Err("features must cover the new prefix under every view"));
        }
    }

    #[test]
    fn rejects_bad_roots() {
        let p = prefix(&["f2f3", "e7e5", "g2g4", "d8h4"]);
        let feats = vec![[-1f32; 3]; p.len()];
        assert_eq!(Coverage::build(&p, &vec![0.; VOCAB], &feats, -1, 8, 2.5, true).err(), Some("terminal root"));
        let p = prefix(&["e2e4"]);
        assert_eq!(Coverage::build(&p, &vec![0.; 10], &vec![[-1f32; 3]; p.len()], -1, 8, 2.5, true).err(), Some("root dimensions"));
        assert_eq!(Coverage::build(&p, &vec![0.; VOCAB], &vec![[-1f32; 3]; p.len()], -1, -1, 2.5, true).err(), Some("no search context or negative budget"));
        let mut cov = Coverage::build(&p, &vec![0.; VOCAB], &vec![[-1f32; 3]; p.len()], -1, 0, 2.5, false).unwrap();
        assert!(cov.is_done() && cov.nodes[0].elapsed == 0.);
        assert_eq!(cov.next(), Err("already finished"));
    }
}
