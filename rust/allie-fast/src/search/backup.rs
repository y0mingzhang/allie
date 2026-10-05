//! value.cpp's `ScaledCount`: the soft value backup over a compact tree at temperature
//! exp(a + b log1p((subtree count - 1) / scale)), and its `reduce` to the root's moves: the value for the
//! mover and its gradients in a and b. Children are walked newest first (the C++ linked lists), so every sum
//! adds in the same order.

/// tree.cpp `compact()`: one entry a node, in id order (a parent before its children).
pub struct Compact {
    pub parent: Vec<i32>,
    /// move token - 378 (-379 at a root)
    pub mv: Vec<i32>,
    pub depth: Vec<i32>,
    pub born: Vec<i32>,
    pub degree: Vec<i32>,
    pub prior: Vec<f64>,
    pub boot: Vec<f64>,
    /// the sum of the node's children's priors (1 up to rounding once expanded, 0 before)
    pub mass: Vec<f64>,
    /// Position::outcome (-1 ongoing)
    pub terminal: Vec<f64>,
    pub roots: Vec<i32>,
}

pub struct Backup {
    first: Vec<i32>,
    next: Vec<i32>,
    roots: Vec<i32>,
    mv: Vec<i32>,
    degree: Vec<i32>,
    prior: Vec<f64>,
    /// -boot: win minus loss for the side to move
    base: Vec<f64>,
    mass: Vec<f64>,
    term: Vec<f64>,
    lc: Vec<f64>,
}

impl Backup {
    /// Nodes born after `budget` (a pull number) are left out, as if the search had stopped there.
    pub fn new(t: &Compact, budget: i32, scale: f64) -> Result<Backup, &'static str> {
        if !(scale > 0.) || !scale.is_finite() {
            return Err("positive scale required");
        }
        let n = t.parent.len();
        let (mut first, mut next) = (vec![-1i32; n], vec![-1i32; n]);
        for i in 0..n {
            let p = t.parent[i];
            if p >= i as i32 {
                return Err("parent order");
            }
            if p >= 0 && t.born[i] <= budget {
                next[i] = first[p as usize];
                first[p as usize] = i as i32;
            }
        }
        let mut b = Backup {
            first,
            next,
            roots: t.roots.clone(),
            mv: t.mv.clone(),
            degree: t.degree.clone(),
            prior: t.prior.clone(),
            base: t.boot.iter().map(|x| -x).collect(),
            mass: t.mass.clone(),
            term: t.terminal.clone(),
            lc: vec![0.; n],
        };
        let mut count = vec![1i64; n];
        for i in (0..n).rev() {
            for c in b.children(i) {
                count[i] += count[c];
            }
            b.lc[i] = ((count[i] as f64 - 1.) / scale).ln_1p();
        }
        Ok(b)
    }

    fn children(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        let step = move |c: i32| (c >= 0).then_some(c as usize);
        std::iter::successors(step(self.first[i]), move |&c| step(self.next[c]))
    }

    /// ids: [roots, k] move ids (token - 378). Returns [3, roots, k]: the value for the mover (the root's
    /// own where the move was never expanded), d/da, d/db (0 there).
    pub fn reduce(&self, a: f64, b: f64, ids: &[i64], k: usize) -> Result<Vec<f64>, &'static str> {
        let (n, r) = (self.first.len(), self.roots.len());
        if ids.len() != r * k {
            return Err("root IDs shape");
        }
        let (mut v, mut ga, mut gb) = (vec![0.; n], vec![0.; n], vec![0.; n]);
        for i in (0..n).rev() {
            if self.term[i] >= 0. {
                v[i] = if self.term[i] == 0.5 { 0. } else { -1. };
                continue;
            }
            if self.mass[i] == 0. || self.first[i] < 0 {
                v[i] = self.base[i];
                continue;
            }
            let tau = (a + b * self.lc[i]).exp();
            let (mut active, mut seen, mut hi) = (0, 0., f64::NEG_INFINITY);
            for c in self.children(i) {
                active += 1;
                seen += self.prior[c];
                if self.prior[c] > 0. {
                    hi = hi.max(-v[c]);
                }
            }
            let rest = if active == self.degree[i] { 0. } else { (self.mass[i] - seen).max(0.) };
            if rest > 0. {
                hi = hi.max(self.base[i]);
            }
            let tail = if rest > 0. { rest * ((self.base[i] - hi) / tau).exp() } else { 0. };
            let (mut z, mut mean, mut da, mut db) = (tail, tail * self.base[i], 0., 0.);
            for c in self.children(i) {
                let w = self.prior[c] * ((-v[c] - hi) / tau).exp();
                z += w;
                mean -= w * v[c];
                da -= w * ga[c];
                db -= w * gb[c];
            }
            v[i] = hi + tau * (z / self.mass[i]).ln();
            let local = v[i] - mean / z;
            ga[i] = local + da / z;
            gb[i] = self.lc[i] * local + db / z;
        }
        let mut out = vec![0.; 3 * r * k];
        for (row, &root) in self.roots.iter().enumerate() {
            let root = root as usize;
            let mut child = vec![-1i32; 1968];
            for c in self.children(root) {
                child[self.mv[c] as usize] = c as i32;
            }
            for j in 0..k {
                let id = ids[row * k + j];
                if !(0..1968).contains(&id) {
                    return Err("move ID");
                }
                let c = child[id as usize];
                let at = |part: usize| (part * r + row) * k + j;
                out[at(0)] = if c < 0 { self.base[root] } else { -v[c as usize] };
                if c >= 0 {
                    out[at(1)] = -ga[c as usize];
                    out[at(2)] = -gb[c as usize];
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A root whose one expanded move leads to node 1, two of whose three moves are expanded: node 3 a good
    /// reply for its mover, node 2 a bad one.
    fn tree() -> Compact {
        Compact {
            parent: vec![-1, 0, 1, 1],
            mv: vec![-379, 5, 11, 13],
            depth: vec![0, 1, 2, 2],
            born: vec![0, 1, 2, 3],
            degree: vec![1, 3, 0, 0],
            prior: vec![0., 1., 0.5, 0.3],
            boot: vec![0.1, 0.2, -0.9, 0.4],
            mass: vec![1., 1., 0., 0.],
            terminal: vec![-1.; 4],
            roots: vec![0],
        }
    }

    #[test]
    fn soft_backup() {
        let t = tree();
        let out = Backup::new(&t, 3, 16.).unwrap().reduce(0., 0., &[5, 6], 2).unwrap();
        // node 1 at tau 1: hi .4 (its best reply, for it), the unseen .2 of prior mass at its own value -.2
        let z = 0.2 * (-0.6f64).exp() + 0.3 + 0.5 * (-1.3f64).exp();
        assert!((out[0] + 0.4 + z.ln()).abs() < 1e-12);
        assert_eq!(out[1], -0.1); // never expanded: the root's own value
        assert!(out[2].is_finite() && out[4].is_finite() && out[3] == 0. && out[5] == 0.);
        let cut = Backup::new(&t, 2, 16.).unwrap().reduce(0., 0., &[5], 1).unwrap(); // node 3 born after the budget
        assert!((cut[0] - (0.2 - (0.5 + 0.5 * (-0.7f64).exp()).ln())).abs() < 1e-12);
        let one = Backup::new(&t, 1, 16.).unwrap().reduce(0., 0., &[5], 1).unwrap(); // node 1 unexpanded: its own value, for the root's mover
        assert_eq!(one[0], 0.2);
        assert!(Backup::new(&t, 3, 0.).is_err());
        assert!(Backup::new(&t, 3, 16.).unwrap().reduce(0., 0., &[1968], 1).is_err());
    }

    #[test]
    fn terminal_children() {
        let t = Compact {
            parent: vec![-1, 0, 0],
            mv: vec![-379, 5, 7],
            depth: vec![0, 1, 1],
            born: vec![0, 1, 2],
            degree: vec![2, 0, 0],
            prior: vec![0., 0.6, 0.4],
            boot: vec![0.; 3],
            mass: vec![1., 0., 0.],
            terminal: vec![-1., 0., 0.5],
            roots: vec![0],
        };
        let out = Backup::new(&t, 2, 16.).unwrap().reduce(-1., 0.5, &[5, 7], 2).unwrap();
        assert_eq!((out[0], out[1]), (1., 0.)); // mate delivered: +1 for the mover; a draw 0
        assert_eq!(&out[2..], &[0.; 4]);
    }
}
