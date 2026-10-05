//! board.py's clock bookkeeping of search nodes as tree.py's `Nodes` keeps it: a node's features are the
//! mover's seconds, the opponent's and the mover's previous own think time (-1 unknown), its parent's
//! advanced by the think time the parent's time head predicts (`advance_clocks`, `predicted_seconds`,
//! `root_other_previous`). One `Clocks` per view: views have their own increment and root features, the
//! predicted think times are the tree's.

use std::ops::Range;
use std::sync::OnceLock;

use crate::chess::HEADER;

pub const TIME: Range<usize> = 2350..2413;

/// The time head's bin centres: np.r_[arange(16), 16 exp(arange(47) / 7.06)].
fn centres() -> &'static [f64; 63] {
    static C: OnceLock<[f64; 63]> = OnceLock::new();
    C.get_or_init(|| std::array::from_fn(|i| if i < 16 { i as f64 } else { 16. * ((i - 16) as f64 / 7.06).exp() }))
}

/// The expected think time in seconds under one row of logits (softmax of the time head times the centres).
pub fn predicted_seconds(z: &[f64]) -> f64 {
    let t = &z[TIME];
    let max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = t.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = e.iter().sum();
    e.iter().zip(centres()).map(|(p, c)| p / sum * c).sum()
}

pub struct Clocks {
    pub inc: i64,
    /// per node (by id): mover seconds, opponent seconds, mover's previous own think time
    pub feats: Vec<[f64; 3]>,
    /// per node: the think time its mover spent before (the child's third feature), -1 unknown
    pub other: Vec<f64>,
}

impl Clocks {
    /// features: the game's clock features at every prefix token (float32, -1 unknown); inc: the increment
    /// in seconds, -1 without a clock. The root's `other` is the opponent's last think time read off the
    /// prefix's clocks (board.py root_other_previous), in float32 as numpy computes it.
    pub fn new(features: &[[f32; 3]], inc: i64) -> Clocks {
        let n = features.len();
        let other = if n < HEADER + 3 || inc < 0 {
            -1.
        } else {
            let (before, after) = (features[n - 2][0], features[n - 1][1]);
            let x = before - after + inc as f32;
            if before >= 0. && after >= 0. && x >= 0. { f64::from(x) } else { -1. }
        };
        Clocks { inc, feats: vec![features[n - 1].map(f64::from)], other: vec![other] }
    }

    /// The next node's features below `parent`, whose prefix is `length` tokens and whose mover is predicted
    /// to think `elapsed` seconds (board.py advance_clocks; np.rint is round half to even). Each side's
    /// first move does not tick the clock.
    pub fn push(&mut self, parent: usize, length: usize, elapsed: f64) {
        let p = self.feats[parent];
        let ply = length as i64 - HEADER as i64;
        let valid = p[0] >= 0. && p[1] >= 0. && self.inc >= 0;
        let spent = if ply < 2 { 0. } else { elapsed.round_ties_even().max(0.).min(p[0].max(0.)) };
        let after = (p[0] - spent).max(0.) + if ply < 2 { 0. } else { self.inc as f64 };
        let previous = if ply + 1 >= 4 { self.other[parent] } else { -1. };
        self.feats.push(if valid { [p[1], after, previous] } else { [-1.; 3] });
        self.other.push(if valid && ply >= 2 { spent } else { -1. });
    }

    /// A re-rooted tree's clocks: the root's from `features` (the game's, two plies longer) as `new`
    /// computes them, then the kept nodes' (kept[1..], old ids in new id order) as they were.
    pub fn rebase(&mut self, features: &[[f32; 3]], kept: &[usize]) {
        let fresh = Clocks::new(features, self.inc);
        let feats = std::iter::once(fresh.feats[0]).chain(kept[1..].iter().map(|&i| self.feats[i])).collect();
        let other = std::iter::once(fresh.other[0]).chain(kept[1..].iter().map(|&i| self.other[i])).collect();
        self.feats = feats;
        self.other = other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centres_and_seconds() {
        let c = centres();
        assert_eq!((c[0], c[15], c[16]), (0., 15., 16.));
        assert!((c[17] - 18.43464513047231).abs() < 1e-12 && (c[62] - 16. * (46f64 / 7.06).exp()).abs() < 1e-9);
        let mut z = vec![0.; 2432];
        z[2350 + 20] = 50.; // all the mass in bin 20
        assert!((predicted_seconds(&z) - c[20]).abs() < 1e-12);
    }

    #[test]
    fn advance() {
        let f = |a: f32, b: f32, c: f32| [a, b, c];
        let game = [f(-1., -1., -1.); 11].into_iter().chain([f(180., 180., -1.), f(180., 178., -1.), f(176., 178., -1.), f(170., 165., 3.), f(163., 170., 5.)]).collect::<Vec<_>>();
        let mut c = Clocks::new(&game, 2);
        assert_eq!(c.other[0], 2.); // 170 - 170 + 2
        c.push(0, 16, 4.4); // ply 5: 4 s spent, 163 - 4 + 2 left for the mover, the opponent's previous own time
        assert_eq!((c.feats[1], c.other[1]), ([170., 161., 2.], 4.));
        c.push(1, 17, 10.5); // half to even
        assert_eq!((c.feats[2], c.other[2]), ([161., 162., 4.], 10.));
        c.push(2, 18, 1000.); // capped by the clock
        assert_eq!((c.feats[3], c.other[3]), ([162., 2., 10.], 161.));
        let mut early = Clocks::new(&game[..12], 2); // the first moves do not tick
        assert_eq!(early.other[0], -1.);
        early.push(0, 12, 7.);
        assert_eq!((early.feats[1], early.other[1]), ([180., 180., -1.], -1.));
        let mut none = Clocks::new(&game, -1);
        none.push(0, 16, 4.);
        assert_eq!((none.feats[1], none.other[1]), ([-1.; 3], -1.));
        // re-rooted at node 2 (kept with node 3): the root's clocks from the longer game, the rest as they were
        let (f2, f3, o3) = (c.feats[2], c.feats[3], c.other[3]);
        let longer: Vec<_> = game.iter().copied().chain([f(170., 158., 5.), f(150., 160., 8.)]).collect();
        c.rebase(&longer, &[2, 3]);
        assert_eq!((c.feats.len(), c.feats[0], c.other[0]), (2, [150., 160., 8.], 160. - 150. + 2.));
        assert_eq!((c.feats[1], c.other[1]), (f3, o3));
        assert_ne!(c.feats[0], f2);
    }
}
