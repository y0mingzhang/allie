//! Chess rules (shakmaty), Allie's move vocabulary and its 68-byte board state, as the model's encoder
//! (src/allie/model/board_encode.cpp) and allie.data.vocab define them. `Position` follows the native
//! search's Position (src/allie/search/native/tree.cpp) exactly: its legal-move order, its outcome rules
//! and chess-library's repetition count over the moves pushed into it.

use std::sync::OnceLock;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Bitboard, Chess, Color, EnPassantMode, File, Move, Position as _, Role, Square};

pub const MOVE_START: i64 = 378;
pub const HEADER: usize = 11;
/// squares 0..63 (a1 = 0; 1-6 white PNBRQK, 7-12 black), white to move, castling rights (1 K, 2 Q, 4 k, 8 q),
/// en-passant file + 1 after any double pawn push, inside a game
pub const START: [u8; 68] = {
    let mut b = [0u8; 68];
    let back = [4u8, 2, 3, 5, 6, 3, 2, 4];
    let mut f = 0;
    while f < 8 {
        b[f] = back[f];
        b[8 + f] = 1;
        b[48 + f] = 7;
        b[56 + f] = back[f] + 6;
        f += 1;
    }
    b[64] = 1;
    b[65] = 15;
    b[67] = 1;
    b
};

/// The move vocabulary: every queen or knight move between squares and every promotion, as sorted UCI
/// (allie.lichess.tokens._moves); token = MOVE_START + index.
pub struct Vocab {
    pub uci: Vec<String>,
    /// from, to, promotion role (0 none, 2 n, 3 b, 4 r, 5 q) per move
    pub table: Vec<[u8; 3]>,
    index: Vec<i16>,
}

impl Vocab {
    fn build() -> Vocab {
        let name = |s: usize| format!("{}{}", (b'a' + (s % 8) as u8) as char, s / 8 + 1);
        let mut uci = Vec::new();
        for a in 0..64usize {
            for b in 0..64 {
                let (df, dr) = ((a % 8).abs_diff(b % 8), (a / 8).abs_diff(b / 8));
                if a != b && (df == 0 || dr == 0 || df == dr || df.min(dr) == 1 && df.max(dr) == 2) {
                    uci.push(name(a) + &name(b));
                }
            }
            if a / 8 == 1 || a / 8 == 6 {
                let last = if a / 8 == 1 { 0 } else { 56 };
                for f in (a % 8).saturating_sub(1)..=(a % 8 + 1).min(7) {
                    uci.extend("bnqr".chars().map(|p| format!("{}{}{p}", name(a), name(last + f))));
                }
            }
        }
        uci.sort();
        let sq = |s: &[u8]| (s[0] - b'a') + (s[1] - b'1') * 8;
        let table: Vec<[u8; 3]> = uci
            .iter()
            .map(|m| {
                let m = m.as_bytes();
                let promo = m.get(4).map_or(0, |&p| b" nbrq".iter().position(|&c| c == p).unwrap() as u8 + 1);
                [sq(&m[0..2]), sq(&m[2..4]), promo]
            })
            .collect();
        let mut index = vec![-1i16; 64 * 64 * 6];
        for (i, &[fr, to, promo]) in table.iter().enumerate() {
            index[Vocab::key(fr, to, promo)] = i as i16;
        }
        Vocab { uci, table, index }
    }

    fn key(fr: u8, to: u8, promo: u8) -> usize {
        (fr as usize * 64 + to as usize) * 6 + promo as usize
    }

    /// The token of a move, or -1 when the vocabulary has none.
    pub fn token(&self, fr: u8, to: u8, promo: u8) -> i64 {
        match self.index[Vocab::key(fr, to, promo)] {
            -1 => -1,
            i => MOVE_START + i as i64,
        }
    }

    pub fn entry(&self, token: i64) -> Result<[u8; 3], &'static str> {
        usize::try_from(token - MOVE_START)
            .ok()
            .and_then(|i| self.table.get(i).copied())
            .ok_or("not a move token")
    }
}

pub fn vocab() -> &'static Vocab {
    static VOCAB: OnceLock<Vocab> = OnceLock::new();
    VOCAB.get_or_init(Vocab::build)
}

/// The board state after a move token (tokens.advance, with board_encode.cpp's checks). The move
/// must be legal; only the checks cheap enough for the encoder are made.
pub fn advance(b: &mut [u8; 68], token: i64) -> Result<(), &'static str> {
    let [fr, to, promotion] = vocab().entry(token)?;
    let (fr, to) = (fr as usize, to as usize);
    if b[67] != 1 {
        return Err("not inside a game");
    }
    let white = b[64] != 0;
    let piece = b[fr];
    if piece == 0 || (piece <= 6) != white {
        return Err("no piece of the side to move on the from square");
    }
    let captured = b[to];
    if captured != 0 && (captured <= 6) == white {
        return Err("captures an own piece");
    }
    let kind = (piece - 1) % 6 + 1;
    let ep = if b[66] != 0 { b[66] as usize - 1 + if white { 40 } else { 16 } } else { usize::MAX };
    if kind == 1 && to == ep && captured == 0 && fr % 8 != to % 8 {
        let victim = if white { to - 8 } else { to + 8 };
        if b[victim] != if white { 7 } else { 1 } {
            return Err("no pawn to take en passant");
        }
        b[victim] = 0;
    }
    b[fr] = 0;
    b[to] = if promotion != 0 { promotion + if white { 0 } else { 6 } } else { piece };
    if kind == 6 {
        b[65] &= if white { 12 } else { 3 };
        if fr.abs_diff(to) == 2 {
            let rook = if to > fr { to + 1 } else { to - 2 };
            if b[rook] != if white { 4 } else { 10 } {
                return Err("no rook to castle with");
            }
            b[(fr + to) / 2] = b[rook];
            b[rook] = 0;
        }
    }
    for (square, right) in [(0, 2), (7, 1), (56, 8), (63, 4)] {
        if fr == square || to == square {
            b[65] &= 15 ^ right;
        }
    }
    b[66] = if kind == 1 && fr.abs_diff(to) == 16 { (to % 8 + 1) as u8 } else { 0 };
    b[64] = !white as u8;
    Ok(())
}

/// The board state at every prefix position of a token sequence: START for the header tokens, then one
/// advance per move token (a Game's boards).
pub fn encode(tokens: &[i64]) -> Result<Vec<u8>, &'static str> {
    let mut out = Vec::with_capacity(tokens.len() * 68);
    let mut state = START;
    for (i, &t) in tokens.iter().enumerate() {
        if i >= HEADER {
            advance(&mut state, t)?;
        }
        out.extend_from_slice(&state);
    }
    Ok(out)
}

fn hash(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::PseudoLegal).0
}

fn index(sq: Square) -> i32 {
    sq as i32
}

/// A position reached from the start by pushed move tokens, with the hashes of the positions before each
/// of them (chess-library's prev_states_), so copies carry the repetition history.
#[pyclass]
#[derive(Clone)]
pub struct Position {
    pos: Chess,
    hash: u64,
    history: Vec<u64>,
}

impl Default for Position {
    fn default() -> Self {
        let pos = Chess::default();
        Position { hash: hash(&pos), pos, history: Vec::new() }
    }
}

impl Position {
    /// A token's move in this position, built as chess-library's uciToMove does (castling is the king's
    /// two-square move; a pawn's diagonal move onto an empty square is en passant), if legal.
    fn to_move(&self, token: i64) -> Result<Move, &'static str> {
        let [fr, to, promo] = vocab().entry(token)?;
        let (from, to) = (Square::new(fr as u32), Square::new(to as u32));
        let board = self.pos.board();
        let role = board.role_at(from).ok_or("illegal move")?;
        let m = if role == Role::King && from.rank() == to.rank() && from.file().distance(to.file()) == 2 {
            let rook = Square::from_coords(if to > from { File::H } else { File::A }, from.rank());
            Move::Castle { king: from, rook }
        } else if role == Role::Pawn && from.file() != to.file() && board.role_at(to).is_none() {
            Move::EnPassant { from, to }
        } else {
            let promotion = (promo != 0).then(|| Role::ALL[promo as usize - 1]);
            Move::Normal { role, from, capture: board.role_at(to), to, promotion }
        };
        if self.pos.is_legal(&m) {
            Ok(m)
        } else {
            Err("illegal move")
        }
    }

    /// chess-library's isRepetition(count): the current position occurred `count` times before, among
    /// the positions since the last irreversible move.
    fn repetition(&self, count: usize) -> bool {
        let size = self.history.len() as i64;
        let floor = (size - self.pos.halfmoves() as i64 - 1).max(0);
        let mut i = size - 2;
        let mut c = 0;
        while i >= floor {
            if self.history[i as usize] == self.hash {
                c += 1;
                if c == count {
                    return true;
                }
            }
            i -= 2;
        }
        false
    }

    pub fn push_token(&mut self, token: i64) -> Result<(), &'static str> {
        let m = self.to_move(token)?;
        self.history.push(self.hash);
        self.pos.play_unchecked(&m);
        self.hash = hash(&self.pos);
        Ok(())
    }

    /// Legal moves as tokens in tree.cpp's order: castling, king moves out of check, other piece moves,
    /// pawn pushes (single, then double), pawn captures, en passant; promotions q, r, b, n.
    pub fn legal_tokens(&self) -> Vec<i64> {
        let (v, check) = (vocab(), self.pos.is_check());
        let mut keyed: Vec<([i32; 4], i64)> = Vec::with_capacity(64);
        for m in self.pos.legal_moves() {
            let (from, to, promo) = match m {
                Move::Castle { king, rook } => {
                    let file = if rook > king { File::G } else { File::C };
                    (king, Square::from_coords(file, king.rank()), 0)
                }
                Move::Normal { from, to, promotion, .. } => (from, to, promotion.map_or(0, |r| r as u8)),
                Move::EnPassant { from, to } => (from, to, 0),
                Move::Put { .. } => unreachable!(),
            };
            let (f, t) = (index(from), index(to));
            let mut key = [0, -f, -t, 0];
            match m {
                Move::Castle { rook, .. } => key = [1, -index(rook), -t, 0],
                _ if m.role() == Role::King && check => key[0] = -1,
                Move::EnPassant { .. } => key[0] = 5,
                _ if m.role() == Role::Pawn => {
                    if from.file() != to.file() {
                        key[0] = 2;
                    } else {
                        key = [if (f - t).abs() == 16 { 4 } else { 3 }, -t, 0, 0];
                    }
                    key[3] = match promo {
                        5 => 0,
                        4 => 1,
                        3 => 2,
                        2 => 3,
                        _ => 0,
                    };
                }
                _ => {}
            }
            keyed.push((key, v.token(from as u8, to as u8, promo)));
        }
        keyed.sort_unstable();
        keyed.into_iter().map(|x| x.1).collect()
    }

}

fn err(e: &'static str) -> PyErr {
    PyValueError::new_err(e)
}

#[pymethods]
impl Position {
    #[new]
    pub fn new() -> Self {
        Position::default()
    }

    /// The position after a game prefix's move tokens (prefix[11:]), as allie.search.native.from_prefix.
    #[staticmethod]
    pub fn from_tokens(prefix: Vec<i64>) -> PyResult<Self> {
        let mut p = Position::default();
        for &t in prefix.get(HEADER..).unwrap_or_default() {
            p.push_token(t).map_err(err)?;
        }
        Ok(p)
    }

    pub fn push(&mut self, token: i64) -> PyResult<()> {
        self.push_token(token).map_err(err)
    }

    pub fn child(&self, token: i64) -> PyResult<Self> {
        let mut p = self.clone();
        p.push(token)?;
        Ok(p)
    }

    pub fn legal(&self) -> Vec<i64> {
        self.legal_tokens()
    }

    /// -1 ongoing, else the expected score for White: checkmate, stalemate, insufficient material,
    /// 75-move rule and five-fold repetition, as tree.cpp.
    pub fn outcome(&self) -> f64 {
        if self.pos.legal_moves().is_empty() {
            return match (self.pos.is_check(), self.pos.turn()) {
                (false, _) => 0.5,
                (true, Color::White) => 0.,
                (true, Color::Black) => 1.,
            };
        }
        let b = self.pos.board();
        if (b.pawns() | b.rooks() | b.queens()).is_empty() {
            let (bishops, knights) = (b.bishops(), b.knights());
            let dark = Bitboard::DARK_SQUARES.0;
            if bishops.is_empty() && knights.count() <= 1
                || knights.is_empty() && (bishops.0 & dark == 0 || bishops.0 & !dark == 0)
            {
                return 0.5;
            }
        }
        if self.pos.halfmoves() >= 150 || self.repetition(4) {
            return 0.5;
        }
        -1.
    }

    #[getter]
    pub fn white(&self) -> bool {
        self.pos.turn() == Color::White
    }

    pub fn fen(&self) -> String {
        shakmaty::fen::Fen::from_position(self.pos.clone(), EnPassantMode::PseudoLegal).to_string()
    }
}

#[pyfunction]
fn moves() -> Vec<String> {
    vocab().uci.clone()
}

#[pyfunction]
fn advance_board<'py>(py: Python<'py>, state: &[u8], token: i64) -> PyResult<Bound<'py, PyBytes>> {
    let mut b: [u8; 68] = state.try_into().map_err(|_| err("board state must be 68 bytes"))?;
    advance(&mut b, token).map_err(err)?;
    Ok(PyBytes::new_bound(py, &b))
}

#[pyfunction]
fn encode_boards<'py>(py: Python<'py>, tokens: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let out = match tokens.extract::<PyReadonlyArray1<i64>>() {
        Ok(a) => encode(a.as_slice()?),
        Err(_) => encode(&tokens.extract::<Vec<i64>>()?),
    }
    .map_err(err)?;
    let n = out.len() / 68;
    PyArray1::from_vec_bound(py, out).reshape([n, 68])
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Position>()?;
    m.add_function(wrap_pyfunction!(moves, m)?)?;
    m.add_function(wrap_pyfunction!(advance_board, m)?)?;
    m.add_function(wrap_pyfunction!(encode_boards, m)?)?;
    m.add("MOVE_START", MOVE_START)?;
    m.add("HEADER", HEADER)?;
    m.add("START", PyBytes::new_bound(m.py(), &START))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn token(uci: &str) -> i64 {
        MOVE_START + vocab().uci.iter().position(|m| m == uci).unwrap() as i64
    }

    #[test]
    fn vocabulary() {
        let v = vocab();
        assert_eq!(v.uci.len(), 1968);
        assert_eq!(v.uci[0], "a1a2");
        assert_eq!(&v.uci[23..28], ["a2a1", "a2a1b", "a2a1n", "a2a1q", "a2a1r"]);
        assert!(v.uci.windows(2).all(|w| w[0] < w[1]));
        for (i, m) in v.uci.iter().enumerate() {
            let [fr, to, promo] = v.table[i];
            assert_eq!(v.token(fr, to, promo), MOVE_START + i as i64, "{m}");
        }
        assert_eq!(v.token(4, 6, 0), token("e1g1"));
        assert_eq!(v.token(0, 1, 1), -1);
        assert_eq!(v.entry(377), Err("not a move token"));
        assert_eq!(v.entry(378 + 1968), Err("not a move token"));
    }

    #[test]
    fn start_order_and_outcome() {
        let p = Position::default();
        let legal: Vec<&str> = p.legal_tokens().iter().map(|&t| vocab().uci[(t - MOVE_START) as usize].as_str()).collect();
        let mut want = vec!["g1h3", "g1f3", "b1c3", "b1a3"];
        want.extend(["h2h3", "g2g3", "f2f3", "e2e3", "d2d3", "c2c3", "b2b3", "a2a3"]);
        want.extend(["h2h4", "g2g4", "f2f4", "e2e4", "d2d4", "c2c4", "b2b4", "a2a4"]);
        assert_eq!(legal, want);
        assert_eq!(p.outcome(), -1.);
        assert!(p.white());
        assert_eq!(p.fen(), "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn pushes_and_errors() {
        let mut p = Position::default();
        assert_eq!(p.push_token(5), Err("not a move token"));
        assert_eq!(p.push_token(token("e2e5")), Err("illegal move"));
        assert_eq!(p.push_token(token("e1g1")), Err("illegal move"));
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            p.push_token(token(m)).unwrap();
        }
        assert_eq!(p.outcome(), 0.);
        assert!(p.legal_tokens().is_empty());
        let mut p = Position::default();
        for _ in 0..4 {
            for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
                assert_eq!(p.outcome(), -1.);
                p.push_token(token(m)).unwrap();
            }
        }
        assert_eq!(p.outcome(), 0.5);
        assert_eq!(p.child(token("e2e4")).unwrap().outcome(), -1.);
        assert_eq!(p.outcome(), 0.5);
    }

    #[test]
    fn board_state() {
        let mut b = START;
        advance(&mut b, token("e2e4")).unwrap();
        assert_eq!((b[12], b[28], b[64], b[65], b[66]), (0, 1, 0, 15, 5));
        advance(&mut b, token("e7e5")).unwrap();
        assert_eq!((b[52], b[36], b[64], b[66]), (0, 7, 1, 5));
        assert_eq!(advance(&mut b, token("d1d2")), Err("captures an own piece"));
        assert_eq!(advance(&mut b, token("e7e6")), Err("no piece of the side to move on the from square"));
        let rows = encode(&[2348, 199, 12, 1, 5, 0, 0, 1, 5, 0, 0, token("e2e4")]).unwrap();
        assert_eq!(rows.len(), 12 * 68);
        assert!(rows[..11 * 68].chunks(68).all(|r| r == START));
        assert_eq!(rows[11 * 68 + 66], 5);
    }

    /// A random game of legal moves (xorshift), its positions and the tokens that left them.
    fn random_game(seed: u64, plies: usize) -> (Vec<Position>, Vec<i64>) {
        let (mut p, mut s) = (Position::default(), seed | 1);
        let (mut positions, mut tokens) = (Vec::new(), Vec::new());
        while positions.len() < plies && p.outcome() < 0. {
            let legal = p.legal_tokens();
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let t = legal[(s % legal.len() as u64) as usize];
            positions.push(p.clone());
            tokens.push(t);
            p.push_token(t).unwrap();
        }
        (positions, tokens)
    }

    #[test]
    #[ignore]
    fn bench() {
        let (mut positions, mut tokens) = (Vec::new(), Vec::new());
        for seed in 0..200 {
            let (p, t) = random_game(seed, 150);
            positions.extend(p);
            tokens.extend(t);
        }
        let n = positions.len();
        let (start, mut count) = (Instant::now(), 0usize);
        for _ in 0..10 {
            for p in &positions {
                count += p.legal_tokens().len();
            }
        }
        let legal = start.elapsed().as_nanos() as f64 / (10 * n) as f64;
        let start = Instant::now();
        for _ in 0..10 {
            for p in &positions {
                std::hint::black_box(p.outcome());
            }
        }
        let outcome = start.elapsed().as_nanos() as f64 / (10 * n) as f64;
        let start = Instant::now();
        for _ in 0..10 {
            for (p, &t) in positions.iter().zip(&tokens) {
                std::hint::black_box(p.child(t).unwrap());
            }
        }
        let child = start.elapsed().as_nanos() as f64 / (10 * n) as f64;
        let games: Vec<Vec<i64>> = (0..200)
            .map(|seed| [vec![2348; HEADER], random_game(seed, 150).1].concat())
            .collect();
        let moves: usize = games.iter().map(|t| t.len() - HEADER).sum();
        let start = Instant::now();
        for _ in 0..10 {
            for t in &games {
                std::hint::black_box(encode(t).unwrap());
            }
        }
        let adv = start.elapsed().as_nanos() as f64 / (10 * moves) as f64;
        println!(
            "{n} positions, {count} legal moves: legal {legal:.0} ns, outcome {outcome:.0} ns, child {child:.0} ns, advance {adv:.0} ns"
        );
    }
}
