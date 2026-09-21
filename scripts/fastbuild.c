/*
 * C core of fastbuild.py: chessdata.record() on Hugging Face rows (python-chess SAN semantics,
 * clocks, evals, end state, blake2b hashes), a bucket-sorted zstd spill, and shard columns.
 * A row the fast path cannot reproduce exactly is handed back to Python (fb_parse returns it).
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef uint64_t u64; typedef int64_t i64; typedef uint32_t u32; typedef int32_t i32;
typedef uint16_t u16; typedef int16_t i16; typedef uint8_t u8; typedef int8_t i8;

typedef struct ZSTD_CCtx_s ZSTD_CCtx;
typedef struct ZSTD_DCtx_s ZSTD_DCtx;
ZSTD_CCtx *ZSTD_createCCtx(void);
size_t ZSTD_freeCCtx(ZSTD_CCtx *c);
size_t ZSTD_compressCCtx(ZSTD_CCtx *c, void *dst, size_t cap, const void *src, size_t n, int level);
ZSTD_DCtx *ZSTD_createDCtx(void);
size_t ZSTD_freeDCtx(ZSTD_DCtx *d);
size_t ZSTD_decompressDCtx(ZSTD_DCtx *d, void *dst, size_t cap, const void *src, size_t n);
size_t ZSTD_compressBound(size_t n);
unsigned ZSTD_isError(size_t code);

#define BIT(s) (1ULL << (s))
#define lsb(b) __builtin_ctzll(b)
#define msb(b) (63 - __builtin_clzll(b))
#define RANK1 0xffULL
#define RANK8 (0xffULL << 56)
#define FILEA 0x0101010101010101ULL
#define DARK 0xaa55aa55aa55aa55ULL
#define NB 2100 /* dense buckets: 7 formats x 25 max-Elo bins x 12 min-Elo bins */
#define MISSING (-32768)

enum { P = 1, N, B, R, Q, K };
enum { EVENT, SITE, WHITE, BLACK, RESULT, WTITLE, BTITLE, WELO, BELO, WDIFF, BDIFF, DATE, TIME,
       ECO, OPENING, TERM, TC, MOVETEXT, NCOL };
enum { S_SITE, S_WHITE, S_BLACK, S_WTITLE, S_BTITLE, S_ECO, S_OPENING, NSTR };
enum { OK, SKIP_ELO, SKIP_ILLEGAL, FALLBACK };
enum { NUL_WDIFF = NSTR, NUL_BDIFF };

static u64 KN[64], KI[64], PA[2][64], RAY[8][64];
static u8 CLS[256]; /* 1 whitespace, 2 may start a STRIP match */
static u16 IDX[64][64][6];
static const u64 *VAL;
static long NVAL;

struct pos { u64 bb[7], co[2], castle; u8 sq[64]; int turn, ep, half; };
struct key { u64 bb[6], co[2], castle, turn; };
struct col { const u8 *valid; i64 voff; const i32 *off; const void *data; };

struct game {
	const char *s[NSTR]; u32 len[NSTR];
	u32 nulls; u8 rated; i8 fmt, kind, term, result, end;
	i16 welo, belo, wdiff, bdiff, inc;
	i32 base; i64 utc; u64 move_hash, token_hash;
	const u16 *mv; const u32 *clk; const i16 *ev; u32 nm, nc, ne;
};

struct hdr {
	u32 size, local, nm, nc, ne, nulls, len[NSTR];
	u64 move_hash, token_hash; i64 utc; i32 base;
	i16 welo, belo, wdiff, bdiff, inc;
	i8 fmt, kind, term, result, end; u8 rated;
};

struct run {
	char *buf; size_t len, cap; u32 n;
	i64 bad[3]; i64 order[2]; i64 norder;
	u16 *mv, *tok; u32 *clk; i16 *ev; struct key *win; size_t mcap, wn;
};

struct item { double key; u64 g, off; };
struct bucket { char *buf; struct item *it; u8 *leak; u64 n; };

static void *xrealloc(void *p, size_t n)
{
	if (!(p = realloc(p, n ? n : 1)))
		abort();
	return p;
}

static inline u64 slide(int s, u64 occ, int d)
{
	u64 r = RAY[d][s], b = r & occ;
	return b ? r ^ RAY[d][d & 1 ? msb(b) : lsb(b)] : r;
}

static inline u64 ratt(int s, u64 o) { return slide(s, o, 0) | slide(s, o, 1) | slide(s, o, 2) | slide(s, o, 3); }
static inline u64 batt(int s, u64 o) { return slide(s, o, 4) | slide(s, o, 5) | slide(s, o, 6) | slide(s, o, 7); }

static inline u64 att(int pc, int s, u64 occ)
{
	switch (pc) {
	case N: return KN[s];
	case B: return batt(s, occ);
	case R: return ratt(s, occ);
	case Q: return batt(s, occ) | ratt(s, occ);
	default: return KI[s];
	}
}

static inline u64 attackers(const struct pos *p, int s, int c, u64 occ)
{
	return p->co[c] & ((KN[s] & p->bb[N]) | (KI[s] & p->bb[K]) | (PA[c ^ 1][s] & p->bb[P]) |
			   (batt(s, occ) & (p->bb[B] | p->bb[Q])) | (ratt(s, occ) & (p->bb[R] | p->bb[Q])));
}

static inline int ksq(const struct pos *p, int c) { return lsb(p->bb[K] & p->co[c]); }

static void startpos(struct pos *p)
{
	static const u8 back[8] = {R, N, B, Q, K, B, N, R};

	memset(p, 0, sizeof(*p));
	for (int f = 0; f < 8; f++) {
		p->sq[f] = back[f], p->sq[8 + f] = P, p->sq[48 + f] = P | 8, p->sq[56 + f] = back[f] | 8;
		p->bb[back[f]] |= BIT(f) | BIT(56 + f);
	}
	p->bb[P] = 0xff00ULL | 0xff00ULL << 40;
	p->co[0] = 0xffffULL;
	p->co[1] = 0xffffULL << 48;
	p->castle = BIT(0) | BIT(7) | BIT(56) | BIT(63);
	p->ep = -1;
}

/* Board.push() of a pseudo-legal move */
static void make(struct pos *p, int f, int t, int pr)
{
	int us = p->turn, pc = p->sq[f] & 7, cap = p->sq[t] & 7, ep = p->ep;
	u64 fb = BIT(f), tb = BIT(t);

	p->ep = -1;
	p->half = pc == P || cap ? 0 : p->half + 1;
	p->castle &= ~(fb | tb);
	if (pc == K)
		p->castle &= us ? ~RANK8 : ~RANK1;
	if (cap) {
		p->bb[cap] ^= tb;
		p->co[us ^ 1] ^= tb;
	}
	p->bb[pc] ^= fb;
	p->co[us] ^= fb | tb;
	p->sq[f] = 0;
	if (pc == P) {
		if (t == ep && !cap) {
			int c = t + (us ? 8 : -8);
			p->bb[P] ^= BIT(c);
			p->co[us ^ 1] ^= BIT(c);
			p->sq[c] = 0;
		} else if (t - f == 16 || f - t == 16) {
			p->ep = (f + t) / 2;
		}
		if (pr)
			pc = pr;
	}
	p->bb[pc] ^= tb;
	p->sq[t] = pc | us << 3;
	if (pc == K && (t - f == 2 || f - t == 2)) {
		int rf = t > f ? t + 1 : t - 2, rt = (f + t) / 2;
		p->bb[R] ^= BIT(rf) | BIT(rt);
		p->co[us] ^= BIT(rf) | BIT(rt);
		p->sq[rt] = p->sq[rf];
		p->sq[rf] = 0;
	}
	p->turn ^= 1;
}

/* Board._is_safe(): our king is not attacked after f -> t capturing on x (-1 if nothing) */
static int safe(const struct pos *p, int f, int t, int x)
{
	int c = p->turn, k = (p->sq[f] & 7) == K ? t : ksq(p, c);
	u64 xb = x >= 0 ? BIT(x) : 0, occ = ((p->co[0] | p->co[1]) & ~BIT(f) & ~xb) | BIT(t);

	return !(attackers(p, k, c ^ 1, occ) & ~xb);
}

static inline int captured(const struct pos *p, int f, int t)
{
	if (p->sq[t])
		return t;
	return (p->sq[f] & 7) == P && t == p->ep ? t + (p->turn ? 8 : -8) : -1;
}

/* generate_castling_moves(); side 0 king side, 1 queen side */
static int can_castle(const struct pos *p, int side)
{
	int c = p->turn, e = c ? 60 : 4, rk = side ? e - 4 : e + 3, kt = side ? e - 2 : e + 2, rt = side ? e - 1 : e + 1;
	u64 occ = p->co[0] | p->co[1];

	if (!(p->castle & BIT(rk)) || p->sq[e] != (K | c << 3) || p->sq[rk] != (R | c << 3))
		return 0;
	if (occ & (side ? BIT(e - 1) | BIT(e - 2) | BIT(e - 3) : BIT(e + 1) | BIT(e + 2)))
		return 0;
	return !attackers(p, e, c ^ 1, occ ^ BIT(e)) && !attackers(p, rt, c ^ 1, occ ^ BIT(e)) &&
	       !attackers(p, kt, c ^ 1, occ ^ BIT(e) ^ BIT(rk) ^ BIT(rt));
}

/* Board.is_legal() of (f, t, pr) */
static int legal(const struct pos *p, int f, int t, int pr)
{
	int c = p->turn, v = p->sq[f], pc = v & 7;
	u64 own = p->co[c], occ = own | p->co[c ^ 1];

	if (!v || v >> 3 != c || own & BIT(t))
		return 0;
	if (pc == P) {
		int d = c ? -8 : 8;
		if ((c ? t < 8 : t >= 56) ? pr < N || pr > Q : pr)
			return 0;
		if (t == f + d) {
			if (occ & BIT(t))
				return 0;
		} else if (t == f + 2 * d) {
			if (f >> 3 != (c ? 6 : 1) || occ & (BIT(f + d) | BIT(t)))
				return 0;
		} else if (!(PA[c][f] & BIT(t)) || (!(p->co[c ^ 1] & BIT(t)) && t != p->ep)) {
			return 0;
		}
	} else {
		if (pr)
			return 0;
		if (pc == K && f == (c ? 60 : 4) && (t == f + 2 || t == f - 2))
			return can_castle(p, t < f);
		if (!(att(pc, f, occ) & BIT(t)))
			return 0;
	}
	return safe(p, f, t, captured(p, f, t));
}

static int any_legal(const struct pos *p)
{
	int c = p->turn, d = c ? -8 : 8;
	u64 own = p->co[c], occ = own | p->co[c ^ 1];

	for (u64 m = own; m; m &= m - 1) {
		int f = lsb(m), pc = p->sq[f] & 7;
		u64 t;
		if (pc == P) {
			t = PA[c][f] & (p->co[c ^ 1] | (p->ep >= 0 ? BIT(p->ep) : 0));
			if (!(occ & BIT(f + d))) {
				t |= BIT(f + d);
				if (f >> 3 == (c ? 6 : 1) && !(occ & BIT(f + 2 * d)))
					t |= BIT(f + 2 * d);
			}
		} else {
			t = att(pc, f, occ) & ~own;
		}
		for (; t; t &= t - 1)
			if (safe(p, f, lsb(t), captured(p, f, lsb(t))))
				return 1;
	}
	return can_castle(p, 0) || can_castle(p, 1);
}

static int legal_ep(const struct pos *p)
{
	int c = p->turn;

	if (p->ep < 0)
		return 0;
	for (u64 m = p->bb[P] & p->co[c] & PA[c ^ 1][p->ep]; m; m &= m - 1)
		if (safe(p, lsb(m), p->ep, p->ep + (c ? 8 : -8)))
			return 1;
	return 0;
}

static int ptype(int ch)
{
	switch (ch | 32) {
	case 'n': return N;
	case 'b': return B;
	case 'r': return R;
	case 'q': return Q;
	case 'k': return K;
	}
	return 0;
}

/* Board.parse_san(): the unique legal move as f | t << 6 | promotion << 12; -1 if it raises */
static int parse_san(const struct pos *p, const char *s, int n, int *mv)
{
	int c = p->turn, pr = 0, t, ff = -1, fr = -1, pc = 0, found = 0, f = -1;
	u64 cand, own = p->co[c], occ = own | p->co[c ^ 1], mask;

	if (n && (s[n - 1] == '+' || s[n - 1] == '#'))
		n--;
	if ((n == 3 || n == 5) && (!memcmp(s, "O-O-O", n) || !memcmp(s, "0-0-0", n))) {
		int e = c ? 60 : 4, side = n == 5;
		if (!can_castle(p, side))
			return -1;
		*mv = e | (side ? e - 2 : e + 2) << 6;
		return 0;
	}
	if (n && ptype(s[n - 1])) {
		pr = ptype(s[--n]);
		if (n && s[n - 1] == '=')
			n--;
	}
	if (n < 2 || s[n - 2] < 'a' || s[n - 2] > 'h' || s[n - 1] < '1' || s[n - 1] > '8')
		return -1;
	t = (s[n - 1] - '1') * 8 + s[n - 2] - 'a';
	n -= 2;
	if (n && (s[n - 1] == '-' || s[n - 1] == 'x'))
		n--;
	if (n && s[n - 1] >= '1' && s[n - 1] <= '8')
		fr = s[--n] - '1';
	if (n && s[n - 1] >= 'a' && s[n - 1] <= 'h')
		ff = s[--n] - 'a';
	if (n && s[n - 1] >= 'B' && s[n - 1] <= 'R' && ptype(s[n - 1]))
		pc = ptype(s[--n]);
	if (n)
		return -1;
	mask = (ff >= 0 ? FILEA << ff : ~0ULL) & (fr >= 0 ? RANK1 << 8 * fr : ~0ULL);
	if (pc) {
		if (pr || own & BIT(t))
			return -1;
		for (cand = p->bb[pc] & own & mask & att(pc, t, occ); cand; cand &= cand - 1)
			if (safe(p, lsb(cand), t, p->sq[t] ? t : -1)) {
				if (found++)
					return -1;
				f = lsb(cand);
			}
	} else if (ff >= 0 && fr >= 0) { /* find_move() */
		int fpr = pr;
		f = fr * 8 + ff;
		if (!fpr && p->bb[P] & BIT(f) && BIT(t) & (RANK1 | RANK8))
			fpr = Q;
		if (!fpr && (f == 4 || f == 60) && p->bb[K] & BIT(f))
			t = t == f + 3 ? f + 2 : t == f - 4 ? f - 2 : t;
		if (!legal(p, f, t, fpr) || fpr != pr)
			return -1;
		found = 1;
	} else {
		for (cand = p->bb[P] & own & mask & (ff >= 0 ? ~0ULL : FILEA << (t & 7)); cand; cand &= cand - 1)
			if (legal(p, lsb(cand), t, pr)) {
				if (found++)
					return -1;
				f = lsb(cand);
			}
	}
	if (found != 1)
		return -1;
	*mv = f | t << 6 | pr << 12;
	return 0;
}

static const u64 IV[8] = {0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
			  0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
			  0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};
static const u8 SIGMA[12][16] = {
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
	{11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
	{9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
	{12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
	{6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
	{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

#define ROR(x, n) ((x) >> (n) | (x) << (64 - (n)))
#define G(a, b, c, d, x, y) \
	(a += b + (x), d = ROR(d ^ a, 32), c += d, b = ROR(b ^ c, 24), \
	 a += b + (y), d = ROR(d ^ a, 16), c += d, b = ROR(b ^ c, 63))

static void blake_block(u64 h[8], const u8 *blk, u64 t, int last)
{
	u64 m[16], v[16];

	memcpy(m, blk, sizeof(m));
	for (int i = 0; i < 8; i++)
		v[i] = h[i], v[i + 8] = IV[i];
	v[12] ^= t;
	v[14] ^= last ? ~0ULL : 0;
	for (int r = 0; r < 12; r++) {
		const u8 *s = SIGMA[r];
		G(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
		G(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
		G(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
		G(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);
		G(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
		G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
		G(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
		G(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
	}
	for (int i = 0; i < 8; i++)
		h[i] ^= v[i] ^ v[i + 8];
}

/* chessdata.hash64(): blake2b with an 8-byte digest, read little endian */
static u64 hash64(const u16 *x, size_t n)
{
	const u8 *p = (const u8 *)x;
	u8 blk[128] = {0};
	u64 h[8], t = 0;

	memcpy(h, IV, sizeof(h));
	h[0] ^= 0x01010000 ^ 8;
	for (n *= 2; n > 128; p += 128, n -= 128)
		blake_block(h, p, t += 128, 0);
	if (n)
		memcpy(blk, p, n);
	blake_block(h, blk, t + n, 1);
	return h[0];
}

static inline int space(int c) { return CLS[(u8)c] == 1; }
static inline int digit(int c) { return c >= '0' && c <= '9'; }

static u64 num(const char *s, long n)
{
	u64 v = 0;

	while (n--)
		v = v * 10 + *s++ - '0';
	return v;
}

/* end of the chessdata.STRIP match at s[i], or -1 */
static long strip(const char *s, long i, long n)
{
	const char *e;
	long j;

	switch (s[i]) {
	case '{':
		return (e = memchr(s + i + 1, '}', n - i - 1)) ? e - s + 1 : -1;
	case '(':
		return (e = memchr(s + i + 1, ')', n - i - 1)) ? e - s + 1 : -1;
	case '$':
		for (j = i + 1; j < n && digit(s[j]); j++)
			;
		return j > i + 1 ? j : -1;
	case '*':
		return i + 1;
	case '?':
	case '!':
		for (j = i; j < n && (s[j] == '?' || s[j] == '!'); j++)
			;
		return j;
	}
	if (!digit(s[i]))
		return -1;
	for (j = i; j < n && digit(s[j]); j++)
		;
	if (j < n && s[j] == '.')
		return j + 2 < n && s[j + 1] == '.' && s[j + 2] == '.' ? j + 3 : j + 1;
	if (n - i >= 3 && (!memcmp(s + i, "1-0", 3) || !memcmp(s + i, "0-1", 3)))
		return i + 3;
	return n - i >= 7 && !memcmp(s + i, "1/2-1/2", 7) ? i + 7 : -1;
}

/* chessdata.CLOCK.findall() and whether "%eval" occurs: count, values stored below cap; -1 to fall back */
static long clocks(const char *s, long n, u32 *out, long cap, int *eval)
{
	const char *e = s + n, *p = s;
	long k = 0;

	for (*eval = 0; (p = memchr(p, '%', e - p)); p++) {
		const char *q = p + 5, *d[3];
		long len[3];
		int g;
		if (e - p >= 5 && !memcmp(p, "%eval", 5))
			*eval = 1;
		if (p == s || p[-1] != '[' || e - p < 5 || memcmp(p, "%clk ", 5))
			continue;
		for (g = 0; g < 3; g++, q++) {
			for (d[g] = q; q < e && digit(*q); q++)
				;
			len[g] = q - d[g];
			if (!len[g] || q >= e || *q != (g < 2 ? ':' : ']'))
				break;
		}
		if (g < 3)
			continue;
		if (len[0] > 9 || len[1] > 9 || len[2] > 9)
			return -1;
		u64 v = 3600 * num(d[0], len[0]) + 60 * num(d[1], len[1]) + num(d[2], len[2]);
		if (v > UINT32_MAX)
			return -1;
		if (k < cap)
			out[k] = v;
		k++;
		p = q - 1;
	}
	return k;
}

/* chessdata.EVAL.search() in one comment: 1 and the value, 0 if none, -1 to fall back */
static int eval_value(const char *a, const char *b, i16 *v)
{
	for (const char *p = a; (p = memchr(p, '[', b - p)); p++) {
		const char *q = p + 7, *vs, *ds;
		int mate, dots = 0;
		if (b - p < 7 || memcmp(p, "[%eval ", 7))
			continue;
		mate = q < b && *q == '#';
		vs = q += mate;
		q += q < b && *q == '-';
		for (ds = q; q < b && (digit(*q) || *q == '.'); q++)
			dots += *q == '.';
		if (q == ds || q >= b || *q != ']')
			continue;
		long nd = q - ds - dots;
		if (mate) {
			if (dots || nd > 9)
				return -1;
			i64 x = (32000 - (i64)num(ds, nd)) * (*vs == '-' ? -1 : 1);
			if (x < -32768 || x > 32767)
				return -1;
			*v = x;
		} else {
			char buf[64];
			if (dots > 1 || !nd || q - vs >= (long)sizeof(buf))
				return -1;
			memcpy(buf, vs, q - vs);
			buf[q - vs] = 0;
			double y = nearbyint(strtod(buf, NULL) * 100);
			*v = y > 30000 ? 30000 : y < -30000 ? -30000 : (i16)y;
		}
		return 1;
	}
	return 0;
}

static int not_move(const char *t, long n)
{
	long i = 0;

	if (n == 3 && (!memcmp(t, "1-0", 3) || !memcmp(t, "0-1", 3)))
		return 1;
	if ((n == 7 && !memcmp(t, "1/2-1/2", 7)) || (n == 1 && *t == '*'))
		return 1;
	if (*t == '$') {
		for (i = 1; i < n && digit(t[i]); i++)
			;
		return n > 1 && i == n;
	}
	for (; i < n && digit(t[i]); i++)
		;
	return i && ((i + 1 == n && t[i] == '.') || (i + 3 == n && !memcmp(t + i, "...", 3)));
}

/* chessdata.eval_list() of a movetext containing "%eval", stored below cap: length, 0 for [], -1 to fall back */
static long eval_list(const char *s, long n, i16 *out, long cap)
{
	long i = 0, k = 0;
	int seen = 0, r;
	i16 v;

	while (i < n) {
		if (s[i] == '{') {
			const char *e = memchr(s + i + 1, '}', n - i - 1);
			if (!e) {
				i++;
				continue;
			}
			if (k && (r = eval_value(s + i + 1, e, &v))) {
				if (r < 0)
					return -1;
				seen = 1;
				if (k <= cap)
					out[k - 1] = v;
			}
			i = e - s + 1;
		} else if (space(s[i]) || s[i] == '}') {
			i++;
		} else {
			long j = i;
			while (j < n && !space(s[j]) && s[j] != '{' && s[j] != '}')
				j++;
			if (!not_move(s + i, j - i)) {
				if (k < cap)
					out[k] = MISSING;
				k++;
			}
			i = j;
		}
	}
	return seen ? k : 0;
}

static void key(struct key *k, const struct pos *p)
{
	memcpy(k->bb, p->bb + 1, sizeof(k->bb));
	k->co[0] = p->co[0];
	k->co[1] = p->co[1];
	k->castle = p->castle;
	k->turn = p->turn;
}

/* STRIP, split, parse_san, push; win holds the positions since the last irreversible move (is_repetition) */
static int moves(struct run *r, struct pos *p, const char *s, long n, u32 *nm)
{
	long i = 0, e = -1, t;
	size_t w = 0;
	u32 k = 0;
	int c, mv, f, to, pc;

	key(&r->win[w++], p);
	while (i < n) {
		if ((c = CLS[(u8)s[i]]) == 1 || (c == 2 && (e = strip(s, i, n)) >= 0)) {
			i = c == 1 ? i + 1 : e;
			continue;
		}
		for (t = i++; i < n && (c = CLS[(u8)s[i]]) != 1 && (c != 2 || strip(s, i, n) < 0); i++)
			;
		if (parse_san(p, s + t, i - t, &mv) < 0)
			return -1;
		f = mv & 63, to = mv >> 6 & 63, pc = p->sq[f] & 7;
		if (pc == P || p->sq[to] || p->castle & (BIT(f) | BIT(to)) ||
		    (pc == K && p->castle & (p->turn ? RANK8 : RANK1)) || legal_ep(p))
			w = 0;
		make(p, f, to, mv >> 12);
		key(&r->win[w++], p);
		r->mv[k++] = IDX[f][to][mv >> 12];
	}
	*nm = k;
	r->wn = w;
	return 0;
}

static int insufficient(const struct pos *p, int c)
{
	u64 us = p->co[c];

	if (us & (p->bb[P] | p->bb[R] | p->bb[Q]))
		return 0;
	if (us & p->bb[N])
		return __builtin_popcountll(us) <= 2 && !(p->co[c ^ 1] & ~p->bb[K] & ~p->bb[Q]);
	if (us & p->bb[B])
		return (!(p->bb[B] & DARK) || !(p->bb[B] & ~DARK)) && !p->bb[P] && !p->bb[N];
	return 1;
}

/* chessdata.end_state() */
static int end_state(const struct run *r, const struct pos *p)
{
	const struct key *w = r->win;
	long last = r->wn - 1, n = 0;
	int c = p->turn;

	if (!any_legal(p))
		return attackers(p, ksq(p, c), c ^ 1, p->co[0] | p->co[1]) ? 1 : 2;
	if (insufficient(p, 0) && insufficient(p, 1))
		return 3;
	if (!legal_ep(p))
		for (long x = last - 2; x >= 0; x -= 2)
			if (!memcmp(w + x, w + last, sizeof(*w)) && ++n == 2)
				return 4;
	return p->half >= 100 ? 5 : 0;
}

static inline int valid(const struct col *c, long i)
{
	return c->data && (!c->valid || c->valid[(c->voff + i) >> 3] >> ((c->voff + i) & 7) & 1);
}

static inline const char *str(const struct col *c, long i, u32 *len)
{
	*len = c->off[i + 1] - c->off[i];
	return (const char *)c->data + c->off[i];
}

static int find(const char *s, u32 n, const char *const *opt, int k)
{
	for (int i = 0; i < k; i++)
		if (strlen(opt[i]) == n && !memcmp(s, opt[i], n))
			return i;
	return k;
}

static const char *const FORMATS[] = {"UltraBullet", "Bullet", "Blitz", "Rapid", "Classical", "Correspondence"};
static const char *const KINDS[] = {"game", "tournament", "swiss"};
static const char *const TERMS[] = {"Normal", "Time forfeit", "Abandoned", "Rules infraction", "Unterminated"};
static const char *const RESULTS[] = {"1-0", "0-1", "1/2-1/2", "*"};

static int elo_digits(int v, u16 *out)
{
	char b[16];
	int n = snprintf(b, sizeof(b), "%04d", v);

	for (int i = 0; i < n; i++)
		out[i] = b[i] - '0';
	return n;
}

static int seconds_id(int base)
{
	static const int low[6] = {0, 15, 30, 45, 60, 90};

	if (base == -1)
		return 192 + 185;
	for (int i = 0; i < 6; i++)
		if (base == low[i])
			return 192 + i;
	return base >= 120 && base <= 10800 && base % 60 == 0 ? 192 + 4 + base / 60 : 2349;
}

static int increment_id(int inc) { return inc == -1 ? 191 : inc >= 0 && inc <= 180 ? 10 + inc : 2349; }

static void reserve(struct run *r, size_t n)
{
	if (n <= r->mcap)
		return;
	r->mcap = 2 * n;
	r->mv = xrealloc(r->mv, r->mcap * sizeof(*r->mv));
	r->tok = xrealloc(r->tok, (r->mcap + 32) * sizeof(*r->tok));
	r->clk = xrealloc(r->clk, r->mcap * sizeof(*r->clk));
	r->ev = xrealloc(r->ev, r->mcap * sizeof(*r->ev));
	r->win = xrealloc(r->win, (r->mcap + 1) * sizeof(*r->win));
}

static void put(struct run *r, const struct game *g)
{
	struct hdr h;
	size_t sz = sizeof(h) + 2 * g->nm + 4 * g->nc + 2 * g->ne;
	char *d;

	for (int k = 0; k < NSTR; k++)
		sz += g->len[k];
	sz = (sz + 7) & ~(size_t)7;
	if (r->len + sz > r->cap)
		r->buf = xrealloc(r->buf, r->cap = 2 * (r->len + sz));
	d = r->buf + r->len;
	memset(&h, 0, sizeof(h));
	h.size = sz, h.local = r->n, h.nm = g->nm, h.nc = g->nc, h.ne = g->ne, h.nulls = g->nulls;
	memcpy(h.len, g->len, sizeof(h.len));
	h.move_hash = g->move_hash, h.token_hash = g->token_hash, h.utc = g->utc, h.base = g->base;
	h.welo = g->welo, h.belo = g->belo, h.wdiff = g->wdiff, h.bdiff = g->bdiff, h.inc = g->inc;
	h.fmt = g->fmt, h.kind = g->kind, h.term = g->term, h.result = g->result, h.end = g->end, h.rated = g->rated;
	memcpy(d, &h, sizeof(h));
	d += sizeof(h);
	for (int k = 0; k < NSTR; k++)
		if (g->len[k])
			memcpy(d, g->s[k], g->len[k]), d += g->len[k];
	if (g->nm)
		memcpy(d, g->mv, 2 * g->nm), d += 2 * g->nm;
	if (g->nc)
		memcpy(d, g->clk, 4 * g->nc), d += 4 * g->nc;
	if (g->ne)
		memcpy(d, g->ev, 2 * g->ne), d += 2 * g->ne;
	memset(d, 0, r->buf + r->len + sz - d);
	r->len += sz;
	r->n++;
}

/* chessdata.record(hf_header(row), movetext) */
static int row(struct run *r, const struct col *c, long i)
{
	struct game g;
	struct pos p;
	const char *mt = "", *s;
	u32 n = 0, len;
	long k;
	int ev;

	if (!valid(&c[WELO], i) || !valid(&c[BELO], i))
		return SKIP_ELO;
	memset(&g, 0, sizeof(g));
	g.welo = ((const i16 *)c[WELO].data)[i];
	g.belo = ((const i16 *)c[BELO].data)[i];
	if (g.welo < 0 || g.belo < 0)
		return FALLBACK; /* elo_digits() raises */
	if (valid(&c[MOVETEXT], i))
		mt = str(&c[MOVETEXT], i, &n);
	for (u32 j = 0; j < n; j++)
		if (mt[j] & 0x80)
			return FALLBACK;
	reserve(r, n / 2 + 8);
	startpos(&p);
	if (moves(r, &p, mt, n, &g.nm) < 0)
		return SKIP_ILLEGAL;
	g.mv = r->mv;

	if ((k = clocks(mt, n, r->clk, g.nm, &ev)) < 0)
		return FALLBACK;
	g.clk = r->clk, g.nc = k == g.nm ? k : 0;
	if ((k = ev ? eval_list(mt, n, r->ev, g.nm) : 0) < 0)
		return FALLBACK;
	g.ev = r->ev, g.ne = k == g.nm ? k : 0;

	g.base = g.inc = -1;
	if (valid(&c[TC], i) && (s = str(&c[TC], i, &len), memchr(s, '+', len))) {
		const char *plus = memchr(s, '+', len);
		long a = plus - s, b = len - a - 1;
		if (a < 1 || a > 9 || b < 1 || b > 4)
			return FALLBACK;
		for (long j = 0; j < len; j++)
			if (j != a && !digit(s[j]))
				return FALLBACK;
		g.base = num(s, a), g.inc = num(plus + 1, b);
	}

	u32 ns = 0, tl[3] = {0};
	const char *tok[3] = {""};
	if (valid(&c[EVENT], i)) {
		s = str(&c[EVENT], i, &len);
		for (u32 j = 0, a = 0; j <= len && ns < 3; j++)
			if (j == len || s[j] == ' ')
				tok[ns] = s + a, tl[ns++] = j - a, a = j + 1;
	} else {
		ns = 1;
	}
	g.rated = tl[0] == 5 && !memcmp(tok[0], "Rated", 5);
	g.fmt = g.rated && ns < 2 ? 6 : find(tok[g.rated], tl[g.rated], FORMATS, 6);
	g.kind = ns - g.rated < 2 ? 3 : find(tok[g.rated + 1], tl[g.rated + 1], KINDS, 3);

	static const int cols[NSTR] = {SITE, WHITE, BLACK, WTITLE, BTITLE, ECO, OPENING};
	for (int j = 0; j < NSTR; j++) {
		g.s[j] = "";
		if (!valid(&c[cols[j]], i)) {
			g.nulls |= j == S_SITE ? 0 : 1u << j;
			continue;
		}
		g.s[j] = str(&c[cols[j]], i, &g.len[j]);
		if ((j == S_WTITLE || j == S_BTITLE) && !g.len[j])
			g.nulls |= 1u << j;
	}
	for (u32 j = g.len[S_SITE]; j--;)
		if (g.s[S_SITE][j] == '/') {
			g.s[S_SITE] += j + 1, g.len[S_SITE] -= j + 1;
			break;
		}
	if (valid(&c[WDIFF], i))
		g.wdiff = ((const i16 *)c[WDIFF].data)[i];
	else
		g.nulls |= 1u << NUL_WDIFF;
	if (valid(&c[BDIFF], i))
		g.bdiff = ((const i16 *)c[BDIFF].data)[i];
	else
		g.nulls |= 1u << NUL_BDIFF;
	g.term = valid(&c[TERM], i) ? (s = str(&c[TERM], i, &len), find(s, len, TERMS, 5)) : 5;
	g.result = valid(&c[RESULT], i) ? (s = str(&c[RESULT], i, &len), find(s, len, RESULTS, 4)) : 4;

	i32 day = valid(&c[DATE], i) ? ((const i32 *)c[DATE].data)[i] : 0;
	i32 ms = valid(&c[TIME], i) ? ((const i32 *)c[TIME].data)[i] : 0;
	if (day < -719162 || day > 2932896 || ms < 0 || ms >= 86400000)
		return FALLBACK; /* to_pylist() raises */
	g.utc = valid(&c[DATE], i) && valid(&c[TIME], i) ? (i64)day * 86400 + ms / 1000 : 0;

	g.end = end_state(r, &p);
	u16 *t = r->tok;
	k = 0;
	t[k++] = 2348, t[k++] = seconds_id(g.base), t[k++] = increment_id(g.inc);
	k += elo_digits(g.welo, t + k);
	k += elo_digits(g.belo, t + k);
	for (u32 j = 0; j < g.nm; j++)
		t[k++] = g.mv[j] + 378;
	t[k++] = g.term == 0 ? 2346 : 2347;
	g.token_hash = hash64(t, k);
	g.move_hash = hash64(g.mv, g.nm);
	put(r, &g);
	return OK;
}

void fb_init(const char *moves, const u64 *val, long nval)
{
	static const int dr[8] = {1, -1, 0, 0, 1, -1, 1, -1}, df[8] = {0, 0, 1, -1, 1, -1, -1, 1};
	static const int kr[8] = {1, 2, 2, 1, -1, -2, -2, -1}, kf[8] = {2, 1, -1, -2, -2, -1, 1, 2};

	for (int s = 0; s < 64; s++) {
		int r = s >> 3, f = s & 7;
		for (int d = 0; d < 8; d++) {
			RAY[d][s] = KI[s] = KN[s] = 0;
			for (int rr = r + dr[d], ff = f + df[d]; rr >= 0 && rr < 8 && ff >= 0 && ff < 8; rr += dr[d], ff += df[d])
				RAY[d][s] |= BIT(rr * 8 + ff);
		}
		for (int d = 0; d < 8; d++) {
			int rr = r + dr[d], ff = f + df[d];
			if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
				KI[s] |= BIT(rr * 8 + ff);
			rr = r + kr[d], ff = f + kf[d];
			if (rr >= 0 && rr < 8 && ff >= 0 && ff < 8)
				KN[s] |= BIT(rr * 8 + ff);
		}
		PA[0][s] = PA[1][s] = 0;
		if (r < 7)
			PA[0][s] = (f > 0 ? BIT(s + 7) : 0) | (f < 7 ? BIT(s + 9) : 0);
		if (r > 0)
			PA[1][s] = (f > 0 ? BIT(s - 9) : 0) | (f < 7 ? BIT(s - 7) : 0);
	}
	for (const char *c = " \t\n\v\f\r\x1c\x1d\x1e\x1f"; *c; c++)
		CLS[(u8)*c] = 1;
	for (const char *c = "{($*?!0123456789"; *c; c++)
		CLS[(u8)*c] = 2;
	for (int i = 0; *moves; i++) {
		int f = moves[0] - 'a' + (moves[1] - '1') * 8, t = moves[2] - 'a' + (moves[3] - '1') * 8;
		int pr = moves[4] > ' ' ? ptype(moves[4]) : 0;
		IDX[f][t][pr] = i;
		moves += pr ? 5 : 4;
		while (*moves == ' ')
			moves++;
	}
	VAL = val, NVAL = nval;
}

struct run *fb_run_new(void)
{
	struct run *r = calloc(1, sizeof(*r));

	if (!r)
		abort();
	return r;
}

void fb_run_free(struct run *r)
{
	free(r->buf), free(r->mv), free(r->tok), free(r->clk), free(r->ev), free(r->win), free(r);
}

void fb_skip(struct run *r, int kind)
{
	if (!r->bad[kind]++)
		r->order[r->norder++] = kind;
}

/* rows [i, n): the first row needing Python, or n */
long fb_parse(struct run *r, const struct col *c, long i, long n)
{
	for (; i < n; i++) {
		int s = row(r, c, i);
		if (s == FALLBACK)
			return i;
		if (s != OK)
			fb_skip(r, s);
	}
	return n;
}

void fb_append(struct run *r, const struct game *g) { put(r, g); }

void fb_run_info(const struct run *r, i64 *out)
{
	out[0] = r->n, out[1] = r->bad[SKIP_ELO], out[2] = r->bad[SKIP_ILLEGAL];
	out[3] = r->norder, out[4] = r->order[0], out[5] = r->order[1], out[6] = r->len;
}

/* the record at byte offset off as a game (pointers into the run); the next offset */
u64 fb_record(const char *buf, u64 off, struct game *g)
{
	struct hdr h;
	const char *d = buf + off + sizeof(h);

	memcpy(&h, buf + off, sizeof(h));
	memset(g, 0, sizeof(*g));
	for (int k = 0; k < NSTR; k++)
		g->s[k] = d, g->len[k] = h.len[k], d += h.len[k];
	g->nulls = h.nulls, g->rated = h.rated, g->fmt = h.fmt, g->kind = h.kind, g->term = h.term;
	g->result = h.result, g->end = h.end, g->welo = h.welo, g->belo = h.belo, g->wdiff = h.wdiff;
	g->bdiff = h.bdiff, g->inc = h.inc, g->base = h.base, g->utc = h.utc;
	g->move_hash = h.move_hash, g->token_hash = h.token_hash;
	g->mv = (const u16 *)d, g->nm = h.nm, d += 2 * h.nm;
	g->clk = (const u32 *)d, g->nc = h.nc, d += 4 * h.nc;
	g->ev = (const i16 *)d, g->ne = h.ne;
	return off + h.size;
}

const char *fb_run_buf(const struct run *r) { return r->buf; }

static inline int dense(const struct hdr *h)
{
	int hi = (h->welo > h->belo ? h->welo : h->belo) / 100, lo = (h->welo < h->belo ? h->welo : h->belo) / 200;

	hi = hi < 6 ? 6 : hi > 30 ? 30 : hi;
	lo = lo < 3 ? 3 : lo > 14 ? 14 : lo;
	return h->fmt * 300 + (hi - 6) * 12 + lo - 3;
}

static inline int code(int d) { return (d / 300 + 1) * 10000 + (d / 12 % 25 + 6) * 100 + d % 12 + 3; }

static int full_pwrite(int fd, const char *p, size_t n, u64 off)
{
	while (n) {
		ssize_t k = pwrite(fd, p, n, off);
		if (k <= 0)
			return -1;
		p += k, n -= k, off += k;
	}
	return 0;
}

static int full_pread(int fd, char *p, size_t n, u64 off)
{
	while (n) {
		ssize_t k = pread(fd, p, n, off);
		if (k <= 0)
			return -1;
		p += k, n -= k, off += k;
	}
	return 0;
}

/* group the run's records by bucket (stable), zstd each group, append all at *at; out: 5 i64 per group */
long fb_spill(const struct run *r, int fd, u64 *at, int level, i64 *out)
{
	size_t *pos = calloc(NB + 1, sizeof(*pos)), *raw = calloc(NB, sizeof(*raw)), clen = 0, cap = 0;
	u32 *cnt = calloc(NB, sizeof(*cnt));
	char *sorted = xrealloc(NULL, r->len), *z;
	ZSTD_CCtx *cc = ZSTD_createCCtx();
	long m = 0, ret = -1;
	struct hdr h;

	for (u64 o = 0; o < r->len; o += h.size) {
		memcpy(&h, r->buf + o, sizeof(h));
		int d = dense(&h);
		raw[d] += h.size, cnt[d]++;
	}
	for (int d = 0; d < NB; d++)
		pos[d + 1] = pos[d] + raw[d], cap += raw[d] ? ZSTD_compressBound(raw[d]) : 0;
	for (u64 o = 0; o < r->len; o += h.size) {
		memcpy(&h, r->buf + o, sizeof(h));
		int d = dense(&h);
		memcpy(sorted + pos[d], r->buf + o, h.size);
		pos[d] += h.size;
	}
	z = xrealloc(NULL, cap);
	for (int d = 0; d < NB; d++) {
		if (!cnt[d])
			continue;
		size_t k = ZSTD_compressCCtx(cc, z + clen, cap - clen, sorted + pos[d] - raw[d], raw[d], level);
		if (ZSTD_isError(k))
			goto out;
		out[5 * m] = code(d), out[5 * m + 1] = clen, out[5 * m + 2] = k, out[5 * m + 3] = raw[d];
		out[5 * m + 4] = cnt[d];
		clen += k, m++;
	}
	u64 base = __atomic_fetch_add(at, clen, __ATOMIC_RELAXED);
	if (full_pwrite(fd, z, clen, base))
		goto out;
	for (long j = 0; j < m; j++)
		out[5 * j + 1] += base;
	ret = m;
out:
	ZSTD_freeCCtx(cc);
	free(pos), free(raw), free(cnt), free(sorted), free(z);
	return ret;
}

static int by_key(const void *a, const void *b)
{
	const struct item *x = a, *y = b;
	return x->key < y->key ? -1 : x->key > y->key ? 1 : x->g < y->g ? -1 : x->g > y->g;
}

static int has_val(u64 h)
{
	long lo = 0, hi = NVAL;

	while (lo < hi) {
		long mid = (lo + hi) / 2;
		if (VAL[mid] < h)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo < NVAL && VAL[lo] == h;
}

static int is_bot(const struct game *g, int k) { return !(g->nulls >> k & 1) && g->len[k] == 3 && !memcmp(g->s[k], "BOT", 3); }

/* chessdata.summarize() terms of one game; st layout in fastbuild.py STATS */
static void tally(i64 *st, const struct game *g, int leak)
{
	i64 pl = g->nm, wm = (pl + 1) / 2, bm = pl / 2, mine;
	int ew = g->welo >= 2400, eb = g->belo >= 2400, wb = is_bot(g, S_WTITLE), bb = is_bot(g, S_BTITLE);
	int wd = g->nulls >> NUL_WDIFF & 1 ? 0 : abs(g->wdiff), bd = g->nulls >> NUL_BDIFF & 1 ? 0 : abs(g->bdiff);

	mine = wm * ew + bm * eb;
	st[0]++, st[1] += pl, st[2 + g->fmt]++, st[9] += mine;
	if ((g->welo + g->belo) / 2 >= 2400)
		st[10] += pl;
	else
		st[11] += mine;
	st[12] += wm * (ew && wb) + bm * (eb && bb);
	st[13] += wm * (ew && wd >= 25) + bm * (eb && bd >= 25);
	st[14] += wm * (ew && wd >= 50) + bm * (eb && bd >= 50);
	st[15] += wb || bb;
	st[16] += !g->nc && pl;
	st[17] += leak;
	st[18 + (g->welo > g->belo ? g->welo : g->belo) / 100]++;
}

/* one bucket: its spill groups (6 i64 each: seq, offset, clen, rlen, count, -) in run order, sorted like
   np.lexsort((rng.random(n), bucket)) */
struct bucket *fb_load(int fd, const i64 *ent, long n, const i64 *base, const double *rand, i64 *st)
{
	struct bucket *b = calloc(1, sizeof(*b));
	ZSTD_DCtx *dc = ZSTD_createDCtx();
	size_t total = 0, at = 0, zcap = 0;
	char *z = NULL;
	struct game g;
	struct hdr h;

	for (long j = 0; j < n; j++)
		total += ent[6 * j + 3], b->n += ent[6 * j + 4];
	b->buf = xrealloc(NULL, total);
	b->it = xrealloc(NULL, b->n * sizeof(*b->it));
	b->leak = xrealloc(NULL, b->n);
	for (long j = 0, k = 0; j < n; j++) {
		const i64 *e = ent + 6 * j;
		if ((size_t)e[2] > zcap)
			z = xrealloc(z, zcap = 2 * e[2]);
		if (full_pread(fd, z, e[2], e[1]) || ZSTD_decompressDCtx(dc, b->buf + at, e[3], z, e[2]) != (size_t)e[3])
			abort();
		for (u64 o = at; o < at + e[3]; o += h.size, k++) {
			memcpy(&h, b->buf + o, sizeof(h));
			b->it[k].g = base[e[0]] + h.local;
			b->it[k].key = rand[b->it[k].g];
			b->it[k].off = o;
		}
		at += e[3];
	}
	qsort(b->it, b->n, sizeof(*b->it), by_key);
	for (u64 k = 0; k < b->n; k++) {
		fb_record(b->buf, b->it[k].off, &g);
		b->leak[k] = has_val(g.token_hash);
		tally(st, &g, b->leak[k]);
	}
	ZSTD_freeDCtx(dc);
	free(z);
	return b;
}

void fb_bucket_free(struct bucket *b) { free(b->buf), free(b->it), free(b->leak), free(b); }

/* byte counts of the 7 string columns, then element counts of moves, clocks, evals, for rows [lo, lo + n) */
void fb_shard_size(const struct bucket *b, long lo, long n, i64 *sz)
{
	struct game g;

	memset(sz, 0, 10 * sizeof(*sz));
	for (long i = lo; i < lo + n; i++) {
		fb_record(b->buf, b->it[i].off, &g);
		for (int k = 0; k < NSTR; k++)
			sz[k] += g.len[k];
		sz[7] += g.nm, sz[8] += g.nc, sz[9] += g.ne;
	}
}

/* column buffers of rows [lo, lo + n), in fastbuild.py FILL order */
void fb_shard_fill(const struct bucket *b, long lo, long n, int code, void **o)
{
	i64 so[NSTR] = {0}, mo = 0, co = 0, eo = 0;
	struct game g;

	for (int k = 0; k < NSTR; k++)
		((i64 *)o[k])[0] = 0;
	((i64 *)o[34])[0] = ((i64 *)o[36])[0] = ((i64 *)o[41])[0] = 0;
	for (long j = 0; j < n; j++) {
		fb_record(b->buf, b->it[lo + j].off, &g);
		for (int k = 0; k < NSTR; k++) {
			memcpy((char *)o[7 + k] + so[k], g.s[k], g.len[k]);
			((i64 *)o[k])[j + 1] = so[k] += g.len[k];
			if (k)
				((u8 *)o[13 + k])[j] = !(g.nulls >> k & 1);
		}
		((i16 *)o[20])[j] = g.welo, ((i16 *)o[21])[j] = g.belo;
		((i16 *)o[22])[j] = g.nulls >> NUL_WDIFF & 1 ? 0 : g.wdiff;
		((i16 *)o[23])[j] = g.nulls >> NUL_BDIFF & 1 ? 0 : g.bdiff;
		((u8 *)o[24])[j] = !(g.nulls >> NUL_WDIFF & 1), ((u8 *)o[25])[j] = !(g.nulls >> NUL_BDIFF & 1);
		((i8 *)o[26])[j] = g.fmt, ((i8 *)o[27])[j] = g.kind, ((u8 *)o[28])[j] = g.rated;
		((i32 *)o[29])[j] = g.base, ((i16 *)o[30])[j] = g.inc;
		((i8 *)o[31])[j] = g.term, ((i8 *)o[32])[j] = g.result, ((i64 *)o[33])[j] = g.utc;
		memcpy((u16 *)o[35] + mo, g.mv, 2 * g.nm), ((i64 *)o[34])[j + 1] = mo += g.nm;
		memcpy((u32 *)o[37] + co, g.clk, 4 * g.nc), ((i64 *)o[36])[j + 1] = co += g.nc;
		((u64 *)o[38])[j] = g.move_hash, ((u64 *)o[39])[j] = g.token_hash, ((i8 *)o[40])[j] = g.end;
		memcpy((i16 *)o[42] + eo, g.ev, 2 * g.ne), ((i64 *)o[41])[j + 1] = eo += g.ne;
		((u8 *)o[43])[j] = b->leak[lo + j], ((i32 *)o[44])[j] = code;
	}
}
