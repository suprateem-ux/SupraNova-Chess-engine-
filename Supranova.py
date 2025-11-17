# the code can do much more ig..
"""
Bitboard-friendly chess engine (single-file).

Changes from the original:
- PEP8 / autopep8-style cleanup (imports, spacing, line lengths).
- Keep board._transposition_key when present, but safely fall back to
  board.transposition_key().
- Fixed push/pop handling in several places to avoid stack imbalance.
- Narrowed broad exception handlers and clarified some code paths.
- Kept original optimizations (PosBB, hybrid SEE, move ordering, TT, mate
  search, UCI loop) and retained original global configuration values.
- [2024 Copilot PATCH] UCI stop/infinite/quit is robust, using threads and a thread-safe stop_event.
- [2025 Copilot PATCH] Improved phase-dependent PSTs (middlegame/endgame) for all piece types.
"""
from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np 
import chess
import threading

# ---------- CONFIG ----------
MAX_DEPTH = 500
MATE_VALUE = 1_000_000
INFTY = 10**9
DEFAULT_MOVE_TIME = 25.5
ASPIRATION = 25
NULL_REDUCTION = 1
LMR_BASE = 0.75
LMR_DIV = 2.0
FUTILITY_MARGIN = 150
QUIESCENCE_CAP = 2048
MULTIPV_MAX = 4
RANDOM_TIE = False
LOSS_THRESHOLD = 350
REPETITION_PENALTY = 30

# ---------- STATE ----------
@dataclass
class TTEntry:
    key: int
    depth: int
    score: int
    flag: int  # 0=EXACT, 1=LOWER, 2=UPPER
    best: Optional[chess.Move]
    age: int

TT: Dict[int, TTEntry] = {}
TT_AGE = 0
KILLERS: Dict[int, List[Optional[chess.Move]]] = {}
HISTORY: Dict[Tuple[int, int], int] = {}
node_count = 0
start_time = 0.0
time_limit = 0.0
NODE_LIMIT = None

# Separate mate TT (store mate distances / scores to avoid confusion with eval TT)
MateTT: Dict[int, Optional[int]] = {}

# Threading event for robust stoppability
stop_event = threading.Event()

# ---------- PIECES & PST (with PHASE SUPPORT) ----------
PV = {}
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Phase-dependent PSTs (White's perspective, mirror for Black)
PST_MG = {
    chess.PAWN: np.array([
        0,   0,   0,   0,   0,   0,   0,   0,
       78,  83,  86,  73, 102,  82,  85,  90,
        7,  29,  21,  44,  40,  31,  44,   7,
      -17,  16,  -2,  15,  14,   0,  15, -13,
      -26,   3,  10,   9,   6,   1,   0, -23,
      -22,   9,   5, -11, -10,  -2,   3, -19,
      -31,   8,  -7, -37, -36, -14,   3, -31,
        0,   0,   0,   0,   0,   0,   0,   0
    ]).reshape(8, 8),

    chess.KNIGHT: np.array([
      -66, -53, -75, -75, -10, -55, -58, -70,
       -3,  -6, 100, -36,   4,  62,  -4, -14,
       10,  67,   1,  74,  73,  27,  62,  -2,
       24,  24,  45,  37,  33,  41,  25,  17,
       -1,   5,  31,  21,  22,  35,   2,   0,
      -18,  10,  13,  22,  18,  15,  11, -14,
      -23, -15,   2,   0,   2,   0, -23, -20,
      -74, -23, -26, -24, -19, -35, -22, -69
    ]).reshape(8, 8),

    chess.BISHOP: np.array([
      -59, -78, -82, -76, -23,-107, -37, -50,
      -11,  20,  35, -42, -39,  31,   2, -22,
       -9,  39, -32,  41,  52, -10,  28, -14,
       25,  17,  20,  34,  26,  25,  15,  10,
       13,  10,  17,  23,  17,  16,   0,   7,
       14,  25,  24,  15,   8,  25,  20,  15,
       19,  20,  11,   6,   7,   6,  20,  16,
       -7,   2, -15, -12, -14, -15, -10, -10
    ]).reshape(8, 8),

    chess.ROOK: np.array([
       35,  29,  33,   4,  37,  33,  56,  50,
       55,  29,  56,  67,  55,  62,  34,  60,
       19,  35,  28,  33,  45,  27,  25,  15,
        0,   5,  16,  13,  18,  -4,  -9,  -6,
      -28, -35, -16, -21, -13, -29, -46, -30,
      -42, -28, -42, -25, -25, -35, -26, -46,
      -53, -38, -31, -26, -29, -43, -44, -53,
      -30, -24, -18,   5,  -2, -18, -31, -32
    ]).reshape(8, 8),

    chess.QUEEN: np.array([
        6,   1,  -8,-104,  69,  24,  88,  26,
       14,  32,  60, -10,  20,  76,  57,  24,
       -2,  43,  32,  60,  72,  63,  43,   2,
        1, -16,  22,  17,  25,  20, -13,  -6,
      -14, -15,  -2,  -5,  -1, -10, -20, -22,
      -30,  -6, -13, -11, -16, -11, -16, -27,
      -36, -18,   0, -19, -15, -15, -21, -38,
      -39, -30, -31, -13, -31, -36, -34, -42
    ]).reshape(8, 8),

    chess.KING: np.array([
        4,  54,  47, -99, -99,  60,  83, -62,
      -32,  10,  55,  56,  56,  55,  10,   3,
      -62,  12, -57,  44, -67,  28,  37, -31,
      -55,  50,  11,  -4, -19,  13,   0, -49,
      -55, -43, -52, -28, -51, -47,  -8, -50,
      -47, -42, -43, -79, -64, -32, -29, -32,
       -4,   3, -14, -50, -57, -18,  13,   4,
       17,  30,  -3, -14,   6,  -1,  40,  18
    ]).reshape(8, 8),
}

PST_EG = {
    chess.PAWN: np.array([
         0,   0,   0,   0,   0,   0,   0,   0,
        80,  90, 100, 110, 110, 100,  90,  80,
        30,  40,  50,  60,  60,  50,  40,  30,
        20,  25,  35,  45,  45,  35,  25,  20,
        10,  15,  20,  30,  30,  20,  15,  10,
        10,  10,  15,  25,  25,  15,  10,  10,
         0,   0,   0,   0,   0,   0,   0,   0,
         0,   0,   0,   0,   0,   0,   0,   0,
    ]).reshape(8, 8),
    chess.KNIGHT: np.array([
        -60, -50, -40, -40, -40, -40, -50, -60,
        -50, -40,   0,   0,   0,   0, -40, -50,
        -40,   0,  10,  15,  15,  10,   0, -40,
        -40,   0,  15,  20,  20,  15,   0, -40,
        -40,   0,  15,  20,  20,  15,   0, -40,
        -40,   0,  10,  15,  15,  10,   0, -40,
        -50, -40,   0,   0,   0,   0, -40, -50,
        -60, -50, -40, -40, -40, -40, -50, -60,
    ]).reshape(8, 8),
    chess.BISHOP: np.array([
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,  10,   0,   0,   0,   0,  10, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ]).reshape(8, 8),
    chess.ROOK: np.array([
         0,   0,   5,  10,  10,   5,   0,   0,
         5,  10,  10,  15,  15,  10,  10,   5,
         0,   0,   5,  10,  10,   5,   0,   0,
         0,   0,   0,   5,   5,   0,   0,   0,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
        -5,   0,   0,   0,   0,   0,   0,  -5,
         0,   0,   0,   0,   0,   0,   0,   0,
    ]).reshape(8, 8),
    chess.QUEEN: np.array([
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   5,   0,   0,   5,   0, -10,
        -10,   5,   5,   5,   5,   5,   5, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ]).reshape(8, 8),
    chess.KING: np.array([
        -10, -10, -10, -10, -10, -10, -10, -10,
         10,  20,  20,  20,  20,  20,  20,  10,
         10,  20,  30,  30,  30,  30,  20,  10,
         10,  20,  30,  40,  40,  30,  20,  10,
         10,  20,  30,  40,  40,  30,  20,  10,
         10,  20,  30,  30,  30,  30,  20,  10,
         10,  10,  10,  10,  10,  10,  10,  10,
         0,   0,   0,   0,   0,   0,   0,   0,
    ]).reshape(8, 8),
}

# Phase weights (similar to Stockfish)
PHASE_WEIGHTS = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
PHASE_MAX = sum(v * 2 for k, v in PHASE_WEIGHTS.items() if k != chess.KING)

def get_phase(board: chess.Board) -> int:
    phase = PHASE_MAX
    for pt in PHASE_WEIGHTS:
        phase -= PHASE_WEIGHTS[pt] * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    return max(0, min(phase, PHASE_MAX))

def interpolate_pst(board: chess.Board, pt: int, sq: int, color: bool) -> int:
    phase = get_phase(board)
    if color == chess.WHITE:
        r, f = chess.square_rank(sq), chess.square_file(sq)
    else:
        mirrored = chess.square_mirror(sq)
        r, f = chess.square_rank(mirrored), chess.square_file(mirrored)
    mg = PST_MG[pt][r, f]
    eg = PST_EG[pt][r, f]
    score = ((mg * (PHASE_MAX - phase)) + (eg * phase)) // PHASE_MAX
    return score

# ---------- UTIL ----------
def now() -> float:
    return time.time()

def timeout() -> bool:
    return stop_event.is_set() or ((time.time() - start_time) >= time_limit)

def within_node_limit() -> bool:
    return (NODE_LIMIT is None) or (node_count < NODE_LIMIT)

def get_key(board: chess.Board) -> int:
    try:
        return board.zobrist_hash()  # type: ignore[attr-defined]
    except AttributeError:
        return hash(board.fen())

# ---------- BITBOARD HELPERS ----------
def lsb(bitboard: int) -> Optional[int]:
    if bitboard == 0:
        return None
    return (bitboard & -bitboard).bit_length() - 1

def pop_lsb(bitboard: int) -> Tuple[Optional[int], int]:
    if bitboard == 0:
        return None, 0
    l = lsb(bitboard)
    return l, bitboard & (bitboard - 1)

# ---------- BITBOARD CACHE (per-position) ----------
class PosBB:
    __slots__ = ("board", "occ", "piece_bb_w", "piece_bb_b", "all_attack_cache")
    def __init__(self, board: chess.Board):
        self.board = board
        self.occ = int(board.occupied)
        self.piece_bb_w = {pt: int(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUE}
        self.piece_bb_b = {pt: int(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUE}
        self.all_attack_cache: Dict[int, int] = {}
    def attackers_bb(self, color: bool, square: int) -> int:
        return int(self.board.attackers(color, square))

# ---------- STATIC EXCHANGE EVALUATION (bitboard-simulated) ----------
# ---------- Corrected see_safe (simulation-based SEE) ----------
_SEE_CACHE: Dict[Tuple[int, int, int], int] = {}  # optional cache: (zkey, from_sq, to_sq) -> see_value

def see_safe(board: chess.Board, move: chess.Move, use_cache: bool = True) -> int:
    """
    Simulation-based Static Exchange Evaluation (SEE).
    Returns centipawn material gain from performing `move`.
    Positive => capture is (materially) winning, Negative => losing.

    Implementation notes:
    - Uses a board.copy() and pushes moves on the copy to correctly simulate captures.
    - Follows classic engine pattern: gains[0] = victim_value, then each attacker append
      attacker_value - gains[-1], then minimax-backpropagate with gains[i] = max(-gains[i+1], gains[i]).
    - Handles en-passant specially.
    - Optional small cache keyed by zobrist (or fen hash fallback) and from/to squares.
    """

    try:
        # cheap exits
        if not board.is_capture(move) and move.promotion is None:
            return 0

        # cache key (if available)
        try:
            zkey = board.zobrist_hash()  # type: ignore[attr-defined]
        except AttributeError:
            zkey = hash(board.fen())

        cache_key = (zkey, move.from_square, move.to_square)
        if use_cache and cache_key in _SEE_CACHE:
            return _SEE_CACHE[cache_key]

        # local alias to your PIECE_VALUE
        PV = PIECE_VALUE

        b = board.copy()  # simulation board

        to_sq = move.to_square
        from_sq = move.from_square
        us = board.turn  # side making the initial move

        # Determine initial victim value (on the original board, but handle en-passant)
        if board.is_en_passant(move):
            # victim is the pawn captured en-passant; its square is behind the to_sq relative to us
            victim_sq = to_sq + (-8 if us == chess.WHITE else 8)
            victim_value = PV[chess.PAWN]
        else:
            victim = board.piece_at(to_sq)
            victim_value = PV.get(victim.piece_type, 0) if victim else 0

        # Prepare gains list: first entry is the value of what we capture immediately
        gains: List[int] = [victim_value]

        # Make the initial move on the copy (this places our piece on to_sq and removes victim)
        b.push(move)

        # side to move next in the simulation (opponent)
        side = not us

        # Main loop: find least valuable attacker for `to_sq`, simulate capture, repeat
        while True:
            attackers = list(b.attackers(side, to_sq))
            if not attackers:
                break

            # choose least valuable attacker (best_sq)
            best_sq = None
            best_val = None
            for a in attackers:
                p = b.piece_at(a)
                if p is None:
                    continue
                v = PV.get(p.piece_type, 0)
                if best_val is None or v < best_val:
                    best_val = v
                    best_sq = a

            if best_sq is None:
                break

            cap_move = chess.Move(best_sq, to_sq)
            # If that capture isn't legal in the simulated board, stop (pins, discovered issues)
            if cap_move not in b.legal_moves:
                break

            # value of the attacker capturing now
            attacker_piece = b.piece_at(best_sq)
            attacker_val = PV.get(attacker_piece.piece_type, 0) if attacker_piece else 0

            # Append attacker value minus last gain (engine pattern)
            gains.append(attacker_val - gains[-1])

            # perform the capture on the simulation board
            b.push(cap_move)

            # alternate side
            side = not side

        # Minimax-style back-propagation to compute net result for the side who started the capture
        for i in range(len(gains) - 2, -1, -1):
            gains[i] = max(-gains[i + 1], gains[i])

        result = int(gains[0] if gains else 0)

        SAFE_MARGIN = 35  # ~⅓ pawn safety margin
        result -= SAFE_MARGIN if result < 0 else 0

# Clamp absurd values (e.g., rare simulation anomalies)
        if result > 9000 or result < -9000:
            result = 0

        if use_cache:
            _SEE_CACHE[cache_key] = result

        return result

    except Exception:
        # on any unexpected issue, return 0 (safe fallback)
        return 0

# ---------- Sliding attack helpers ----------
def ray_attacks(square: int, deltas: list[int], occ: int) -> int:
    attacks = 0
    for d in deltas:
        sq = square
        while True:
            prev_file = sq & 7      # faster than sq % 8
            sq += d
            if sq < 0 or sq > 63:
                break
            new_file = sq & 7

            # Prevent horizontal wrap (file must move exactly ±1 on diagonals, or 0 on vertical)
            if abs(new_file - prev_file) > 1:
                break

            attacks |= 1 << sq

            # Stop if blocker encountered
            if occ & (1 << sq):
                break
    return attacks

def bishop_attacks(square, occ):
    return ray_attacks(square, [9, 7, -9, -7], occ)

def rook_attacks(square, occ):
    return ray_attacks(square, [8, -8, 1, -1], occ)

def queen_attacks(square, occ):
    return bishop_attacks(square, occ) | rook_attacks(square, occ)

# ---------- SEE function ----------
def see(board: chess.Board, move: chess.Move) -> int:
    if not board.is_capture(move) and not move.promotion:
        return 0

    side = board.turn
    from_sq = move.from_square
    to_sq = move.to_square

    occ = int(board.occupied)

    pieces = {
        chess.WHITE: {pt: int(board.pieces(pt, chess.WHITE)) for pt in PIECE_VALUE},
        chess.BLACK: {pt: int(board.pieces(pt, chess.BLACK)) for pt in PIECE_VALUE},
    }

    # Victim square and value
    if board.is_en_passant(move):
        victim_sq = to_sq + (-8 if side == chess.WHITE else 8)
        victim_value = PIECE_VALUE[chess.PAWN]
        pieces[not side][chess.PAWN] &= ~(1 << victim_sq)
        occ &= ~(1 << victim_sq)
    else:
        victim_sq = to_sq
        victim = board.piece_at(to_sq)
        victim_value = PIECE_VALUE.get(victim.piece_type, 0) if victim else 0

    # Remove moving piece
    moving_piece = board.piece_at(from_sq)
    if moving_piece is None:
        return 0  # Invalid move
    moving_pt = moving_piece.piece_type
    moving_color = moving_piece.color
    pieces[moving_color][moving_pt] &= ~(1 << from_sq)
    occ &= ~(1 << from_sq)
    occ |= (1 << to_sq)

    gains = [victim_value]
    stm = not side  # side to move next

    # Pawn attack helper
    def pawn_attackers(square, color, pawns):
        if color == chess.WHITE:
            return pawns & chess.BB_PAWN_ATTACKS[chess.BLACK][square]
        else:
            return pawns & chess.BB_PAWN_ATTACKS[chess.WHITE][square]

    # All attackers
    def all_attackers(color, occ):
        att = 0
        att |= pawn_attackers(to_sq, color, pieces[color][chess.PAWN])
        att |= chess.BB_KNIGHT_ATTACKS[to_sq] & pieces[color][chess.KNIGHT]
        att |= bishop_attacks(to_sq, occ) & pieces[color][chess.BISHOP]
        att |= rook_attacks(to_sq, occ) & pieces[color][chess.ROOK]
        att |= queen_attacks(to_sq, occ) & pieces[color][chess.QUEEN]
        att |= chess.BB_KING_ATTACKS[to_sq] & pieces[color][chess.KING]
        return att

    # Main SEE loop
    while True:
        atks = all_attackers(stm, occ)
        if not atks:
            break
        # Least valuable attacker
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            bb = pieces[stm][pt] & atks
            if bb:
                sq = (bb & -bb).bit_length() - 1
                pieces[stm][pt] &= ~(1 << sq)
                occ &= ~(1 << sq)
                gains.append(PIECE_VALUE[pt] - gains[-1])
                stm = not stm
                break
        else:
            break

    # Negamax propagation
    for i in range(len(gains) - 2, -1, -1):
        gains[i] = max(-gains[i + 1], gains[i])

    return gains[0] if gains else 0


# ---------- MOVE ORDERING ----------
def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    """
    Modern strongest MVV-LVA:
    - Scaled correctly (small, fast, predictable)
    - Only used for captures
    """
    if not board.is_capture(move):
        return 0

    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    if not victim or not attacker:
        return 0

    # Standard +10 scale for stability
    return PIECE_VALUE[victim.piece_type] * 10 - PIECE_VALUE[attacker.piece_type]


def move_score(board: chess.Board, move: chess.Move,
               pv_move: Optional[chess.Move], ply: int) -> int:
    """
    Strongest move ordering hierarchy:
    1. PV move (absolute top)
    2. Captures (MVV-LVA + SEE penalty for losers)
    3. Promotions
    4. Killer moves (slot 1 > slot 2)
    5. History heuristic
    """
    # ----- PV move (instantly top) -----
    if pv_move and move == pv_move:
        return 100_000_000

    score = 0

    # ----- Captures -----
    if board.is_capture(move):
        score += 20_000 + mvv_lva_score(board, move)

        # SEE: ONLY push *losing* captures down (top engine technique)
        if see(board, move) < 0:
            score -= 10_000

    # ----- Promotions -----
    if move.promotion:
        score += 40_000

    # ----- Killer moves -----
    k1, k2 = KILLERS.get(ply, (None, None))
    if move == k1:
        score += 10_000
    elif move == k2:
        score += 5_000

    # ----- History heuristic -----
    score += HISTORY.get((move.from_square, move.to_square), 0)

    return score


def order_moves(board: chess.Board, moves: List[chess.Move],
                pv_move: Optional[chess.Move], ply: int) -> List[chess.Move]:
    """
    Strongest ordering:
    - Full hierarchical scoring
    - Random tie-breaking (optional)
    """
    scored = []

    for move in moves:
        base = move_score(board, move, pv_move, ply)

        if RANDOM_TIE:
            # tiny noise injected in lowest bits
            base = (base << 8) + random.randint(0, 255)

        scored.append((base, move))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [m for (_, m) in scored]

# ---------- EVALUATION ----------
# Global pawn hash table to cache pawn structure evaluations
# --- EVALUATION (replacement) ---
PAWN_HASH = {}  # simple cache; consider bounded LRU for long games

# Tuned constants (centipawns)
PASSED_PAWN_BASE = 25
PASSED_PAWN_RANK_BONUS = 10     # per rank advanced
ISOLATED_PAWN_PENALTY = -20
DOUBLED_PAWN_PENALTY = -15
PAWN_CHAIN_BONUS = 8
BLOCKADED_PAWN_PENALTY = -18

POS_SCALE = 0.45
MOBILITY_SCALE = 0.22
KING_SAFETY_SCALE = 0.9
SPACE_SCALE = 0.55
CHECK_PENALTY = 40

def pawn_structure_hash(board: chess.Board) -> int:
    """
    Fast, readable pawn structure evaluator with caching.
    Returns a centipawn value (positive = White advantage).
    """
    # key = (white_pawns_bitboard_int, black_pawns_bitboard_int)
    key = (int(board.pieces(chess.PAWN, chess.WHITE)), int(board.pieces(chess.PAWN, chess.BLACK)))
    if key in PAWN_HASH:
        return PAWN_HASH[key]

    white_pawns = set(board.pieces(chess.PAWN, chess.WHITE))
    black_pawns = set(board.pieces(chess.PAWN, chess.BLACK))

    score = 0

    def is_passed(square: int, color: int) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if color == chess.WHITE:
            # any black pawn on same file or adjacent files ahead cancels passed status
            for f in range(max(0, file - 1), min(7, file + 1) + 1):
                for r in range(rank + 1, 8):
                    if chess.square(f, r) in black_pawns:
                        return False
            return True
        else:
            for f in range(max(0, file - 1), min(7, file + 1) + 1):
                for r in range(rank - 1, -1, -1):
                    if chess.square(f, r) in white_pawns:
                        return False
            return True

    def is_isolated(square: int, color: int) -> bool:
        file = chess.square_file(square)
        target_set = white_pawns if color == chess.WHITE else black_pawns
        for df in (-1, 1):
            nf = file + df
            if 0 <= nf < 8:
                for r in range(8):
                    if chess.square(nf, r) in target_set:
                        return False
        return True

    def is_doubled(square: int, color: int) -> bool:
        file = chess.square_file(square)
        target_set = white_pawns if color == chess.WHITE else black_pawns
        cnt = 0
        for r in range(8):
            if chess.square(file, r) in target_set:
                cnt += 1
                if cnt > 1:
                    return True
        return False

    def pawn_chain_bonus(square: int, color: int) -> int:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        target_set = white_pawns if color == chess.WHITE else black_pawns
        if color == chess.WHITE:
            nr = rank - 1
            for df in (-1, 1):
                nf = file + df
                if 0 <= nf < 8 and 0 <= nr < 8:
                    if chess.square(nf, nr) in target_set:
                        return PAWN_CHAIN_BONUS
        else:
            nr = rank + 1
            for df in (-1, 1):
                nf = file + df
                if 0 <= nf < 8 and 0 <= nr < 8:
                    if chess.square(nf, nr) in target_set:
                        return PAWN_CHAIN_BONUS
        return 0

    def is_blockaded(square: int, color: int) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if color == chess.WHITE:
            if rank < 7:
                sq = chess.square(file, rank + 1)
                p = board.piece_at(sq)
                return p is not None and p.color != chess.WHITE
            return False
        else:
            if rank > 0:
                sq = chess.square(file, rank - 1)
                p = board.piece_at(sq)
                return p is not None and p.color != chess.BLACK
            return False

    # Evaluate white pawns
    for sq in white_pawns:
        r = chess.square_rank(sq)
        if is_passed(sq, chess.WHITE):
            score += PASSED_PAWN_BASE + PASSED_PAWN_RANK_BONUS * (r - 1)
        if is_isolated(sq, chess.WHITE):
            score += ISOLATED_PAWN_PENALTY
        if is_doubled(sq, chess.WHITE):
            score += DOUBLED_PAWN_PENALTY
        score += pawn_chain_bonus(sq, chess.WHITE)
        if is_blockaded(sq, chess.WHITE):
            score += BLOCKADED_PAWN_PENALTY

    # Evaluate black pawns (subtract from white score)
    for sq in black_pawns:
        r = 7 - chess.square_rank(sq)  # black advancement from their perspective
        if is_passed(sq, chess.BLACK):
            score -= PASSED_PAWN_BASE + PASSED_PAWN_RANK_BONUS * (r - 1)
        if is_isolated(sq, chess.BLACK):
            score -= ISOLATED_PAWN_PENALTY * 0.9   # slightly different scaling ok
        if is_doubled(sq, chess.BLACK):
            score -= DOUBLED_PAWN_PENALTY * 0.9
        score -= pawn_chain_bonus(sq, chess.BLACK)
        if is_blockaded(sq, chess.BLACK):
            score -= BLOCKADED_PAWN_PENALTY * 0.9

    # store cached integer
    PAWN_HASH[key] = int(score)
    return int(score)


def is_outpost(board: chess.Board, sq: int, color: int) -> bool:
    """
    Simple outpost test:
    - square must be on enemy half (rank >= 3 for white, <= 4 for black)
    - no enemy pawns on adjacent files in forward ranks (so piece cannot be easily kicked)
    """
    rank = chess.square_rank(sq)
    file = chess.square_file(sq)
    if color == chess.WHITE:
        if rank < 3:
            return False
        for df in (-1, 1):
            f = file + df
            if 0 <= f < 8:
                for r in range(rank, 8):
                    if chess.square(f, r) in board.pieces(chess.PAWN, chess.BLACK):
                        return False
        return True
    else:
        if rank > 4:
            return False
        for df in (-1, 1):
            f = file + df
            if 0 <= f < 8:
                for r in range(rank, -1, -1):
                    if chess.square(f, r) in board.pieces(chess.PAWN, chess.WHITE):
                        return False
        return True

def is_open_file(board: chess.Board, file: int) -> bool:
    """Open file: no pawns of either color on that file."""
    for r in range(8):
        sq = chess.square(file, r)
        if (
            sq in board.pieces(chess.PAWN, chess.WHITE)
            or sq in board.pieces(chess.PAWN, chess.BLACK)
        ):
            return False
    return True


def is_semi_open_file(board: chess.Board, file: int, color: chess.Color) -> bool:
    """
    Semi-open file:
    No pawns of *this color* on the file.
    """
    for r in range(8):
        sq = chess.square(file, r)
        if sq in board.pieces(chess.PAWN, color):
            return False
    return True

def evaluate(board: chess.Board) -> int:
    """
    Main evaluation function.
    Returns centipawns from White's perspective (positive = White advantage).
    """
    score = 0.0

    # Material + PST
    for pt, val in PIECE_VALUE.items():
        w_sqs = list(board.pieces(pt, chess.WHITE))
        b_sqs = list(board.pieces(pt, chess.BLACK))
        score += val * (len(w_sqs) - len(b_sqs))

        # per-piece PST scaled by POS_SCALE
        for s in w_sqs:
            score += interpolate_pst(board, pt, s, chess.WHITE) * POS_SCALE
        for s in b_sqs:
            score -= interpolate_pst(board, pt, s, chess.BLACK) * POS_SCALE

    # Mobility (piece attacks) — weighted modestly
    mobility_white = 0
    mobility_black = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s in board.pieces(pt, chess.WHITE):
            mobility_white += len(board.attacks(s))
        for s in board.pieces(pt, chess.BLACK):
            mobility_black += len(board.attacks(s))
    score += (mobility_white - mobility_black) * 2 * MOBILITY_SCALE

    # Legal moves/activity (small)
    legal_count = len(list(board.legal_moves))
    # add small bonus toward side with more legal moves (positive = White)
    if board.turn == chess.WHITE:
        score += (legal_count // 2) * POS_SCALE
    else:
        score -= (legal_count // 2) * POS_SCALE

    # Center control (explicit points for pieces in central squares)
    center_sqs = (chess.E4, chess.D4, chess.E5, chess.D5)
    for c in center_sqs:
        p = board.piece_at(c)
        if p:
            score += (15 if p.color == chess.WHITE else -15) * POS_SCALE

    # Bishop pair
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 35
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 35

    # Outpost bonuses
    for s in board.pieces(chess.KNIGHT, chess.WHITE):
        if is_outpost(board, s, chess.WHITE):
            score += 18 * POS_SCALE
    for s in board.pieces(chess.BISHOP, chess.WHITE):
        if is_outpost(board, s, chess.WHITE):
            score += 12 * POS_SCALE
    for s in board.pieces(chess.KNIGHT, chess.BLACK):
        if is_outpost(board, s, chess.BLACK):
            score -= 18 * POS_SCALE
    for s in board.pieces(chess.BISHOP, chess.BLACK):
        if is_outpost(board, s, chess.BLACK):
            score -= 12 * POS_SCALE

    # Pawn structure (cached)
    score += pawn_structure_hash(board)

    # Rook activity (rank and file)
    for s in board.pieces(chess.ROOK, chess.WHITE):
        rank = chess.square_rank(s)
        file = chess.square_file(s)
        if rank >= 6:
            score += 22 * POS_SCALE
        if is_open_file(board, file):
            score += 18 * POS_SCALE
        elif is_semi_open_file(board, file, chess.WHITE):
            score += 9 * POS_SCALE
    for s in board.pieces(chess.ROOK, chess.BLACK):
        rank = chess.square_rank(s)
        file = chess.square_file(s)
        if rank <= 1:
            score -= 22 * POS_SCALE
        if is_open_file(board, file):
            score -= 18 * POS_SCALE
        elif is_semi_open_file(board, file, chess.BLACK):
            score -= 9 * POS_SCALE

    # Space: count advanced major/minor pieces into opponent half
    space_white = 0
    space_black = 0
    for s in board.pieces(chess.KNIGHT, chess.WHITE) | board.pieces(chess.BISHOP, chess.WHITE) | board.pieces(chess.ROOK, chess.WHITE) | board.pieces(chess.QUEEN, chess.WHITE):
        if chess.square_rank(s) >= 4:
            space_white += 1
    for s in board.pieces(chess.KNIGHT, chess.BLACK) | board.pieces(chess.BISHOP, chess.BLACK) | board.pieces(chess.ROOK, chess.BLACK) | board.pieces(chess.QUEEN, chess.BLACK):
        if chess.square_rank(s) <= 3:
            space_black += 1
    score += 12 * (space_white - space_black) * SPACE_SCALE

    # King safety: simple shield + nearby attackers
    phase = get_phase(board) if 'get_phase' in globals() else 0
    for color in (chess.WHITE, chess.BLACK):
        king_sqs = list(board.pieces(chess.KING, color))
        if not king_sqs:
            continue
        ksq = king_sqs[0]
        kr = chess.square_rank(ksq)
        kf = chess.square_file(ksq)
        shield = 0
        open_files = 0
        # only evaluate shield for castled/early phases
        if (color == chess.WHITE and phase < PHASE_MAX // 2) or (color == chess.BLACK and phase < PHASE_MAX // 2):
            for df in (-1, 0, 1):
                f = kf + df
                if 0 <= f < 8:
                    if color == chess.WHITE:
                        pawn_sq = chess.square(f, kr + 1) if kr + 1 < 8 else None
                        if pawn_sq and pawn_sq in board.pieces(chess.PAWN, chess.WHITE):
                            shield += 1
                        if not any(chess.square(f, r) in board.pieces(chess.PAWN, chess.WHITE) for r in range(8)):
                            open_files += 1
                    else:
                        pawn_sq = chess.square(f, kr - 1) if kr - 1 >= 0 else None
                        if pawn_sq and pawn_sq in board.pieces(chess.PAWN, chess.BLACK):
                            shield += 1
                        if not any(chess.square(f, r) in board.pieces(chess.PAWN, chess.BLACK) for r in range(8)):
                            open_files += 1
        penalty = (3 - shield) * 18 + open_files * 12
        attackers = 0
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if df == 0 and dr == 0:
                    continue
                f = kf + df
                r = kr + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    if board.is_attacked_by(not color, chess.square(f, r)):
                        attackers += 1
        penalty += attackers * 10
        penalty *= KING_SAFETY_SCALE
        if color == chess.WHITE:
            score -= penalty
        else:
            score += penalty

    # Check penalty (small)
    if board.is_check():
        score += (-CHECK_PENALTY if board.turn == chess.WHITE else CHECK_PENALTY)

    # final integer centipawn result (White perspective)
    return int(score if board.turn == chess.WHITE else -score)
# ---------- QUIESCENCE ----------
def quiescence(board: chess.Board, alpha: int, beta: int, depth: int = 0) -> int:
    """2-ply tactical quiescence search with SEE pruning and quiet tactical extensions."""
    global node_count
    if timeout():
        return evaluate(board)
    node_count += 1
    LOSS_THRESHOLD = 350
    REPETITION_PENALTY = 30

    if board.is_repetition(3) or board.can_claim_threefold_repetition() or board.is_fivefold_repetition():
        stand = evaluate(board)
        if stand <= -LOSS_THRESHOLD:
            return 0
        else:
            return -REPETITION_PENALTY

    if board.can_claim_fifty_moves():
        stand = evaluate(board)
        if stand <= -LOSS_THRESHOLD:
            return 0
        else:
            return -REPETITION_PENALTY

    # --- Stand-pat evaluation ---
    stand = evaluate(board)
    if stand >= beta:
        return beta
    if alpha < stand:
        alpha = stand

    # --- Delta pruning ---
    if stand + FUTILITY_MARGIN < alpha and not board.is_check():
        return alpha

    moves: List[chess.Move] = []

    # --- Generate tactical moves ---
    for m in board.legal_moves:
        if board.is_capture(m):
            try:
                s = see_safe(board, m)
            except Exception:
                s = 0
            if s < -70:
                continue
            moves.append(m)

        elif m.promotion == chess.QUEEN:
            moves.append(m)

        elif board.gives_check(m):
            moves.append(m)

        # quiet tactical extensions (forks / discoveries)
        elif depth < 2:  # only do this in first layer, prevents explosion
            try:
                board.push(m)
                delta_eval = evaluate(board) - stand
                board.pop()
                if delta_eval > 80:
                    moves.append(m)
            except Exception:
                continue

    if not moves:
        return stand

    # --- Order moves (MVV-LVA + SEE) ---
    if len(moves) > 1:
        moves = order_moves(board, moves, None, 0)

    # --- Explore moves ---
    for m in moves:
        board.push(m)
        try:
            # two-ply tactical lookahead
            if depth < 1:
                val = -quiescence(board, -beta, -alpha, depth + 1)
            else:
                # after first ply, only extend one more tactical layer
                val = -evaluate(board)
        finally:
            board.pop()

        if val >= beta:
            return beta
        if val > alpha:
            alpha = val

    return alpha
   
# ---------- TRANSPO TABLE ----------
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
TT_SIZE_LIMIT = 2_000_000

def tt_lookup(key: int, depth: int, alpha: int, beta: int) -> Optional[int]:
    """
    Backwards-compatible probe: returns a score if the TT entry provides an immediate usable cutoff,
    otherwise returns None. Does not mutate globals.
    """
    e = TT.get(key)
    if not e:
        return None
    # entry too shallow to be used for this probe
    if e.depth < depth:
        return None

    if e.flag == TT_EXACT:
        return e.score
    if e.flag == TT_LOWER and e.score >= beta:
        # lower bound >= beta -> cutoff
        return e.score
    if e.flag == TT_UPPER and e.score <= alpha:
        # upper bound <= alpha -> cutoff
        return e.score
    return None


def tt_lookup_entry(key: int) -> Optional[TTEntry]:
    """Return the raw TTEntry (or None). Caller can inspect depth/flag/best as needed."""
    return TT.get(key)


def tt_store(key: int, depth: int, score: int, flag: int, best: Optional[chess.Move]) -> None:
    """
    Replacement policy:
      - Never overwrite a deeper entry from the same TT_AGE with a shallower one.
      - Allow replacing older entries (different age) with similar-depth entries.
      - Insert if missing.
      - Simple eviction when TT grows over TT_SIZE_LIMIT: evict some old-age entries first.
    """
    global TT_AGE, TT

    e = TT.get(key)
    if e is None:
        TT[key] = TTEntry(key=key, depth=depth, score=score, flag=flag, best=best, age=TT_AGE)
        # enforce size limit (simple policy)
        if len(TT) > TT_SIZE_LIMIT:
            # prefer removing old entries (age != TT_AGE)
            to_remove = [k for k, ent in TT.items() if ent.age != TT_AGE]
            if not to_remove:
                # fallback: remove a small fraction (arbitrary)
                it = iter(TT)
                to_remove = [next(it)]
            for k in to_remove[:max(1, len(TT)//50)]:
                TT.pop(k, None)
        return

    # don't overwrite a deeper entry from the same generation
    if depth < e.depth and e.age == TT_AGE:
        return

    # allow replacement if:
    #  - depth >= e.depth (new is as deep or deeper)
    #  - OR existing entry is old (age differs) and new depth is not much smaller
    if depth < e.depth and e.age != TT_AGE:
        if depth < e.depth - 1:
            # new entry too shallow compared to old (even if old is old-age)
            return

    # replace
    TT[key] = TTEntry(key=key, depth=depth, score=score, flag=flag, best=best, age=TT_AGE)

# ---------- ALPHA-BETA (full search) ----------
def alpha_beta(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    is_pv: bool = False,
    excluded_move: Optional[chess.Move] = None,
) -> int:
    """
    Negamax-style alpha-beta with:
      - TT probe (EXACT/LOWER/UPPER)
      - repetition / fifty-move handling using side-to-move evaluation
      - null-move pruning
      - LMR (reduction)
      - killer & history updates on cutoffs
    Assumes evaluate(board) returns centipawns FROM WHITE'S PERSPECTIVE.
    """
    global node_count, TT_AGE

    # timeout handling preserved
    if timeout():
        return evaluate(board)

    node_count += 1

    key = get_key(board)
    ttent = TT.get(key)

    orig_alpha = alpha
    orig_beta = beta

    # === TT probe ===
    if ttent and ttent.depth >= depth:
        # ttent.score is assumed to be in same convention (White-perspective)
        if ttent.flag == TT_EXACT:
            return ttent.score
        if ttent.flag == TT_LOWER:
            if ttent.score >= beta:
                return ttent.score
            alpha = max(alpha, ttent.score)
        elif ttent.flag == TT_UPPER:
            if ttent.score <= alpha:
                return ttent.score
            beta = min(beta, ttent.score)
        # keep ttent for PV move ordering later

    # --- Repetition / 50-move handling ---
    # Use evaluation from SIDE-TO-MOVE perspective for decision
    if board.is_repetition() or board.can_claim_threefold_repetition() or board.is_fivefold_repetition():
        stand = evaluate(board)  # white-perspective
        stand_smt = stand if board.turn == chess.WHITE else -stand
        # If side-to-move is losing badly, accept draw (return 0)
        if stand_smt <= -LOSS_THRESHOLD:
            return 0
        # Otherwise slightly penalize repetition to avoid taking it when winning
        return -REPETITION_PENALTY

    if board.can_claim_fifty_moves() or board.is_fivefold_repetition():
        stand = evaluate(board)
        stand_smt = stand if board.turn == chess.WHITE else -stand
        if stand_smt <= -LOSS_THRESHOLD:
            return 0
        return -REPETITION_PENALTY

    # Terminal checks
    if board.is_checkmate():
        return -MATE_VALUE + ply
    if board.is_stalemate():
        return 0

    # Quiescence base
    if depth <= 0:
        return quiescence(board, alpha, beta)

    # PV move (only use if tt entry deep enough)
    pvmove = ttent.best if (ttent and getattr(ttent, "best", None) and ttent.depth >= depth) else None

    # Null-move reduction (negamax style)
    if (not is_pv) and depth >= 3 and not board.is_check() and not board.can_claim_draw():
        board.push(chess.Move.null())
        try:
            val = -alpha_beta(board, depth - 1 - NULL_REDUCTION, -beta, -beta + 1, ply + 1, False, None)
            if val >= beta:
                # successful null-move cutoff => return beta (fail-high)
                return beta
        finally:
            board.pop()

    moves = list(board.legal_moves)
    if not moves:
        return 0

    moves = order_moves(board, moves, pvmove, ply)

    best_score = -INFTY
    best_move: Optional[chess.Move] = None
    first = True

    for i, mv in enumerate(moves):
        if excluded_move and mv == excluded_move:
            continue

        # LMR calculation for non-first, quiet, non-check, non-promo moves
        reduction = 0
        if (not first) and depth >= 3 and (not board.is_capture(mv)) and (not board.gives_check(mv)) and (not mv.promotion):
            reduction = int(LMR_BASE + math.log(max(depth, 2)) / LMR_DIV + math.log(i + 1) / LMR_DIV)
            reduction = max(0, min(reduction, depth - 2))

        board.push(mv)
        try:
            if first:
                val = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, True, None)
            else:
                try_depth = depth - 1 - reduction
                if try_depth < 0:
                    try_depth = 0
                # null-window search
                val = -alpha_beta(board, try_depth, -alpha - 1, -alpha, ply + 1, True, None)
                # if it fails high in null-window, do full search
                if alpha < val < beta:
                    val = -alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, True, None)
        finally:
            board.pop()

        first = False

        if val > best_score:
            best_score = val
            best_move = mv

        if val > alpha:
            alpha = val

        if alpha >= beta:
            # cutoff — update killer/history
            if not board.is_capture(mv):
                km = KILLERS.setdefault(ply, [None, None])
                if km[0] != mv:
                    km[1] = km[0]
                    km[0] = mv
            HISTORY[(mv.from_square, mv.to_square)] = HISTORY.get((mv.from_square, mv.to_square), 0) + depth * depth

            # store cutoff as LOWER bound
            tt_store(key, depth, val, TT_LOWER, mv)
            return val

    # No cutoff — decide final TT flag using original bounds
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= orig_beta:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    tt_store(key, depth, best_score, flag, best_move)
    return best_score
# ---------- MATE-ONLY SEARCH (fast focused solver) ----------
def mate_dfs(board: chess.Board, depth: int, ply: int) -> Optional[int]:
    global node_count
    if timeout():
        # your existing pattern raised TimeoutError in mate_dfs; keep that
        raise TimeoutError
    node_count += 1

    key = get_key(board) ^ depth
    if key in MateTT:
        return MateTT[key]

    if board.is_checkmate() or board.is_stalemate() or depth == 0:
        MateTT[key] = None
        return None

    moves = list(board.legal_moves)
    checks = [m for m in moves if board.gives_check(m)]
    captures = [m for m in moves if board.is_capture(m) and m not in checks]
    others = [m for m in moves if (m not in checks and m not in captures)]
    ordered = checks + captures + others

    for m in ordered:
        board.push(m)
        try:
            if board.is_checkmate():
                mate_score = MATE_VALUE - ply
                MateTT[key] = mate_score
                return mate_score

            opp_has_escape = False
            opp_moves = list(board.legal_moves)
            opp_checks = [om for om in opp_moves if board.gives_check(om)]
            opp_captures = [om for om in opp_moves if board.is_capture(om) and om not in opp_checks]
            opp_others = [om for om in opp_moves if (om not in opp_checks and om not in opp_captures)]
            opp_ordered = opp_checks + opp_captures + opp_others

            for om in opp_ordered:
                board.push(om)
                try:
                    res = mate_dfs(board, depth - 2, ply + 2)
                finally:
                    board.pop()
                if res is None:
                    opp_has_escape = True
                    break

            if not opp_has_escape:
                mate_score = MATE_VALUE - ply
                MateTT[key] = mate_score
                return mate_score
        finally:
            # pop the move m (safe single pop)
            if board.move_stack and board.move_stack[-1] == m:
                board.pop()

    MateTT[key] = None
    return None
def mate_search_root(board: chess.Board, max_mate_ply: int, time_limit_s: float) -> Optional[List[chess.Move]]:
    global start_time, time_limit, node_count, MateTT
    node_count = 0
    MateTT = {}
    start_time = time.time()
    time_limit = time_limit_s
    stop_event.clear()

    for depth in range(1, max_mate_ply + 1):
        if timeout():
            break
        try:
            res = mate_dfs(board, depth, 1)
        except TimeoutError:
            break
        if res is not None:
            pv: List[chess.Move] = []
            b = board.copy()
            ply = 1
            while True:
                if timeout():
                    break
                moves = list(b.legal_moves)
                moves_ord = sorted(moves, key=lambda m: move_score(b, m, None, ply), reverse=True)
                found = False
                for m in moves_ord:
                    b.push(m)
                    try:
                        sat = mate_dfs(b, depth - ply + 1, ply + 1)
                    finally:
                        if b.move_stack:
                            b.pop()
                    if sat is not None:
                        pv.append(m)
                        b.push(m)
                        found = True
                        break
                if not found:
                    break
                if b.is_checkmate():
                    break
                ply += 1
            return pv
    return None

# ---------- ROOT SEARCH & ITERATIVE DEEPENING (full search) ----------

def extract_pv(board: chess.Board, depth_limit: int) -> List[chess.Move]:
    pv = []
    seen = set()   # avoid infinite cycles
    b = board.copy()

    for _ in range(depth_limit):
        key = get_key(b)
        e = TT.get(key)

        if not e or not e.best:
            break
        mv = e.best

        # Prevent illegal or looping PV
        if mv not in b.legal_moves or (b.turn, mv) in seen:
            break

        seen.add((b.turn, mv))
        pv.append(mv)
        b.push(mv)

    return pv
  
def root_search(board, max_depth, movetime=None, nodes_limit=None, multipv=1):
    global node_count, start_time, time_limit, NODE_LIMIT, TT_AGE

    node_count = 0
    TT_AGE = (TT_AGE + 1) % 256
    stop_event.clear()

    start_time = time.time()
    time_limit = movetime or DEFAULT_MOVE_TIME
    NODE_LIMIT = nodes_limit

    multipv = max(1, min(MULTIPV_MAX, multipv))

    multipv_results = []   # canonical: list of (score, pv_line)

    # ---- ITERATIVE DEEPENING ----
    for depth in range(1, max_depth + 1):
        if timeout():
            break

        scored_moves = []

        # Order root moves
        tt_entry = TT.get(get_key(board))
        tt_best = tt_entry.best if tt_entry else None
        moves = order_moves(board, list(board.legal_moves), tt_best, ply=0)

        # Score every move
        for mv in moves:
            if timeout():
                break

            board.push(mv)
            score = -alpha_beta(board, depth - 1, -MATE_VALUE, MATE_VALUE,
                                ply=1, is_pv=True, excluded_move=None)
            board.pop()

            scored_moves.append((mv, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)

        # Only top multipv moves
        top_moves = scored_moves[:multipv]

        multipv_results = []
        for mv, score in top_moves:
            board.push(mv)
            pv_line = [mv] + extract_pv(board, depth_limit=depth)
            board.pop()

            # ✨ CORRECTED FORMAT
            multipv_results.append((score, pv_line))

        # Mate found → stop early
        if multipv_results and multipv_results[0][0] >= MATE_VALUE - 1000:
            break

    # Determine best move safely
    if multipv_results:
        best_move = multipv_results[0][1][0]  # the first PV move
    else:
        try:
            best_move = next(iter(board.legal_moves))
        except StopIteration:
            best_move = None

    return best_move, multipv_results
# ---------- UCI LOOP ----------
def uci_loop() -> None:
    board = chess.Board()
    multipv = 1
    threads = 1
    hash_sz = 32
    random.seed(0xC0FFEE)
    search_thread = None
    search_result = {}
    global node_count, start_time, time_limit, TT_AGE, NODE_LIMIT

    def start_search(go_args):
        command, args = go_args
        if command == "mate":
            mate_n, movetime = args
            pv = mate_search_root(board, mate_n * 2, movetime)
            search_result['pv'] = pv
        else:
            maxd, tlim, nodes, mpv = args
            best, multipv_list = root_search(board, maxd, movetime=tlim, nodes_limit=nodes, multipv=mpv)
            search_result['best'] = best
            search_result['multipv_list'] = multipv_list

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "uci":
            print("id name Supranova-v1.0.27")
            print("id author Supra")
            print("option name Hash type spin default 32 min 1 max 4096")
            print("option name Threads type spin default 1 min 1 max 8")
            print("option name Multipv type spin default 1 min 1 max 4")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("setoption"):
            toks = line.split()
            if "name" in toks:
                try:
                    ni = toks.index("name") + 1
                    if "value" in toks:
                        vi = toks.index("value")
                        name = " ".join(toks[ni:vi])
                        value = " ".join(toks[vi + 1 :])
                    else:
                        name = " ".join(toks[ni:])
                        value = ""
                    if name.lower() == "multipv":
                        try:
                            multipv = max(1, min(MULTIPV_MAX, int(value)))
                        except Exception:
                            pass
                except Exception:
                    pass
        elif line.startswith("position"):
            toks = line.split()
            if "startpos" in toks:
                board.reset()
                if "moves" in toks:
                    for mv in toks[toks.index("moves") + 1 :]:
                        board.push_uci(mv)
            elif "fen" in toks:
                i = toks.index("fen")
                fen = " ".join(toks[i + 1 : i + 7])
                board.set_fen(fen)
                if "moves" in toks:
                    for mv in toks[toks.index("moves") + 1 :]:
                        board.push_uci(mv)
        elif line.startswith("go"):
            # Kill any previous search
            if search_thread and search_thread.is_alive():
                stop_event.set()
                search_thread.join(timeout=1.0)
            stop_event.clear()
            toks = line.split()
            if "mate" in toks:
                try:
                    mate_idx = toks.index("mate")
                    mate_n = int(toks[mate_idx + 1])
                except Exception:
                    mate_n = 30
                movetime = None
                if "movetime" in toks:
                    try:
                        movetime = int(toks[toks.index("movetime") + 1]) / 1000.0
                    except Exception:
                        movetime = None
                tlim = movetime if movetime else 5.0
                search_result.clear()
                search_thread = threading.Thread(target=start_search, args=(("mate", (mate_n, tlim)),))
                search_thread.start()
                while search_thread.is_alive():
                    try:
                        search_thread.join(timeout=0.1)
                    except KeyboardInterrupt:
                        stop_event.set()
                pv = search_result.get('pv')
                if pv:
                    pv_str = " ".join(m.uci() for m in pv)
                    print(f"info score mate {mate_n} nodes {node_count} pv {pv_str}")
                    print("bestmove", pv[0].uci())
                else:
                    print("info string no mate found")
                    best, _ = root_search(board, 6, movetime=1.0, nodes_limit=None, multipv=multipv)
                    if best:
                        print("bestmove", best.uci())
                    else:
                        legal = list(board.legal_moves)
                        if legal:
                            print("bestmove", legal[0].uci())
                        else:
                            print("bestmove 0000")
                sys.stdout.flush()
                continue

            wtime = btime = movetime = depth = nodes = None
            infinite = False
            if "wtime" in toks:
                try:
                    wtime = int(toks[toks.index("wtime") + 1])
                except Exception:
                    pass
            if "btime" in toks:
                try:
                    btime = int(toks[toks.index("btime") + 1])
                except Exception:
                    pass
            if "movetime" in toks:
                try:
                    movetime = int(toks[toks.index("movetime") + 1]) / 1000.0
                except Exception:
                    pass
            if "depth" in toks:
                try:
                    depth = int(toks[toks.index("depth") + 1])
                except Exception:
                    pass
            if "nodes" in toks:
                try:
                    nodes = int(toks[toks.index("nodes") + 1])
                except Exception:
                    pass
            if "infinite" in toks:
                infinite = True

            if movetime:
                tlim = movetime
            elif (wtime or btime) and not infinite:
                remaining = wtime if board.turn == chess.WHITE else btime
                if remaining:
                    tlim = max(0.01, remaining / 1000.0 / 40.0)
                else:
                    tlim = DEFAULT_MOVE_TIME
            elif infinite:
                tlim = 3600.0
            else:
                tlim = DEFAULT_MOVE_TIME

            maxd = depth if depth else 32
            search_result.clear()
            search_thread = threading.Thread(target=start_search, args=(("normal", (maxd, tlim, nodes, multipv)),))
            search_thread.start()
            while search_thread.is_alive():
                try:
                    search_thread.join(timeout=0.1)
                except KeyboardInterrupt:
                    stop_event.set()
            multipv_list = search_result.get('multipv_list')
            best = search_result.get('best')
            if multipv_list:
                for idx, entry in enumerate(multipv_list, start=1):
                    sc = entry[0]      # <-- this is becoming a Move
                    pv = entry[1]
                    if abs(sc) > MATE_VALUE // 2:
                        mate = (MATE_VALUE - abs(sc)) // 100
                        score_str = f"mate {mate if sc > 0 else -mate}"
                    else:
                        score_str = f"cp {int(sc)}"
                    pv_str = " ".join(m.uci() for m in pv)
                    print(f"info multipv {idx} score {score_str} depth {len(pv)} nodes {node_count} pv {pv_str}")

            if best:
                print("bestmove", best.uci())
            else:
                lm = list(board.legal_moves)
                if lm:
                    print("bestmove", lm[0].uci())
                else:
                    print("bestmove 0000")
        elif line == "stop":
            stop_event.set()
            if search_thread and search_thread.is_alive():
                search_thread.join(timeout=1.0)
            best = search_result.get('best')
            if best:
                print("bestmove", best.uci())
            else:
                lm = list(board.legal_moves)
                if lm:
                    print("bestmove", lm[0].uci())
                else:
                    print("bestmove 0000")
            sys.stdout.flush()
        elif line == "quit":
            stop_event.set()
            if search_thread and search_thread.is_alive():
                search_thread.join(timeout=1.0)
            break
        sys.stdout.flush()

if __name__ == "__main__":
    random.seed(0xC0FFEE)
    uci_loop()
