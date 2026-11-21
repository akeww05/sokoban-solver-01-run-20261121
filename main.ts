// Push-optimal Sokoban solver for make.fun sokoban-solver-01-run-20261121.
//
// Design (from the competition brief, written from scratch):
//  - A level is a 1-wide hallway with 1-4 rooms off it. A box pushed out of a
//    room can never come back, so every box stays in its room and the level
//    decomposes: solve each room on its own from its door and add the pushes.
//    The level optimum is the sum of the room optima.
//  - Each room is solved by A* over pushes. A state is the sorted box cells
//    plus the player's reachable region (the player walks freely between
//    pushes, so only the region matters). The heuristic is the minimum-cost
//    matching of boxes to goals over precomputed floor shortest paths, which
//    never overestimates: one push moves one box one step, so it can shrink
//    the matched distance by at most one.
//  - Deadlock pruning: static dead squares (found by a reverse pull search
//    from the goals, ignoring other boxes), corner deadlocks, 2x2 blocks of
//    boxes/walls, and frozen boxes against a wall. All are sound: they only
//    cut states that can never reach all boxes on goals.
//  - Everything is deterministic: fixed neighbour orders, a work budget
//    counted in expansions (never the wall clock), and a heap tie-break on
//    insertion order. Two runs write the same bytes.
//  - The player walks the hallway between rooms with plain BFS, boxes as
//    obstacles, and exits each room through its door before moving on (the
//    last room solved needs no exit).
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed neighbour order: up, down, left, right. Every BFS/expansion in this
// file iterates directions in this order, so output is deterministic.
const DIRS = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

// Hard work cap per room search, counted in expansions, never in seconds.
const MAX_EXPANSIONS = 400000;

type Cell = number;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goal: Uint8Array;
  boxes: number[];
  player: number;
}

interface Room {
  cells: number[];
  cellSet: Set<number>;
  boxes: number[];
  goals: number[];
  door: number;
  entrance: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const i = y * w + x;
      const ch = row[x];
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Split the level into rooms: connected floor components off the hallway row
// (the row the player starts on). Each room opens onto the hallway through
// exactly one door cell; the entrance is the hallway cell beside the door.
function findRooms(level: Level): Room[] {
  const { w, h, wall } = level;
  const prow = Math.floor(level.player / w);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (wall[i] || Math.floor(i / w) === prow || seen[i]) continue;
    const cells: number[] = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length > 0) {
      const c = stack.pop() as number;
      cells.push(c);
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || Math.floor(n / w) === prow || seen[n]) continue;
        seen[n] = 1;
        stack.push(n);
      }
    }
    cells.sort((a, b) => a - b);
    const cellSet = new Set(cells);
    const boxes = level.boxes.filter((b) => cellSet.has(b));
    const goals = cells.filter((c) => level.goal[c]);
    // The door is the room cell vertically adjacent to a hallway floor cell.
    let door = -1;
    let entrance = -1;
    for (const c of cells) {
      const cy = Math.floor(c / w);
      const n = cy < prow ? c + w : c - w;
      if (Math.floor(n / w) === prow && !wall[n]) {
        door = c;
        entrance = n;
        break;
      }
    }
    if (door < 0) continue; // No opening onto the hallway: not a room.
    rooms.push({ cells, cellSet, boxes, goals, door, entrance });
  }
  return rooms;
}

// Breadth-first shortest walk from `from` to `to` over floor cells, treating
// every cell in `blocked` as impassable. Returns the move string, or null
// when `to` cannot be reached. Neighbours are tried in DIRS order.
function walkPath(
  level: Level,
  from: number,
  to: number,
  blocked: Set<number>,
): string | null {
  if (from === to) return '';
  const { w, h, wall } = level;
  const prev = new Int32Array(w * h).fill(-2);
  const prevCh = new Array<string>(w * h).fill('');
  prev[from] = -1;
  const queue = [from];
  let head = 0;
  while (head < queue.length) {
    const c = queue[head++];
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || blocked.has(n) || prev[n] !== -2) continue;
      prev[n] = c;
      prevCh[n] = d.ch;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          s = prevCh[cur] + s;
          cur = prev[cur];
        }
        return s;
      }
      queue.push(n);
    }
  }
  return null;
}

// Flood fill of the player's reachable region from `start`, with `blocked`
// cells impassable. Returns the set and its smallest cell (the region's
// canonical representative: any player cell in the same region is the same
// search state).
function floodRegion(
  level: Level,
  start: number,
  blocked: Set<number>,
  allowed: Set<number>,
): { cells: Set<number>; rep: number } {
  const { w, h, wall } = level;
  const cells = new Set<number>();
  let rep = start;
  const stack = [start];
  cells.add(start);
  while (stack.length > 0) {
    const c = stack.pop() as number;
    if (c < rep) rep = c;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const d of DIRS) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (wall[n] || blocked.has(n) || !allowed.has(n) || cells.has(n)) continue;
      cells.add(n);
      stack.push(n);
    }
  }
  return { cells, rep };
}

// Static dead squares for one room: floor cells a box can never be pushed
// from onto a goal, found by a reverse pull search from the goals that
// ignores other boxes (so it over-approximates reachability and only marks
// cells that are truly dead). The player may stand on the room floor and on
// the entrance cell.
function deadSquares(level: Level, room: Room): Set<number> {
  const { w, h, wall } = level;
  const allowed = new Set(room.cells);
  allowed.add(room.entrance);
  const live = new Set<number>();
  // State (box, player); the player stands next to the box. Start with the
  // box on every goal and the player on every neighbouring floor cell.
  const seen = new Set<number>();
  const queue: Array<[number, number]> = [];
  for (const g of room.goals) {
    const gx = g % w;
    const gy = Math.floor(g / w);
    for (const d of DIRS) {
      const nx = gx + d.dx;
      const ny = gy + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const p = ny * w + nx;
      if (wall[p] || !allowed.has(p)) continue;
      const key = g * (w * h) + p;
      if (seen.has(key)) continue;
      seen.add(key);
      queue.push([g, p]);
      live.add(g);
    }
  }
  let head = 0;
  while (head < queue.length) {
    const [b, p] = queue[head++];
    const px = p % w;
    const py = Math.floor(p / w);
    // Player walks to any neighbouring floor cell (not through the box).
    for (const d of DIRS) {
      const nx = px + d.dx;
      const ny = py + d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const np = ny * w + nx;
      if (wall[np] || !allowed.has(np) || np === b) continue;
      const key = b * (w * h) + np;
      if (seen.has(key)) continue;
      seen.add(key);
      queue.push([b, np]);
    }
    // Player pulls the box: the box slides into the player's cell and the
    // player steps one cell further away from the box.
    const bx = b % w;
    const by = Math.floor(b / w);
    for (const d of DIRS) {
      if (px + d.dx !== bx || py + d.dy !== by) continue;
      const nx = px - d.dx;
      const ny = py - d.dy;
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const np = ny * w + nx;
      if (wall[np] || !allowed.has(np)) continue;
      const nb = p;
      const key = nb * (w * h) + np;
      if (seen.has(key)) continue;
      seen.add(key);
      queue.push([nb, np]);
      live.add(nb);
    }
  }
  const dead = new Set<number>();
  for (const c of room.cells) {
    if (!live.has(c)) dead.add(c);
  }
  return dead;
}

// True when a box configuration can never be solved: a box sits on a dead
// square, in a corner that is not a goal, in a full 2x2 block of walls and
// boxes, or is stuck: every push direction is blocked by a wall or by a box
// that is itself stuck. The stuck set is grown to a fixpoint, so a box
// blocked only by boxes that can still move is never called stuck.
function isDeadlocked(
  level: Level,
  boxes: number[],
  boxSet: Set<number>,
  dead: Set<number>,
): boolean {
  const { w, wall, goal } = level;
  for (const b of boxes) {
    if (goal[b]) continue;
    if (dead.has(b)) return true;
    const bx = b % w;
    const by = Math.floor(b / w);
    const north = by > 0 ? b - w : -1;
    const south = by + 1 < level.h ? b + w : -1;
    const west = bx > 0 ? b - 1 : -1;
    const east = bx + 1 < w ? b + 1 : -1;
    const wN = north < 0 || wall[north];
    const wS = south < 0 || wall[south];
    const wW = west < 0 || wall[west];
    const wE = east < 0 || wall[east];
    // Corner: walls on two perpendicular sides. Such a box can never move:
    // two push directions run into the walls and the other two need the
    // player to stand on a wall cell.
    if ((wN || wS) && (wW || wE)) return true;
    // 2x2 block of walls/boxes holding this box. Every box in a full 2x2
    // block is immobile: two push directions run into the block and the
    // other two need the player to stand on a block cell.
    const blocks: Array<[number, number, number, number]> = [
      [b, east, south, south + 1],
      [b, west, south, south - 1],
      [b, east, north, north + 1],
      [b, west, north, north - 1],
    ];
    for (const blk of blocks) {
      if (blk.some((c) => c < 0 || c >= w * level.h)) continue;
      if (blk.every((c) => wall[c] || boxSet.has(c))) {
        if (blk.some((c) => boxSet.has(c) && !goal[c])) return true;
      }
    }
  }
  // Stuck fixpoint: a box (never one already on a goal) joins when every
  // push direction is blocked, either because the destination is a wall or
  // a stuck box, or because the cell the player must stand on is a wall or
  // a stuck box. Boxes that join never move again, so the directions stay
  // blocked; a box blocked only by a box that can still move never joins.
  const stuck = new Set<number>();
  const blockedCell = (c: number): boolean =>
    c < 0 || c >= w * level.h || wall[c] || (boxSet.has(c) && stuck.has(c));
  let changed = true;
  while (changed) {
    changed = false;
    for (const b of boxes) {
      if (goal[b] || stuck.has(b)) continue;
      let allBlocked = true;
      for (const d of DIRS) {
        const step = d.dy * w + d.dx;
        if (!blockedCell(b + step) && !blockedCell(b - step)) {
          allBlocked = false;
          break;
        }
      }
      if (allBlocked) {
        stuck.add(b);
        changed = true;
      }
    }
  }
  return stuck.size > 0;
}

// Minimum-cost matching of boxes to goals over precomputed floor distances.
// Returns Infinity when some box cannot reach any goal.
function matchingHeuristic(
  boxes: number[],
  goals: number[],
  dist: Map<number, Map<number, number>>,
): number {
  const n = boxes.length;
  // Brute-force permutations: n is 3 or 4, so at most 24.
  const perm = goals.map((_, i) => i);
  let best = Infinity;
  const rec = (k: number, sum: number): void => {
    if (sum >= best) return;
    if (k === n) {
      best = sum;
      return;
    }
    for (let i = k; i < n; i++) {
      const t = perm[k];
      perm[k] = perm[i];
      perm[i] = t;
      const d = dist.get(boxes[k])?.get(goals[perm[k]]) ?? -1;
      if (d >= 0) rec(k + 1, sum + d);
      const t2 = perm[k];
      perm[k] = perm[i];
      perm[i] = t2;
    }
  };
  rec(0, 0);
  return best;
}

// A binary heap of node indices ordered by (f, -g, insertion order).
class Heap {
  private items: number[] = [];
  private f: number[];
  private g: number[];
  private seq: number[];
  constructor(f: number[], g: number[], seq: number[]) {
    this.f = f;
    this.g = g;
    this.seq = seq;
  }
  get size(): number {
    return this.items.length;
  }
  private less(a: number, b: number): boolean {
    if (this.f[a] !== this.f[b]) return this.f[a] < this.f[b];
    if (this.g[a] !== this.g[b]) return this.g[a] > this.g[b];
    return this.seq[a] < this.seq[b];
  }
  push(x: number): void {
    this.items.push(x);
    let i = this.items.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(this.items[i], this.items[p])) break;
      const t = this.items[i];
      this.items[i] = this.items[p];
      this.items[p] = t;
      i = p;
    }
  }
  pop(): number {
    const top = this.items[0];
    const last = this.items.pop() as number;
    if (this.items.length > 0) {
      this.items[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < this.items.length && this.less(this.items[l], this.items[m])) m = l;
        if (r < this.items.length && this.less(this.items[r], this.items[m])) m = r;
        if (m === i) break;
        const t = this.items[i];
        this.items[i] = this.items[m];
        this.items[m] = t;
        i = m;
      }
    }
    return top;
  }
}

interface RoomSolution {
  moves: string;
  finalBoxes: number[];
  endPlayer: number;
  pushes: number;
}

// Solve one room from its entrance cell with push-optimal A*. When
// `exitRequired` is set, the goal test also requires the entrance to be
// reachable from the player's final cell, so the player can leave the room
// and walk on to the next one. Returns null when the work budget runs out
// or no solution exists.
function solveRoom(
  level: Level,
  room: Room,
  exitRequired: boolean,
): RoomSolution | null {
  const { w, h, wall } = level;
  const allowed = new Set(room.cells);
  allowed.add(room.entrance);
  const dead = deadSquares(level, room);

  // All-pairs shortest floor distances over the allowed cells (boxes
  // ignored), for the matching heuristic.
  const dist = new Map<number, Map<number, number>>();
  for (const s of allowed) {
    const row = new Map<number, number>();
    const dq = [s];
    row.set(s, 0);
    let head = 0;
    while (head < dq.length) {
      const c = dq[head++];
      const cx = c % w;
      const cy = Math.floor(c / w);
      for (const d of DIRS) {
        const nx = cx + d.dx;
        const ny = cy + d.dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const n = ny * w + nx;
        if (wall[n] || !allowed.has(n) || row.has(n)) continue;
        row.set(n, (row.get(c) as number) + 1);
        dq.push(n);
      }
    }
    dist.set(s, row);
  }

  const startBoxes = [...room.boxes].sort((a, b) => a - b);
  const startSet = new Set(startBoxes);
  if (isDeadlocked(level, startBoxes, startSet, dead)) return null;

  const keyOf = (boxes: number[], rep: number): string =>
    boxes.join(',') + '|' + rep;

  // Node storage, indexed by node id.
  const par: number[] = [];
  const boxFrom: number[] = [];
  const boxTo: number[] = [];
  const nodePlayer: number[] = [];
  const nodeBoxes: number[][] = [];
  const gArr: number[] = [];
  const fArr: number[] = [];
  const seqArr: number[] = [];
  let seq = 0;

  const addNode = (
    parent: number,
    from: number,
    to: number,
    player: number,
    boxes: number[],
    g: number,
    f: number,
  ): number => {
    const id = par.length;
    par.push(parent);
    boxFrom.push(from);
    boxTo.push(to);
    nodePlayer.push(player);
    nodeBoxes.push(boxes);
    gArr.push(g);
    fArr.push(f);
    seqArr.push(seq++);
    return id;
  };

  const startRegion = floodRegion(level, room.entrance, startSet, allowed);
  const h0 = matchingHeuristic(startBoxes, room.goals, dist);
  if (!Number.isFinite(h0)) return null;
  const root = addNode(-1, -1, -1, room.entrance, startBoxes, 0, h0);
  const best = new Map<string, number>();
  best.set(keyOf(startBoxes, startRegion.rep), 0);
  const heap = new Heap(fArr, gArr, seqArr);
  heap.push(root);

  let expansions = 0;
  let goalNode = -1;
  while (heap.size > 0) {
    const id = heap.pop();
    const boxes = nodeBoxes[id];
    const g = gArr[id];
    const player = nodePlayer[id];
    const boxSet = new Set(boxes);
    const region = floodRegion(level, player, boxSet, allowed);
    const key = keyOf(boxes, region.rep);
    if ((best.get(key) ?? Infinity) < g) continue;
    // Goal: every box on a goal (heuristic zero), plus the exit when asked.
    const hHere = matchingHeuristic(boxes, room.goals, dist);
    if (hHere === 0) {
      if (!exitRequired || region.cells.has(room.entrance)) {
        goalNode = id;
        break;
      }
    }
    if (++expansions > MAX_EXPANSIONS) return null;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      const bx = b % w;
      const by = Math.floor(b / w);
      for (const d of DIRS) {
        const pushCell = (by - d.dy) * w + (bx - d.dx);
        const dest = (by + d.dy) * w + (bx + d.dx);
        if (!region.cells.has(pushCell)) continue;
        if (!room.cellSet.has(dest)) continue; // Boxes never leave the room.
        if (boxSet.has(dest)) continue;
        if (dead.has(dest)) continue;
        const next = [...boxes];
        next[bi] = dest;
        next.sort((a, b2) => a - b2);
        const nextSet = new Set(next);
        if (isDeadlocked(level, next, nextSet, dead)) continue;
        const hNext = matchingHeuristic(next, room.goals, dist);
        if (!Number.isFinite(hNext)) continue;
        const g2 = g + 1;
        const nextRegion = floodRegion(level, b, nextSet, allowed);
        const key2 = keyOf(next, nextRegion.rep);
        if ((best.get(key2) ?? Infinity) <= g2) continue;
        best.set(key2, g2);
        const nid = addNode(id, b, dest, b, next, g2, g2 + hNext);
        heap.push(nid);
      }
    }
  }
  if (goalNode < 0) return null;

  // Rebuild the push list, then walk it forward from the entrance, turning
  // each push into the player's walk to the pushing cell plus the push step.
  const pushes: Array<[number, number]> = [];
  let cur = goalNode;
  while (par[cur] >= 0) {
    pushes.push([boxFrom[cur], boxTo[cur]]);
    cur = par[cur];
  }
  pushes.reverse();
  let moves = '';
  let player = room.entrance;
  let boxes = [...startBoxes];
  for (const [from, to] of pushes) {
    const d = { dx: Math.sign(to % w - from % w), dy: Math.sign(Math.floor(to / w) - Math.floor(from / w)) };
    const pushCell = from - (d.dy * w + d.dx);
    const path = walkPath(level, player, pushCell, new Set(boxes));
    if (path === null) return null; // Should not happen: reachability was checked.
    moves += path;
    const step = DIRS.find((dd) => dd.dx === d.dx && dd.dy === d.dy) as { ch: string };
    moves += step.ch;
    boxes = boxes.map((b) => (b === from ? to : b));
    boxes.sort((a, b2) => a - b2);
    player = from;
  }
  return { moves, finalBoxes: boxes, endPlayer: player, pushes: pushes.length };
}

// Solve one whole level. Rooms are taken nearest-first from the player's
// current cell; a room that cannot be solved with an exit is deferred and
// retried once the others are done (as the last room it needs no exit).
function solveLevel(level: Level): string {
  const rooms = findRooms(level);
  if (rooms.length === 0) return '';
  let moves = '';
  let player = level.player;
  const boxes = new Set(level.boxes);
  const queue = [...rooms];
  const deferredExit = new Set<Room>();
  while (queue.length > 0) {
    // Nearest remaining room by hallway walk from the current player cell,
    // preferring rooms that have not already failed with an exit. A room
    // that failed with an exit is only retried once every room left has
    // failed that way, and then without one.
    let candidates = queue.filter((r) => !deferredExit.has(r));
    if (candidates.length === 0) candidates = queue;
    let bestIdx = -1;
    let bestDist = Infinity;
    // Two passes: rooms that have not failed with an exit first, then any
    // room, so a deferred room is only picked when nothing else is reachable.
    for (const pool of [candidates, queue]) {
      for (let i = 0; i < queue.length; i++) {
        if (!pool.includes(queue[i])) continue;
        const path = walkPath(level, player, queue[i].entrance, boxes);
        if (path !== null && path.length < bestDist) {
          bestDist = path.length;
          bestIdx = i;
        }
      }
      if (bestIdx >= 0) break;
    }
    if (bestIdx < 0) return ''; // No room reachable: give up, stay valid.
    const room = queue[bestIdx];
    const isLast = queue.length === 1;
    const exitRequired = !isLast && !deferredExit.has(room);
    const sol = solveRoom(level, room, exitRequired);
    if (sol === null) {
      if (exitRequired) {
        // Defer: solve another room first, retry this one later.
        deferredExit.add(room);
        queue.splice(bestIdx, 1);
        queue.push(room);
        continue;
      }
      return ''; // Unsolvable even without an exit: stay valid.
    }
    const toEntrance = walkPath(level, player, room.entrance, boxes);
    if (toEntrance === null) return '';
    moves += toEntrance;
    moves += sol.moves;
    for (const b of room.boxes) boxes.delete(b);
    for (const b of sol.finalBoxes) boxes.add(b);
    player = sol.endPlayer;
    queue.splice(bestIdx, 1);
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
