# ════════════════════════════════════════════════════════════

# PROJECT: SMART CAMPUS NAVIGATION SYSTEM


# ════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import heapq
import random

# ════════════════════════════════════════════════════════════
# STEP 1 — CAMPUS GRID
#
#   16 rows × 24 cols.  Three building strips:
#
#   Row  0          → North perimeter road
#   Rows 1–3        → Top buildings  (Admin | Library | Block-B | Block-C)
#   Row  4          → East-West road
#   Rows 5–10       → (Lab Block)  ←  Central Quad  →  (Block A)
#   Row  11         → East-West road
#   Rows 12–14      → Bottom buildings (Hostel | Cafeteria | Sports | Workshop)
#   Row  15         → South perimeter road
#   Col  0 / 23    → West / East perimeter roads (always open)
# ════════════════════════════════════════════════════════════

ROWS, COLS = 16, 24
ROAD, WALL = 0, 1

city_map = [[ROAD] * COLS for _ in range(ROWS)]

# ── Named campus buildings ──────────────────────────────────
#   ▸ RENAME the strings to match your university blocks!
# ────────────────────────────────────────────────────────────
buildings = {
    # Top strip (rows 1-3)
    "Admin\nBlock":    [(r, c) for r in range(1, 4)  for c in range(1,  5)],
    "Library":         [(r, c) for r in range(1, 4)  for c in range(6,  12)],
    "Block B":         [(r, c) for r in range(1, 4)  for c in range(13, 18)],
    "Block C":         [(r, c) for r in range(1, 4)  for c in range(19, 23)],
    # Left / Right flanks (rows 5-10)
    "Lab\nBlock":      [(r, c) for r in range(5, 11) for c in range(1,  5)],
    "Block A":         [(r, c) for r in range(5, 11) for c in range(19, 23)],
    # Bottom strip (rows 12-14)
    "Hostel":          [(r, c) for r in range(12, 15) for c in range(1,  5)],
    "Cafeteria":       [(r, c) for r in range(12, 15) for c in range(6,  12)],
    "Sports\nComplex": [(r, c) for r in range(12, 15) for c in range(13, 18)],
    "Workshop":        [(r, c) for r in range(12, 15) for c in range(19, 23)],
}

# ── Distinct colour per building type ───────────────────────
BUILDING_COLORS = {
    "Admin\nBlock":    "#1a3a6b",   # navy
    "Library":         "#5c2d0a",   # warm brown
    "Block B":         "#1a4a2a",   # forest green
    "Block C":         "#5c1a3a",   # deep rose
    "Lab\nBlock":      "#7a3200",   # burnt orange
    "Block A":         "#1a3d5c",   # steel blue
    "Hostel":          "#4a3015",   # earth brown
    "Cafeteria":       "#6b5200",   # golden brown
    "Sports\nComplex": "#0a3d1a",   # dark green
    "Workshop":        "#5c1a1a",   # dark red
}

# Apply walls to city_map
for cells in buildings.values():
    for r, c in cells:
        city_map[r][c] = WALL

# Pre-compute building centroids for label placement
building_centroid = {
    name: (
        sum(r for r, c in cells) / len(cells),
        sum(c for r, c in cells) / len(cells),
    )
    for name, cells in buildings.items()
}

# ── Central Quad: open green area (fully traversable) ───────
grass_cells = frozenset(
    (r, c)
    for r in range(5, 11)
    for c in range(6, 18)
    if city_map[r][c] == ROAD
)

# ════════════════════════════════════════════════════════════
# STEP 2 — TIME COST MODEL
# ════════════════════════════════════════════════════════════

NORMAL_TIME  = 1    # minutes per clear road cell
TRAFFIC_TIME = 5    # minutes per congested cell

def calculate_time(path, traffic_cells):
    return sum(
        TRAFFIC_TIME if cell in traffic_cells else NORMAL_TIME
        for cell in path
    )

# ════════════════════════════════════════════════════════════
# STEP 3 — USER INPUT
# ════════════════════════════════════════════════════════════

def get_cell(prompt):
    while True:
        try:
            raw = input(prompt).strip().replace("(","").replace(")","")
            r, c = map(int, raw.split(","))
            if not (0 <= r < ROWS and 0 <= c < COLS):
                print(f"  Rows 0-{ROWS-1}, Cols 0-{COLS-1}")
                continue
            if city_map[r][c] == WALL:
                print("  That cell is inside a building. Pick a road cell.")
                continue
            return (r, c)
        except:
            print("  Format: row,col   e.g. 0,0")

print("\n  ── SMART CAMPUS NAVIGATION ──────────────────────────────")
print(f"  Grid: {ROWS} rows × {COLS} cols")
print("  Clear road cells to use as START / END:")
print("    Top road     : row  0  (any col 0-23)")
print("    Mid roads    : row  4  or row 11  (any col)")
print("    Bottom road  : row 15  (any col 0-23)")
print("    Perimeter    : col  0  or col 23  (any row)")
print("    Central Quad : rows 5-10, cols 6-17")
print("  Examples: (0,0)→(15,23)  |  (0,11)→(15,11)  |  (4,0)→(11,23)")
print("  ──────────────────────────────────────────────────────────\n")

START = get_cell("  Enter START cell (row,col): ")
END   = get_cell("  Enter END   cell (row,col): ")
while START == END:
    print("  START and END must be different.")
    END = get_cell("  Enter END cell (row,col): ")

# ════════════════════════════════════════════════════════════
# STEP 4 — A* PATHFINDING
# ════════════════════════════════════════════════════════════

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def find_path(start, end, blocked_cells):
    open_list = []
    heapq.heappush(open_list, (0, 0, start))
    came_from = {start: None}
    g_score   = {start: 0}
    explored  = []

    while open_list:
        f, g, current = heapq.heappop(open_list)
        explored.append(current)
        if current == end:
            break
        r, c = current
        for nr, nc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
            if not (0 <= nr < ROWS and 0 <= nc < COLS): continue
            if city_map[nr][nc] == WALL:               continue
            if (nr, nc) in blocked_cells:              continue
            new_g = g_score[current] + 1
            if (nr,nc) not in g_score or new_g < g_score[(nr,nc)]:
                g_score[(nr,nc)] = new_g
                heapq.heappush(open_list,
                    (new_g + manhattan((nr,nc),end), new_g, (nr,nc)))
                came_from[(nr,nc)] = current

    if end not in came_from:
        return [], explored
    path, step = [], end
    while step is not None:
        path.append(step); step = came_from.get(step)
    path.reverse()
    return path, explored

# ════════════════════════════════════════════════════════════
# STEP 5 — FIND PATHS + PLACE TRAFFIC
# ════════════════════════════════════════════════════════════

path1, explored1 = find_path(START, END, set())
if not path1:
    print("\n  No path exists between START and END."); exit()

mid = len(path1) // 2
on_path_candidates = [
    path1[i]
    for i in range(max(0, mid-2), min(len(path1), mid+3))
    if path1[i] not in (START, END)
]
on_path_traffic = set(on_path_candidates[:3])

off_path_roads = [
    (r, c) for r in range(ROWS) for c in range(COLS)
    if city_map[r][c] == ROAD
    and (r, c) not in (START, END)
    and (r, c) not in set(path1)
]
random.shuffle(off_path_roads)
off_path_traffic = set(off_path_roads[:7])
blocked_traffic  = on_path_traffic | off_path_traffic

path2, explored2 = find_path(START, END, blocked_traffic)
while not path2 and blocked_traffic:
    blocked_traffic.pop()
    path2, explored2 = find_path(START, END, blocked_traffic)

if not path2:
    print("\n  No alternate path found. Try different START/END."); exit()

# ════════════════════════════════════════════════════════════
# STEP 6 — TRAVEL TIMES
# ════════════════════════════════════════════════════════════

path1_time = calculate_time(path1, blocked_traffic)
path2_time = calculate_time(path2, blocked_traffic)
time_saved = path1_time - path2_time

print(f"\n  Traffic on first path : {len(on_path_traffic)} cells")
print(f"  Random traffic        : {len(off_path_traffic)} cells")
print(f"  First path time       : {path1_time} min")
print(f"  Smart path time       : {path2_time} min")
print(f"  Time saved            : {time_saved} min")

# ════════════════════════════════════════════════════════════
# STEP 7 — COLORMAP  (7 levels, campus map palette)
#
#   0  grass       — central quad
#   1  pavement    — campus roads
#   2  building    — placeholder (covered by patches)
#   3  explored    — A* visited cells
#   4  first path  — orange route
#   5  traffic     — red congestion
#   6  smart path  — bright green route
# ════════════════════════════════════════════════════════════

CAMPUS_CMAP = mcolors.ListedColormap([
    "#7db87a",   # 0  grass
    "#cec0a8",   # 1  stone pavement
    "#cec0a8",   # 2  building placeholder (same as pavement)
    "#64aadf",   # 3  A* explored
    "#e07840",   # 4  first path (burnt orange)
    "#d93025",   # 5  traffic (red)
    "#00d455",   # 6  smart path (bright green)
])

def make_frame(explored=(), path_1=(), path_2=(), show_traffic=False):
    img = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            if city_map[r][c] == WALL:
                row.append(2)
            elif (r, c) in grass_cells:
                row.append(0)
            else:
                row.append(1)
        img.append(row)
    for r, c in explored:
        img[r][c] = 3
    for r, c in path_1:
        img[r][c] = 4
    if show_traffic:
        for r, c in blocked_traffic:
            img[r][c] = 5
    for r, c in path_2:
        img[r][c] = 6
    return img

# ════════════════════════════════════════════════════════════
# STEP 8 — FIGURE LAYOUT
# ════════════════════════════════════════════════════════════

FIG_BG  = "#f2ede4"   # warm parchment
AXIS_BG = "#e4ddd0"   # map canvas

fig = plt.figure(figsize=(22, 11), facecolor=FIG_BG)
gs  = fig.add_gridspec(
    1, 3, width_ratios=[4, 1, 1], wspace=0.12,
    left=0.02, right=0.98, top=0.90, bottom=0.10
)
ax_map   = fig.add_subplot(gs[0])
ax_steps = fig.add_subplot(gs[1])
ax_time  = fig.add_subplot(gs[2])

fig.suptitle(
    "SMART CAMPUS NAVIGATION  ·  A* Pathfinding",
    color="#1a1a2e", fontsize=16, fontweight="bold", y=0.97
)

ax_map.set_facecolor(AXIS_BG)

im = ax_map.imshow(
    make_frame(),
    cmap=CAMPUS_CMAP, vmin=0, vmax=6,
    interpolation="nearest", origin="upper"
)

# Subtle cell grid
ax_map.set_xticks([])
ax_map.set_yticks([])
ax_map.set_xticks([x - 0.5 for x in range(COLS + 1)], minor=True)
ax_map.set_yticks([y - 0.5 for y in range(ROWS + 1)], minor=True)
ax_map.grid(which="minor", color="#b8a898", linewidth=0.3, alpha=0.55)
ax_map.tick_params(which="minor", length=0)

# Campus perimeter fence
for spine in ax_map.spines.values():
    spine.set_edgecolor("#5a4a3a")
    spine.set_linewidth(2.5)

# ════════════════════════════════════════════════════════════
# STEP 9 — BUILDING PATCHES  (drawn once, persist throughout)
# ════════════════════════════════════════════════════════════

for name, cells in buildings.items():
    rr  = [r for r, c in cells]
    cc_ = [c for r, c in cells]
    r0, r1 = min(rr), max(rr)
    c0, c1 = min(cc_), max(cc_)
    color = BUILDING_COLORS[name]
    cr, cc = building_centroid[name]

    # Drop shadow
    ax_map.add_patch(mpatches.Rectangle(
        (c0 - 0.33, r0 + 0.18),
        c1 - c0 + 0.95, r1 - r0 + 0.95,
        facecolor="#00000025", edgecolor="none", zorder=3
    ))

    # Building body
    ax_map.add_patch(mpatches.Rectangle(
        (c0 - 0.47, r0 - 0.47),
        c1 - c0 + 0.94, r1 - r0 + 0.94,
        facecolor=color, edgecolor="#ffffffb0",
        linewidth=1.6, zorder=4
    ))

    # Rooftop highlight (top stripe, ~15% of building height)
    bh = (r1 - r0 + 0.94) * 0.18
    ax_map.add_patch(mpatches.Rectangle(
        (c0 - 0.47, r0 - 0.47),
        c1 - c0 + 0.94, bh,
        facecolor="#ffffff22", edgecolor="none", zorder=5
    ))

    # Building name
    ax_map.text(
        cc, cr, name,
        ha="center", va="center",
        color="white", fontsize=(7.2 if "\n" in name else 8.5),
        fontweight="bold", multialignment="center",
        zorder=6
    )

# ════════════════════════════════════════════════════════════
# STEP 10 — CAMPUS DETAILS
# ════════════════════════════════════════════════════════════

# Central Quad dashed border
ax_map.add_patch(mpatches.Rectangle(
    (5.52, 4.52), 11.96, 5.96,
    facecolor="none", edgecolor="#3a7a3a",
    linewidth=1.2, linestyle="--", zorder=3, alpha=0.6
))

# Quad label
ax_map.text(
    12, 7.5, "CENTRAL QUAD",
    ha="center", va="center",
    color="#2d6a2d", fontsize=9,
    fontstyle="italic", fontweight="bold",
    alpha=0.72, zorder=5
)

# Diagonal footpaths across quad (visual only)
ax_map.plot([6, 17], [5, 10], color="#b0a080",
            linewidth=0.9, alpha=0.4, zorder=3)
ax_map.plot([6, 17], [10, 5], color="#b0a080",
            linewidth=0.9, alpha=0.4, zorder=3)

# Grass texture dots
_rng = random.Random(42)
for _ in range(45):
    ax_map.plot(
        _rng.uniform(6.2, 17.4),
        _rng.uniform(5.2, 10.4),
        ".", color="#4a8a4a", markersize=2.5, alpha=0.35, zorder=3
    )

# Compass rose
ax_map.text(
    COLS - 0.55, 0.5, "N\n↑",
    ha="center", va="center",
    color="#1a1a2e", fontsize=8, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="white", edgecolor="#888", linewidth=1, alpha=0.9),
    zorder=7
)

# Main Gate annotation
ax_map.annotate(
    "MAIN\nGATE",
    xy=(11.5, ROWS - 0.55),
    xytext=(11.5, ROWS - 2.2),
    ha="center", va="center",
    color="#1a1a2e", fontsize=7, fontweight="bold",
    multialignment="center",
    arrowprops=dict(arrowstyle="->", color="#1a1a2e", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.2",
              facecolor=FIG_BG, edgecolor="#888", linewidth=0.8),
    zorder=7
)

# START / END markers
for (r, c), label in [(START, "S"), (END, "E")]:
    ax_map.plot(c, r, "o", markersize=18,
                color="#222222", markeredgecolor="white",
                markeredgewidth=1.8, zorder=8)
    ax_map.text(c, r, label,
                ha="center", va="center",
                color="white", fontsize=10, fontweight="bold", zorder=9)

# Map legend (below the map)
ax_map.legend(
    handles=[
        mpatches.Patch(facecolor="#7db87a", edgecolor="#888", label="Grass / Open Area"),
        mpatches.Patch(facecolor="#cec0a8", edgecolor="#888", label="Paved Path"),
        mpatches.Patch(facecolor="#64aadf", edgecolor="#888", label="A* Explored"),
        mpatches.Patch(facecolor="#e07840", edgecolor="#888", label="First Route"),
        mpatches.Patch(facecolor="#d93025", edgecolor="#888", label="Traffic"),
        mpatches.Patch(facecolor="#00d455", edgecolor="#888", label="Smart Route"),
    ],
    loc="lower center", ncol=6, fontsize=7.5,
    framealpha=0.93, facecolor=FIG_BG, edgecolor="#aaa",
    bbox_to_anchor=(0.5, -0.09)
)

title_text = ax_map.set_title("", color="#1a1a2e", fontsize=11, pad=6)

# ════════════════════════════════════════════════════════════
# STEP 11 — BAR CHARTS (light theme matching the map)
# ════════════════════════════════════════════════════════════

def style_bar_ax(ax, title_str):
    ax.set_facecolor("#fffbf5")
    ax.set_title(title_str, color="#1a1a2e", fontsize=9.5,
                 fontweight="bold", pad=6)
    ax.tick_params(colors="#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

def draw_steps_bar(steps1, steps2):
    ax_steps.clear()
    style_bar_ax(ax_steps, "Steps\n(fewer = shorter)")
    bars = ax_steps.bar(
        ["First\nPath", "Smart\nPath"], [steps1, steps2],
        color=["#e07840", "#00d455"], edgecolor="#aaa", linewidth=0.8
    )
    for bar, val in zip(bars, [steps1, steps2]):
        if val > 0:
            ax_steps.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                          str(val), color="#333", fontsize=11, ha="center")
    ax_steps.set_ylim(0, max(len(path1), len(path2)) + 5)
    ax_steps.set_ylabel("Steps", color="#333333", fontsize=9)

def draw_time_bar(time1, time2):
    ax_time.clear()
    style_bar_ax(ax_time, "Travel Time\n(fewer = faster)")
    bars = ax_time.bar(
        ["First\nPath", "Smart\nPath"], [time1, time2],
        color=["#e07840", "#00d455"], edgecolor="#aaa", linewidth=0.8
    )
    for bar, val in zip(bars, [time1, time2]):
        if val > 0:
            ax_time.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                         f"{val}m", color="#333", fontsize=10, ha="center")
    if time1 > 0 and time2 > 0:
        ax_time.text(1, time2/2, "✓ FASTER",
                     ha="center", va="center",
                     color="#0a6a25", fontsize=8, fontweight="bold",
                     rotation=90, alpha=0.85)
    ax_time.set_ylim(0, max(path1_time, path2_time) + 5)
    ax_time.set_ylabel("Minutes", color="#333333", fontsize=9)

# ════════════════════════════════════════════════════════════
# STEP 12 — ANIMATION (5 phases)
# ════════════════════════════════════════════════════════════

p1 = len(explored1)
p2 = p1 + len(path1)
p3 = p2 + 20
p4 = p3 + len(explored2)
p5 = p4 + len(path2)

def update(frame):

    if frame < p1:                               # Phase 1: A* explores
        im.set_data(make_frame(explored=explored1[:frame]))
        title_text.set_text("Phase 1 — A* searching for first route…")
        draw_steps_bar(0, 0); draw_time_bar(0, 0)

    elif frame < p2:                             # Phase 2: first path
        n = frame - p1
        im.set_data(make_frame(explored=explored1, path_1=path1[:n]))
        title_text.set_text("Phase 2 — First route found  (orange)")
        draw_steps_bar(n, 0)
        draw_time_bar(calculate_time(path1[:n], set()), 0)

    elif frame < p3:                             # Phase 3: traffic appears
        im.set_data(make_frame(path_1=path1, show_traffic=True))
        title_text.set_text("⚠  Phase 3 — Traffic on campus!  Orange route slows down")
        draw_steps_bar(len(path1), 0)
        draw_time_bar(path1_time, 0)

    elif frame < p4:                             # Phase 4: re-explore
        n = frame - p3
        im.set_data(make_frame(explored=explored2[:n], path_1=path1, show_traffic=True))
        title_text.set_text("Phase 4 — Re-routing around traffic…")
        draw_steps_bar(len(path1), 0)
        draw_time_bar(path1_time, 0)

    else:                                        # Phase 5: smart path
        n = frame - p4
        im.set_data(make_frame(path_1=path1, path_2=path2[:n], show_traffic=True))
        title_text.set_text(
            f"✅  Smart route saves {time_saved} min!   "
            f"Orange = {path1_time} min   Green = {path2_time} min"
        )
        draw_steps_bar(len(path1), min(n, len(path2)))
        draw_time_bar(path1_time, calculate_time(path2[:n], blocked_traffic))
        if n >= len(path2):
            ani.event_source.stop()


ani = animation.FuncAnimation(
    fig, update, frames=p5, interval=120, repeat=False
)

plt.show()

# ════════════════════════════════════════════════════════════
# STEP 13 — FINAL RESULTS
# ════════════════════════════════════════════════════════════

print("\n  ════════════════════════════════════════════")
print("  SMART CAMPUS NAVIGATION — FINAL RESULTS")
print("  ════════════════════════════════════════════")
print(f"  First route steps  : {len(path1)}")
print(f"  Smart route steps  : {len(path2)}")
print(f"  Extra steps taken  : {len(path2) - len(path1)}")
print("  ──────────────────────────────────────────")
print(f"  First route time   : {path1_time} min  (hits traffic)")
print(f"  Smart route time   : {path2_time} min  (avoids traffic)")
print(f"  Time saved         : {time_saved} min  ← this is why it's smart!")
print("  ════════════════════════════════════════════")
