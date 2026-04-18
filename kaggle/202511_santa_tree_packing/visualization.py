"""
Santa's Tree Packing — Manim/ManimGL visualization
====================================================
Run with:
  manimgl visualization.py Scene1_Title
  manimgl visualization.py Scene2_TreeShape
  manimgl visualization.py Scene3_BLF
  manimgl visualization.py Scene4_GA
  manimgl visualization.py Scene5_Incremental
  manimgl visualization.py FullPipeline          (all scenes, ~90 s)
"""
from manim import *
import numpy as np

# ─── Palette ──────────────────────────────────────────────────────────────────
BG      = "#0D1117"
C_TREE  = "#238636"   # placed tree green
C_LITE  = "#3FB950"   # highlight green
C_DIM   = "#145724"   # settled/locked tree
C_OUTLINE = "#0A2E12" # dark green outline for all trees
C_NEW   = "#58A6FF"   # incoming / new tree
C_GOLD  = "#E3B341"   # titles, labels
C_RED   = "#F85149"   # collision, error
C_BOX   = "#388BFD"   # packing box outline
C_GRY   = "#6E7681"   # subtle grey
C_WHT   = "#E6EDF3"   # body text
C_PUR   = "#BC8CFF"   # GA / purple accent

# ─── Christmas-tree geometry ──────────────────────────────────────────────────
# 15-vertex polygon (unit scale, pointing up, trunk at bottom)
_TV = np.array([
    [ 0.0000,  0.800],                       # tip
    [ 0.1250,  0.500], [ 0.0625,  0.500],    # right top tier
    [ 0.2000,  0.250], [ 0.1000,  0.250],    # right mid tier
    [ 0.3500,  0.000],                        # right base
    [ 0.0750,  0.000], [ 0.0750, -0.200],    # right trunk
    [-0.0750, -0.200], [-0.0750,  0.000],    # left trunk
    [-0.3500,  0.000],                        # left base
    [-0.1000,  0.250], [-0.2000,  0.250],    # left mid tier
    [-0.0625,  0.500], [-0.1250,  0.500],    # left top tier
])

def _R(pts: np.ndarray, deg: float) -> np.ndarray:
    """Rotate 2-D point array by deg degrees (CCW)."""
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return (np.array([[c, -s], [s, c]]) @ pts.T).T

def mk_tree(sc=1., ang=0, cx=0, cy=0,
            col=C_TREE, fill=0.85, sw=2.5) -> Polygon:
    """Return a Manim Polygon representing a Christmas tree."""
    v = _R(_TV * sc, ang) + np.array([cx, cy])
    return Polygon(*[[x, y, 0.] for x, y in v],
                   fill_color=col, fill_opacity=fill,
                   stroke_color=C_OUTLINE, stroke_width=sw)

def tbounds(sc: float, ang: float):
    """Axis-aligned bounding box (min_x, max_x, min_y, max_y) at origin."""
    v = _R(_TV * sc, ang)
    return v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 1 — Title
# ═══════════════════════════════════════════════════════════════════════════════
class Scene1_Title(Scene):
    def construct(self):
        self.camera.background_color = BG

        # Faint background trees
        bg = VGroup(*[
            mk_tree(sc=0.75, ang=a, cx=x, cy=y, col=C_DIM, fill=0.20)
            for a, x, y in [
                (  0, -5.5,  2.2), (180, -3.8, -1.8), ( 90, -1.5,  2.8),
                (270,  1.8,  2.6), (  0,  4.3,  1.2), (135,  5.6, -1.2),
                (315, -5.4, -1.5), (225,  3.5, -2.6), ( 45, -0.8, -2.6),
            ]
        ])
        self.add(bg)

        title = Text("Santa's Tree Packing", font_size=56, color=C_GOLD)
        sub = Text(
            "Hybrid Genetic Algorithm  +  Bottom-Left Fill",
            font_size=26, color=C_WHT,
        )
        sub.next_to(title, DOWN, buff=0.45)

        tags = VGroup(
            Text("① BLF Placement",    font_size=22, color=C_LITE),
            Text("② GA Angle Search",  font_size=22, color=C_NEW),
            Text("③ Incremental Build",font_size=22, color=C_PUR),
        ).arrange(RIGHT, buff=1.3)
        tags.next_to(sub, DOWN, buff=0.65)

        VGroup(title, sub, tags).center()

        self.play(FadeIn(title, shift=UP * 0.3), run_time=1.0)
        self.play(FadeIn(sub), run_time=0.7)
        self.play(
            LaggedStart(*[FadeIn(t, shift=UP * 0.2) for t in tags],
                        lag_ratio=0.35),
            run_time=1.0,
        )
        self.wait(2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 2 — Tree Shape & Rotation
# ═══════════════════════════════════════════════════════════════════════════════
class Scene2_TreeShape(Scene):
    def construct(self):
        self.camera.background_color = BG

        hdr = Text("The Christmas Tree Polygon", font_size=36, color=C_GOLD)
        hdr.to_edge(UP, buff=0.35)
        self.play(Write(hdr), run_time=0.8)

        # ── Centred tree (no labels) ─────────────────────────────────────────
        sc = 2.6
        tree = mk_tree(sc=sc, ang=0, cx=0, cy=-0.2, col=C_TREE)
        self.play(Create(tree), run_time=1.2)

        rot_note = Text("Can rotate  0° – 360°", font_size=24, color=C_WHT)
        rot_note.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(rot_note), run_time=0.6)
        self.wait(0.5)

        # ── Live rotation with angle counter ────────────────────────────────
        def angle_text(val):
            t = Text(f"{val}°", font_size=48, color=C_GOLD)
            t.move_to([0, 2.85, 0])
            return t

        angle_grp = angle_text(0)
        self.play(FadeIn(angle_grp), run_time=0.4)

        step = 10
        for ang in range(step, 361, step):
            new_t = mk_tree(sc=sc, ang=ang % 360, cx=0, cy=-0.2, col=C_TREE)
            new_grp = angle_text(ang % 360)
            self.play(
                Transform(tree, new_t),
                Transform(angle_grp, new_grp),
                run_time=0.04,
            )

        self.wait(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 3 — Bottom-Left Fill (BLF)
# ═══════════════════════════════════════════════════════════════════════════════
class Scene3_BLF(Scene):
    # Positions from actual BLF run with varied angles:
    #   i_stock=1 → T1 fits, T2 collides → grow to 2×2
    #   i_stock=2 → 6 trees fit, T7 collides → grow to 3×3
    #   i_stock=3 → all 9 trees placed

    VIS_S = 1.5    # screen units per real unit
    OX    = -5.5   # screen x of real (0,0)
    OY    = -3.0   # screen y of real (0,0)

    # (angle, cx, cy) — 6 trees placed in 2×2 box before collision
    FINALS_S2 = [
        (  0, 0.3500, 0.2000),
        (180, 0.3500, 1.8000),
        ( 90, 1.3500, 0.3500),
        (270, 0.6580, 0.8160),
        ( 45, 1.1845, 1.4323),
        (135, 1.7525, 1.5151),
    ]

    # (angle, cx, cy) — 9 trees in final 3×3 box
    FINALS_S3 = [
        (  0, 0.3500, 0.2000),
        (180, 0.3508, 1.7980),
        ( 90, 0.8000, 2.1310),
        (270, 0.8002, 2.5570),
        ( 45, 0.9946, 0.2475),
        (135, 1.6894, 0.5657),
        (225, 1.7968, 1.0241),
        (315, 0.6955, 0.9343),
        (  0, 1.5412, 1.3273),
    ]

    def r2s(self, rx, ry):
        return self.OX + rx * self.VIS_S, self.OY + ry * self.VIS_S

    def _start_cx_cy(self, ang, xct, yct):
        """Screen cx/cy that places tree's top-right bbox corner at (xct, yct)."""
        v = _R(_TV, ang)
        cx = self.OX + xct * self.VIS_S - v[:, 0].max() * self.VIS_S
        cy = self.OY + yct * self.VIS_S - v[:, 1].max() * self.VIS_S
        return cx, cy

    def _box(self, real_side, label_txt):
        w = h = real_side * self.VIS_S
        rect = DashedVMobject(
            Rectangle(width=w, height=h, color=C_BOX, stroke_width=2.5),
            num_dashes=max(24, int(real_side * 26)),
        ).move_to([self.OX + w / 2, self.OY + h / 2, 0])
        lbl = Text(label_txt, font_size=15, color=C_BOX)
        lbl.next_to(rect, UP, buff=0.08)
        return rect, lbl

    def _place_tree(self, ang, cx_r, cy_r, xct, yct, speed=1.0):
        sx, sy = self._start_cx_cy(ang, xct, yct)
        fx, fy = self.r2s(cx_r, cy_r)
        t = mk_tree(sc=self.VIS_S, ang=ang, cx=sx, cy=sy, col=C_NEW, fill=0.75)
        self.play(FadeIn(t), run_time=0.22 / speed)
        if abs(fx - sx) > 0.05:
            self.play(t.animate.shift([fx - sx, 0, 0]), run_time=0.30 / speed)
        if abs(fy - sy) > 0.05:
            self.play(t.animate.shift([0, fy - sy, 0]), run_time=0.30 / speed)
        placed = mk_tree(sc=self.VIS_S, ang=ang, cx=fx, cy=fy, col=C_TREE, fill=0.90)
        self.play(Transform(t, placed), run_time=0.18 / speed)
        return t

    def construct(self):
        self.camera.background_color = BG

        hdr = Text("Bottom-Left Fill  (BLF)", font_size=36, color=C_GOLD)
        hdr.to_edge(UP, buff=0.35)
        self.play(Write(hdr))

        origin_dot = Dot([self.OX, self.OY, 0], color=C_WHT, radius=0.07)
        origin_lbl = Text("(0,0)", font_size=13, color=C_GRY)
        origin_lbl.next_to(origin_dot, DL, buff=0.05)
        self.play(FadeIn(origin_dot), FadeIn(origin_lbl))

        steps = VGroup(
            Text("BLF Algorithm:",                    font_size=18, color=C_GOLD),
            Text("1. Try smallest box first",          font_size=14, color=C_WHT),
            Text("2. Place at top-right corner",       font_size=14, color=C_WHT),
            Text("3. Slide LEFT → wall / tree",        font_size=14, color=C_WHT),
            Text("4. Slide DOWN → floor / tree",       font_size=14, color=C_WHT),
            Text("5. Collision? → grow box, restart",  font_size=14, color=C_RED),
            Text("6. Tree placed  ✓",                  font_size=14, color=C_LITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        steps.move_to([3.2, 0.8, 0])
        self.play(FadeIn(steps, shift=LEFT * 0.2), run_time=0.7)

        # ── Phase 1: i_stock=1 (1×1) ──────────────────────────────────────────
        box1, lbl1 = self._box(1.0, "i_stock=1  (1×1)")
        self.play(Create(box1), FadeIn(lbl1))

        p1 = self._place_tree(0, 0.3500, 0.2000, 1.0, 1.0)

        # T2 (180°) appears at top-right and immediately collides
        sx2, sy2 = self._start_cx_cy(180, 1.0, 1.0)
        t2 = mk_tree(sc=self.VIS_S, ang=180, cx=sx2, cy=sy2, col=C_NEW, fill=0.75)
        self.play(FadeIn(t2), run_time=0.22)
        self.play(t2.animate.set_color(C_RED), p1.animate.set_color(C_RED), run_time=0.25)
        clash1 = Text("COLLISION — box too small!", font_size=16, color=C_RED)
        clash1.move_to([3.2, -1.5, 0])
        hi5 = SurroundingRectangle(steps[5], color=C_RED, buff=0.04, stroke_width=2.0)
        self.play(FadeIn(clash1), Create(hi5), run_time=0.3)
        self.wait(0.3)
        self.play(FadeOut(p1), FadeOut(t2), FadeOut(clash1), FadeOut(hi5),
                  FadeOut(box1), FadeOut(lbl1), run_time=0.3)

        # ── Phase 2: i_stock=2 (2×2) ──────────────────────────────────────────
        box2, lbl2 = self._box(2.0, "i_stock=2  (2×2)")
        self.play(Create(box2), FadeIn(lbl2))
        pl2 = Text("Restarting — 2×2 box", font_size=14, color=C_PUR)
        pl2.move_to([3.2, -1.5, 0])
        self.play(FadeIn(pl2))

        placed2 = []
        for ang, cx_r, cy_r in self.FINALS_S2:
            placed2.append(self._place_tree(ang, cx_r, cy_r, 2.0, 2.0))

        # T7 (225°) collides in 2×2 box
        sx7, sy7 = self._start_cx_cy(225, 2.0, 2.0)
        t7 = mk_tree(sc=self.VIS_S, ang=225, cx=sx7, cy=sy7, col=C_NEW, fill=0.75)
        self.play(FadeIn(t7), run_time=0.22)
        self.play(t7.animate.set_color(C_RED),
                  *[t.animate.set_color(C_RED) for t in placed2], run_time=0.25)
        clash2 = Text("COLLISION — grow to 3×3!", font_size=16, color=C_RED)
        clash2.move_to([3.2, -2.0, 0])
        self.play(FadeIn(clash2), run_time=0.3)
        self.wait(0.3)
        self.play(FadeOut(t7), *[FadeOut(t) for t in placed2],
                  FadeOut(clash2), FadeOut(pl2),
                  FadeOut(box2), FadeOut(lbl2), run_time=0.3)

        # ── Phase 3: i_stock=3 (3×3) ──────────────────────────────────────────
        box3, lbl3 = self._box(3.0, "i_stock=3  (3×3)")
        self.play(Create(box3), FadeIn(lbl3))
        pl3 = Text("Restarting — placing 9 trees in 3×3 box", font_size=14, color=C_PUR)
        pl3.move_to([3.2, -1.5, 0])
        self.play(FadeIn(pl3))

        for ang, cx_r, cy_r in self.FINALS_S3:
            self._place_tree(ang, cx_r, cy_r, 3.0, 3.0, speed=3.0)

        # ── Tight bounding box of placed trees ────────────────────────────────
        # From actual BLF run: x=[0, 2.3624], y=[0, 2.9070]
        BBOX_W, BBOX_H = 2.3624, 2.9070
        bw = BBOX_W * self.VIS_S
        bh = BBOX_H * self.VIS_S
        bbox_rect = DashedVMobject(
            Rectangle(width=bw, height=bh, color=C_GOLD, stroke_width=2.5),
            num_dashes=28,
        ).move_to([self.OX + bw / 2, self.OY + bh / 2, 0])
        bbox_note = Text("actual bbox of trees", font_size=13, color=C_GOLD)
        bbox_note.next_to(bbox_rect, UP, buff=0.06)

        self.play(FadeOut(box3), FadeOut(lbl3), FadeOut(pl3), run_time=0.3)
        self.play(Create(bbox_rect), FadeIn(bbox_note), run_time=0.7)

        # Width and height labels
        w_lbl = Text(f"W = {BBOX_W:.2f}", font_size=13, color=C_GOLD)
        w_lbl.next_to(bbox_rect, DOWN, buff=0.10)
        h_lbl = Text(f"H = {BBOX_H:.2f}", font_size=13, color=C_GOLD)
        h_lbl.rotate(PI / 2)
        h_lbl.next_to(bbox_rect, LEFT, buff=0.10)
        self.play(FadeIn(w_lbl), FadeIn(h_lbl), run_time=0.5)
        self.wait(0.4)

        # ── Square with side = max(W, H) ──────────────────────────────────────
        side = max(BBOX_W, BBOX_H)   # = BBOX_H = 2.9070
        sw = side * self.VIS_S
        sq_rect = Rectangle(width=sw, height=sw, color=C_LITE, stroke_width=2.5)
        sq_rect.move_to([self.OX + sw / 2, self.OY + sw / 2, 0])

        side_lbl = Text(f"side = max(W, H) = {side:.2f}", font_size=14, color=C_LITE)
        side_lbl.move_to([3.2, -1.5, 0])
        self.play(
            ReplacementTransform(bbox_rect, sq_rect),
            FadeOut(bbox_note), FadeOut(w_lbl), FadeOut(h_lbl),
            FadeIn(side_lbl),
            run_time=0.7,
        )
        self.wait(0.3)

        # ── Score ─────────────────────────────────────────────────────────────
        score = side ** 2 / 9
        score_txt = Text(
            f"score = side² / N = {side:.2f}² / 9 = {score:.3f}",
            font_size=16, color=C_GOLD,
        )
        score_txt.move_to([3.2, -2.1, 0])
        minimize = Text("← GA minimizes this", font_size=14, color=C_NEW)
        minimize.next_to(score_txt, DOWN, buff=0.12)
        self.play(Write(score_txt), run_time=0.8)
        self.play(FadeIn(minimize))
        self.wait(1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 4 — Genetic Algorithm
# ═══════════════════════════════════════════════════════════════════════════════
class Scene4_GA(Scene):
    # 4 individuals (3 trees each) showing diverse angle strategies
    POP = [
        [  0.0,  90.0, 270.0],   # mixed — best
        [  0.0,   0.0,   0.0],   # all upright — wasteful
        [ 45.0, 135.0, 225.0],   # diagonal — moderate
        [ 90.0,  90.0,  90.0],   # all sideways — wasteful
    ]
    FIT  = [4.32, 7.56, 6.18, 7.10]
    POS  = [(-4.2, 1.2), (-1.0, 1.2), (-4.2, -1.6), (-1.0, -1.6)]

    def _card(self, px, py, angles, col):
        sc = 0.40
        trees = VGroup(*[
            mk_tree(sc=sc, ang=a, cx=px + (j-1)*0.68, cy=py, col=col, fill=0.80)
            for j, a in enumerate(angles)
        ])
        ang_lbls = VGroup(*[
            Text(f"{int(a % 360)}°", font_size=11, color=C_GRY).next_to(trees[j], DOWN, buff=0.06)
            for j, a in enumerate(angles)
        ])
        return VGroup(trees, ang_lbls)

    def _fit_lbl(self, card, fitness, col=C_GRY):
        lbl = Text(f"score = {fitness:.2f}", font_size=14, color=col)
        lbl.next_to(card, DOWN, buff=0.10)
        return lbl

    def _phase(self, txt, col=C_PUR):
        t = Text(txt, font_size=19, color=col)
        t.move_to([2.8, 2.7, 0])
        return t

    def _desc(self, *lines, y=1.9):
        grp = VGroup(*[Text(l, font_size=13, color=C_WHT) for l in lines])
        grp.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        grp.move_to([2.8, y, 0])
        return grp

    def construct(self):
        self.camera.background_color = BG

        hdr = Text("Genetic Algorithm — Evolving Rotation Angles",
                   font_size=32, color=C_GOLD)
        hdr.to_edge(UP, buff=0.35)
        self.play(Write(hdr), run_time=0.7)

        # ── ① Random Generation ───────────────────────────────────────────────
        pt = self._phase("① Random Generation")
        desc = self._desc(
            "Pop. size = 20 individuals",
            "Individual = [θ₁, θ₂, … θₙ]   (N rotation angles)",
            "θᵢ  ~  Uniform(0°, 360°)",
        )
        self.play(Write(pt), FadeIn(desc, shift=LEFT*0.1), run_time=0.7)

        cards = []
        cols = [C_TREE, C_GRY, C_GRY, C_GRY]
        for (px, py), angles, col in zip(self.POS, self.POP, cols):
            c = self._card(px, py, angles, col)
            cards.append(c)
            self.play(FadeIn(c, shift=UP*0.1), run_time=0.38)
        self.wait(0.3)

        # ── ② BLF Adjustment + Fitness ────────────────────────────────────────
        self.play(FadeOut(pt), FadeOut(desc), run_time=0.25)
        pt = self._phase("② BLF Adjustment + Fitness Evaluation")
        desc = self._desc(
            "BLF packs trees → tight bounding square",
            "score = side² / N     (lower is better)",
        )
        self.play(Write(pt), FadeIn(desc), run_time=0.6)

        fit_lbls = []
        for k, (c, fit) in enumerate(zip(cards, self.FIT)):
            lbl = self._fit_lbl(c, fit, col=C_GOLD if k == 0 else C_GRY)
            fit_lbls.append(lbl)
            self.play(FadeIn(lbl), run_time=0.28)
        self.wait(0.4)

        # ── ③ Ranked Wheel Selection ──────────────────────────────────────────
        self.play(FadeOut(pt), FadeOut(desc), run_time=0.25)
        pt = self._phase("③ Ranked Wheel Selection")
        desc = self._desc(
            "Sort individuals by score (ascending)",
            "Assign rank-based selection probs (sp=1.5)",
            "Better score  →  higher selection probability",
        )
        self.play(Write(pt), FadeIn(desc), run_time=0.6)

        # ranked_order: best first  →  [0, 2, 3, 1]
        ranked = sorted(range(4), key=lambda i: self.FIT[i])
        probs  = [0.375, 0.292, 0.208, 0.125]
        rank_lbls = []
        for rank, idx in enumerate(ranked):
            col = C_GOLD if rank < 2 else C_GRY
            rl = Text(f"#{rank+1}  p={probs[rank]:.3f}", font_size=13, color=col)
            rl.next_to(cards[idx], UP, buff=0.08)
            rank_lbls.append(rl)
            self.play(FadeIn(rl), run_time=0.25)

        pr1 = SurroundingRectangle(cards[ranked[0]], color=C_GOLD, stroke_width=2.5, buff=0.08)
        pr2 = SurroundingRectangle(cards[ranked[1]], color=C_LITE, stroke_width=2.0, buff=0.08)
        pl1 = Text("P1", font_size=13, color=C_GOLD).next_to(pr1, LEFT, buff=0.06)
        pl2 = Text("P2", font_size=13, color=C_LITE).next_to(pr2, LEFT, buff=0.06)
        self.play(Create(pr1), Create(pr2), FadeIn(pl1), FadeIn(pl2), run_time=0.5)
        self.wait(0.4)

        # ── ④ Arithmetic Crossover ────────────────────────────────────────────
        self.play(FadeOut(pt), FadeOut(desc),
                  *[FadeOut(r) for r in rank_lbls],
                  FadeOut(pl1), FadeOut(pl2), run_time=0.25)
        pt = self._phase("④ Arithmetic Crossover  (p_cross = 0.9)")

        p1a    = self.POP[ranked[0]]
        p2a    = self.POP[ranked[1]]
        alpha  = 0.6
        child_a = [alpha * a + (1 - alpha) * b for a, b in zip(p1a, p2a)]

        formula = Text("child = 0.6 · P1  +  0.4 · P2", font_size=15, color=C_WHT)
        formula.move_to([2.8, 2.15, 0])
        self.play(Write(pt), FadeIn(formula), run_time=0.5)

        rows = VGroup(*[
            Text(f"0.6 × {a1:.0f}°  +  0.4 × {a2:.0f}°  =  {ac:.1f}°",
                 font_size=13, color=C_WHT)
            for a1, a2, ac in zip(p1a, p2a, child_a)
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        rows.move_to([2.8, 1.4, 0])
        self.play(LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.3), run_time=0.8)
        self.wait(0.3)

        # Show offspring card — placed between grid and right panel
        CHILD_CX, CHILD_CY = 1.5, -1.0
        child_card = self._card(CHILD_CX, CHILD_CY, [int(a) % 360 for a in child_a], C_PUR)
        offspring_lbl = Text("Offspring", font_size=13, color=C_PUR)
        offspring_lbl.next_to(child_card, UP, buff=0.08)
        self.play(FadeIn(child_card, shift=UP*0.1), FadeIn(offspring_lbl), run_time=0.5)
        self.wait(0.3)

        # ── ⑤ Angle Adjustment Mutation ──────────────────────────────────────
        self.play(FadeOut(pt), FadeOut(formula), FadeOut(rows),
                  FadeOut(pr1), FadeOut(pr2), run_time=0.25)
        pt = self._phase("⑤ Angle Mutation  (p_ind=0.4,  p_gene=0.25)")
        desc = self._desc(
            "Each gene mutated independently with p=0.25",
            "Δθ  ~  Uniform(−30°, +30°)",
            "θ_new  =  (θ + Δθ)  mod  360°",
        )
        self.play(Write(pt), FadeIn(desc), run_time=0.6)

        # Highlight the gene being mutated directly on the offspring card
        mut_idx      = 1   # middle tree
        mut_tree_mob = child_card[0][mut_idx]
        mut_ang_mob  = child_card[1][mut_idx]
        gene_rect    = SurroundingRectangle(mut_tree_mob, color=C_NEW,
                                            stroke_width=2.0, buff=0.06)
        self.play(Create(gene_rect), run_time=0.4)

        old_a = child_a[mut_idx]
        delta = 22.5
        new_a = (old_a + delta) % 360
        demo  = Text(f"{old_a:.1f}°  +  (+{delta:.1f}°)  →  {new_a:.1f}°",
                     font_size=14, color=C_NEW)
        demo.move_to([2.8, 0.5, 0])
        self.play(FadeIn(demo), run_time=0.4)

        # Transform the highlighted tree to the mutated angle in-place
        new_tree_mob = mk_tree(sc=0.40, ang=new_a,
                               cx=CHILD_CX + (mut_idx - 1) * 0.68,
                               cy=CHILD_CY, col=C_PUR, fill=0.80)
        new_ang_lbl  = Text(f"{int(new_a % 360)}°", font_size=11, color=C_NEW)
        new_ang_lbl.next_to(new_tree_mob, DOWN, buff=0.06)
        self.play(Transform(mut_tree_mob, new_tree_mob),
                  Transform(mut_ang_mob, new_ang_lbl),
                  run_time=0.5)
        self.play(FadeOut(gene_rect), run_time=0.2)
        self.wait(0.3)

        child_a_mut        = list(child_a)
        child_a_mut[mut_idx] = new_a

        # ── ⑥ Elitist Replacement ────────────────────────────────────────────
        self.play(FadeOut(pt), FadeOut(desc), FadeOut(demo), run_time=0.25)
        pt = self._phase("⑥ Elitist Replacement")
        desc = self._desc(
            "Pool = current population + new children",
            "Sort merged pool by score",
            "Keep best N=20  —  worst are discarded",
        )
        self.play(Write(pt), FadeIn(desc), run_time=0.6)

        worst_idx = ranked[-1]
        wr = SurroundingRectangle(cards[worst_idx], color=C_RED, stroke_width=2.5, buff=0.08)
        wl = Text("worst — dropped", font_size=13, color=C_RED)
        wl.next_to(wr, UP, buff=0.06)
        self.play(Create(wr), FadeIn(wl), run_time=0.5)
        self.wait(0.3)

        # Offspring moves into the grid slot of the dropped individual
        child_fit = 3.98
        px_w, py_w = self.POS[worst_idx]
        target_card    = self._card(px_w, py_w, [int(a) % 360 for a in child_a_mut], C_LITE)
        new_fit_lbl    = self._fit_lbl(target_card, child_fit, col=C_LITE)
        self.play(
            FadeOut(cards[worst_idx]), FadeOut(fit_lbls[worst_idx]),
            FadeOut(wr), FadeOut(wl),
            FadeOut(child_card), FadeOut(offspring_lbl),
            FadeIn(target_card, shift=UP*0.15), FadeIn(new_fit_lbl),
            run_time=0.7,
        )

        improve = Text(
            f"score {child_fit:.2f}  <  {self.FIT[worst_idx]:.2f}   improvement",
            font_size=15, color=C_LITE,
        )
        improve.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(improve), run_time=0.5)
        self.wait(0.5)

        loop = Text("Repeat up to 200 iterations  (early stop: 20 stagnant)",
                    font_size=15, color=C_PUR)
        loop.to_edge(DOWN, buff=0.45)
        self.play(ReplacementTransform(improve, loop), run_time=0.5)
        self.wait(1.5)


# ═══════════════════════════════════════════════════════════════════════════════
# Scene 5 — Incremental Build
# ═══════════════════════════════════════════════════════════════════════════════
class Scene5_Incremental(Scene):
    # Final angles chosen by the GA for each tree
    ANGLES = [0, 180, 90, 270]
    # Equally-spaced x positions; y fixed at centre
    XS  = [-5.0, -1.8, 1.4, 4.6]
    CY  = -0.3
    SC  = 1.3

    def construct(self):
        self.camera.background_color = BG

        hdr = Text("Incremental Build Strategy", font_size=36, color=C_GOLD)
        hdr.to_edge(UP, buff=0.35)
        self.play(Write(hdr))

        note = Text(
            "GA optimises one new angle at a time — previous trees are locked",
            font_size=18, color=C_GRY,
        )
        note.next_to(hdr, DOWN, buff=0.30)
        self.play(FadeIn(note))

        step_txt = Text("N = 1", font_size=28, color=C_GOLD)
        step_txt.to_edge(RIGHT, buff=1.0).shift(UP * 0.5)
        self.play(Write(step_txt))

        placed = []   # (tree_mob, ang_lbl_mob) for locked trees

        for n, (ang, cx) in enumerate(zip(self.ANGLES, self.XS)):
            # Update counter
            new_step = Text(f"N = {n+1}", font_size=28, color=C_GOLD)
            new_step.to_edge(RIGHT, buff=1.0).shift(UP * 0.5)
            if n > 0:
                self.play(Transform(step_txt, new_step), run_time=0.3)

            # Spawn new tree and run GA angle search
            tree = mk_tree(sc=self.SC, ang=0, cx=cx, cy=self.CY,
                           col=C_PUR, fill=0.55)
            self.play(FadeIn(tree), run_time=0.3)
            self._ga_search(tree, ang, cx, self.CY)

            # Show settled angle label
            ang_lbl = Text(f"{ang}°", font_size=18, color=C_NEW)
            ang_lbl.next_to(tree, DOWN, buff=0.12)
            self.play(FadeIn(ang_lbl), run_time=0.3)
            self.wait(0.35)

            # Lock (dim) all but the last stage
            if n < len(self.ANGLES) - 1:
                locked = mk_tree(sc=self.SC, ang=ang, cx=cx, cy=self.CY,
                                 col=C_DIM, fill=0.70)
                lock_lbl = Text("locked", font_size=13, color=C_GRY)
                lock_lbl.next_to(locked, UP, buff=0.08)
                self.play(Transform(tree, locked), FadeIn(lock_lbl),
                          run_time=0.4)
                self.wait(0.2)
                self.play(FadeOut(ang_lbl), FadeOut(lock_lbl), run_time=0.25)
                placed.append(tree)

        # ── Final summary ─────────────────────────────────────────────────────
        self.wait(0.5)
        summary = Text("Repeats for N = 1 … 200 trees", font_size=24, color=C_PUR)
        summary.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(summary))
        self.wait(1.5)

    def _ga_search(self, tree_mob, final_ang, cx, cy):
        """Spin through candidate angles then settle on final_ang."""
        candidates = [(final_ang + d) % 360 for d in [-75, -45, 30, 60, -20, 0]]
        search_lbl = Text("GA searching…", font_size=16, color=C_PUR)
        search_lbl.next_to(tree_mob, UP, buff=0.12)
        self.play(FadeIn(search_lbl), run_time=0.2)
        for cand in candidates:
            self.play(Transform(tree_mob,
                                mk_tree(sc=self.SC, ang=cand, cx=cx, cy=cy,
                                        col=C_PUR, fill=0.55)),
                      run_time=0.15)
        self.play(Transform(tree_mob,
                            mk_tree(sc=self.SC, ang=final_ang, cx=cx, cy=cy,
                                    col=C_NEW, fill=0.85)),
                  FadeOut(search_lbl), run_time=0.30)


# ═══════════════════════════════════════════════════════════════════════════════
# FullPipeline — all scenes concatenated
# ═══════════════════════════════════════════════════════════════════════════════
class FullPipeline(Scene):
    """Plays all five scenes back-to-back with a brief separator."""

    def construct(self):
        for SceneCls in [
            Scene1_Title,
            Scene2_TreeShape,
            Scene3_BLF,
            Scene4_GA,
            Scene5_Incremental,
        ]:
            prev_cls = self.__class__
            self.__class__ = SceneCls
            SceneCls.construct(self)
            self.__class__ = prev_cls
            self.wait(0.5)
            self.clear()
