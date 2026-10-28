from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------
# 1. Theme and output
# ---------------------------------------------------------------------
BLUE = "#3478F6"
NAVY = "#142B4A"
SLATE = "#718096"
MID_GRAY = "#A8B6C8"
PALE = "#E8EEF6"
WHITE = "#FFFFFF"
TEXT = "#26364B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
    "axes.titleweight": "bold",
    "axes.titlesize": 17,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.facecolor": WHITE,
    "figure.facecolor": WHITE,
})

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

OUT_DIR = SCRIPT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(fig, filename):
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight",
                facecolor=WHITE, pad_inches=0.08)
    print(f"Saved: {OUT_DIR / filename}")
    plt.close(fig)


# ---------------------------------------------------------------------
# 2. Auto-size helper: make the canvas aspect match the axes aspect
# ---------------------------------------------------------------------
def autosize_figure(fig, axs, unit=1.1, title_h=1.0, gap=0.35, margin=0.10):
    axs = np.atleast_1d(axs).ravel().tolist()
    n = len(axs)

    xspans = [ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axs]
    yspans = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axs]
    yspan = max(yspans)

    for ax in axs:
        y0, y1 = ax.get_ylim()
        mid = 0.5 * (y0 + y1)
        ax.set_ylim(mid - 0.5 * yspan, mid + 0.5 * yspan)

    width = sum(xspans) * unit + (n - 1) * gap + 2 * margin
    height = yspan * unit + title_h + 2 * margin
    fig.set_size_inches(width, height)

    fig.set_layout_engine("constrained")


def add_title(ax, title, subtitle=None):
    if not subtitle:
        ax.set_title(title, loc="left", color=NAVY, pad=12, fontweight="bold")
        return

    ax.set_title(title, loc="left", color=NAVY, pad=32, fontweight="bold")

    ax.text(0, 1.02, subtitle, transform=ax.transAxes,
            ha="left", va="bottom", color=SLATE, fontsize=10.5)


def arrow(ax, start, end, color=BLUE, lw=2.0, label=None,
          label_offset=(0, 0), fontsize=15, zorder=4, mutation_scale=14):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder,
    )
    ax.add_patch(patch)
    if label:
        mid = (np.asarray(start) + np.asarray(end)) / 2 + np.asarray(label_offset)
        ax.text(mid[0], mid[1], label, color=color, fontsize=fontsize,
                ha="center", va="center", zorder=zorder + 2,
                bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.88, pad=1.2))


def angle_arc(ax, center, radius, start_deg, end_deg, label,
              label_radius=None, color=NAVY, fontsize=15, text_offset=(0, 0)):
    ax.add_patch(Arc(
        center, 2 * radius, 2 * radius, theta1=start_deg, theta2=end_deg,
        color=color, linewidth=1.5, zorder=5,
    ))
    mid = np.deg2rad((start_deg + end_deg) / 2)
    r = radius + 0.20 if label_radius is None else label_radius
    pos = np.asarray(center) + r * np.array([np.cos(mid), np.sin(mid)])
    pos += np.asarray(text_offset)
    ax.text(*pos, label, color=color, fontsize=fontsize,
            ha="center", va="center", zorder=8,
            bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.9, pad=0.6))


def reference_line(ax, start, end, color=MID_GRAY, lw=1.2):
    ax.plot([start[0], end[0]], [start[1], end[1]],
            linestyle=(0, (4, 3)), color=color, linewidth=lw, zorder=1)


def setup_2d(ax, xlim, ylim, axes=True, equal=True):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if axes:
        arrow(ax, (xlim[0] + 0.08, 0), (xlim[1] - 0.08, 0),
              color=PALE, lw=1.1, mutation_scale=10, zorder=0)
        arrow(ax, (0, ylim[0] + 0.08), (0, ylim[1] - 0.08),
              color=PALE, lw=1.1, mutation_scale=10, zorder=0)


def joint(ax, point, size=115, color=NAVY):
    ax.scatter(*point, s=size, color=color, edgecolors=WHITE,
               linewidths=1.3, zorder=10)


def legend_style(ax, handles, **kwargs):
    defaults = dict(
        frameon=True, fancybox=True, framealpha=0.97,
        facecolor=WHITE, edgecolor=PALE, fontsize=9.5,
        borderpad=0.9, labelspacing=0.75, handlelength=2.0,
    )
    defaults.update(kwargs)
    leg = ax.legend(handles=handles, **defaults)
    leg.get_frame().set_linewidth(1.0)
    return leg


# ---------------------------------------------------------------------
# Figure 1: Vector addition and subtraction
# ---------------------------------------------------------------------
def fig1_vector_add_sub():
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    a = np.array([2.0, 1.0])
    b = np.array([1.0, 2.5])

    ax = axes[0]
    # 由 4.2 收窄至 4.0
    setup_2d(ax, (-0.5, 4.0), (-0.5, 4.0))
    ax.plot([a[0], a[0] + b[0]], [a[1], a[1] + b[1]],
            ls=(0, (4, 3)), color=MID_GRAY, lw=1.3, zorder=1)
    ax.plot([b[0], a[0] + b[0]], [b[1], a[1] + b[1]],
            ls=(0, (4, 3)), color=MID_GRAY, lw=1.3, zorder=1)
    arrow(ax, (0, 0), a, BLUE, label=r"$\mathbf{a}$", label_offset=(-0.18, 0.18))
    arrow(ax, (0, 0), b, SLATE, label=r"$\mathbf{b}$", label_offset=(0.18, -0.12))
    arrow(ax, (0, 0), a + b, NAVY, lw=2.4,
          label=r"$\mathbf{a}+\mathbf{b}$", label_offset=(0.22, 0.18))
    add_title(ax, "Vector addition", "Parallelogram rule")

    ax = axes[1]
    setup_2d(ax, (-0.5, 4.0), (-0.5, 4.0))
    arrow(ax, (0, 0), a, BLUE, label=r"$\mathbf{a}$", label_offset=(-0.18, 0.18))
    arrow(ax, (0, 0), b, SLATE, label=r"$\mathbf{b}$", label_offset=(0.18, -0.12))
    arrow(ax, b, a, NAVY, lw=2.4, label=r"$\mathbf{a}-\mathbf{b}$",
          label_offset=(0.25, -0.25))
    add_title(ax, "Vector subtraction", "Difference between vector tips")

    autosize_figure(fig, axes, unit=1.0, title_h=1.1)
    save_figure(fig, "vector_add_sub.png")


# ---------------------------------------------------------------------
# Figure 2: Dot product and projection
# ---------------------------------------------------------------------
def fig2_dot_projection():
    fig, ax = plt.subplots(constrained_layout=True)
    # 原范围最大到 5.0 与 4.0，极度空旷，收紧至 4.5 与 3.5
    setup_2d(ax, (-0.4, 4.5), (-0.4, 3.5))
    a = np.array([2.0, 3.0])
    b = np.array([4.0, 0.5])
    proj = (np.dot(a, b) / np.dot(b, b)) * b

    reference_line(ax, a, proj, color=SLATE)
    arrow(ax, (0, 0), b, SLATE, label=r"$\mathbf{b}$", label_offset=(0.18, -0.22))
    arrow(ax, (0, 0), proj, NAVY, lw=2.5,
          label=r"$\mathrm{proj}_{\mathbf{b}}\mathbf{a}$", label_offset=(0.1, -0.32),
          fontsize=12.5)
    arrow(ax, (0, 0), a, BLUE, label=r"$\mathbf{a}$", label_offset=(-0.18, 0.2))
    ax.text(*(0.5 * (a + proj) + np.array([0.18, 0.02])), "perpendicular",
            color=SLATE, fontsize=9.5, rotation=0,
            bbox=dict(facecolor=WHITE, edgecolor="none", alpha=0.9, pad=1.5))

    ang_b = np.degrees(np.arctan2(b[1], b[0]))
    ang_a = np.degrees(np.arctan2(a[1], a[0]))
    angle_arc(ax, (0, 0), 0.72, ang_b, ang_a, r"$\theta$", label_radius=0.98)

    add_title(ax, "Dot product and projection",
              r"$\mathbf{a}\cdot\mathbf{b}=\|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta$")

    autosize_figure(fig, ax, unit=1.3, title_h=1.0)
    save_figure(fig, "dot_projection.png")


# ---------------------------------------------------------------------
# Figure 3: 2D rotation
# ---------------------------------------------------------------------
def fig3_rotation_2d():
    fig, ax = plt.subplots(constrained_layout=True)
    # 原范围 [-2.1, 2.5]，缩小至 [-1.2, 1.9]，紧贴图形与文字标签
    setup_2d(ax, (-1.2, 1.9), (-1.2, 1.9))
    theta = np.deg2rad(60)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    v = np.array([1.6, 0.2])
    v_rot = R @ v

    ax.add_patch(Circle((0, 0), 1.0, fill=False, color=PALE,
                        linewidth=1.2, linestyle=(0, (4, 3)), zorder=0))
    arrow(ax, (0, 0), v, BLUE, label=r"$\mathbf{v}$", label_offset=(0.12, -0.22))
    arrow(ax, (0, 0), v_rot, NAVY, lw=2.4,
          label=r"$\mathbf{v}'=\mathbf{R}(\theta)\mathbf{v}$",
          label_offset=(-0.2, 0.22), fontsize=12.5)

    ang_v = np.degrees(np.arctan2(v[1], v[0]))
    ang_rot = np.degrees(np.arctan2(v_rot[1], v_rot[0]))
    angle_arc(ax, (0, 0), 0.68, ang_v, ang_rot, r"$\theta$",
              label_radius=0.95)
    add_title(ax, "Rotation in 2D", r"$\mathbf{v}'=\mathbf{R}(\theta)\mathbf{v}$")

    autosize_figure(fig, ax, unit=1.6, title_h=1.0)
    save_figure(fig, "rotation_2d.png")


# ---------------------------------------------------------------------
# Figure 4: Two-link planar arm
# ---------------------------------------------------------------------
def fig4_two_link_arm():
    fig, ax = plt.subplots(constrained_layout=True)
    L1, L2 = 2.0, 1.5
    theta1_deg, theta2_deg = 35, 50
    theta1, theta2 = np.deg2rad([theta1_deg, theta2_deg])

    p0 = np.array([0.0, 0.0])
    p1 = p0 + L1 * np.array([np.cos(theta1), np.sin(theta1)])
    theta12 = theta1 + theta2
    p2 = p1 + L2 * np.array([np.cos(theta12), np.sin(theta12)])

    # 原先 x 轴画到了 4.7，y 轴到 3.35，但内容最右才到 x=2
    setup_2d(ax, (-0.6, 2.8), (-0.6, 3.1), axes=False)
    arrow(ax, (-0.05, 0), (2.55, 0), MID_GRAY, lw=1.2, mutation_scale=11)
    arrow(ax, (0, -0.05), (0, 2.9), MID_GRAY, lw=1.2, mutation_scale=11)
    ax.text(2.58, -0.12, r"$x$", fontsize=15, color=SLATE)
    ax.text(-0.15, 2.94, r"$y$", fontsize=15, color=SLATE)

    reference_line(ax, p0, (2.0, 0))
    link1_direction = np.array([np.cos(theta1), np.sin(theta1)])
    reference_line(ax, p1, p1 + 1.25 * link1_direction)

    ax.plot(*zip(p0, p1), color=BLUE, linewidth=10,
            solid_capstyle="round", zorder=3)
    ax.plot(*zip(p1, p2), color=SLATE, linewidth=10,
            solid_capstyle="round", zorder=3)

    joint(ax, p0, 155, NAVY)
    joint(ax, p1, 135, BLUE)
    joint(ax, p2, 155, NAVY)

    off1 = np.array([-0.12, 0.17])
    arrow(ax, p0 + off1, p1 + off1, BLUE, lw=1.6, mutation_scale=13)
    arrow(ax, p1 + np.array([0.08, 0.19]),
          p2 + np.array([0.08, 0.19]), SLATE, lw=1.6, mutation_scale=13)

    m1 = (p0 + p1) / 2 + off1 + np.array([-0.12, 0.14])
    m2 = (p1 + p2) / 2 + np.array([0.08, 0.19]) + np.array([0.0, 0.14])
    ax.text(*m1, r"$\mathbf{p}_1$", color=BLUE, fontsize=17, ha="center")
    ax.text(*m2, r"$\mathbf{d}_2$", color=SLATE, fontsize=17, ha="center")

    angle_arc(ax, p0, 0.56, 0, theta1_deg, r"$\theta_1$",
              label_radius=0.82, fontsize=15)
    angle_arc(ax, p1, 0.42, theta1_deg, theta1_deg + theta2_deg,
              r"$\theta_2$", label_radius=0.66, fontsize=15,
              text_offset=(0.04, 0.04))

    ax.text(p0[0] - 0.06, p0[1] - 0.28, "base", color=NAVY,
            fontsize=11.5, fontweight="bold", ha="center")
    ax.text(p1[0] - 0.05, p1[1] + 0.25, "joint 1", color=BLUE,
            fontsize=11, fontweight="bold", ha="center")
    ax.text(p2[0] + 0.12, p2[1] + 0.08, "end-effector",
            color=NAVY, fontsize=11, ha="left")
    ax.text(p2[0] + 0.12, p2[1] - 0.12, r"$\mathbf{p}_{ee}$",
            color=NAVY, fontsize=15, ha="left")

    mid_L1 = (p0 + p1) / 2 + np.array([0.20, -0.20])
    mid_L2 = (p1 + p2) / 2 + np.array([0.13, -0.20])
    ax.text(*mid_L1, r"$L_1$", color=BLUE, fontsize=18, ha="center")
    ax.text(*mid_L2, r"$L_2$", color=SLATE, fontsize=18, ha="center")

    handles = [
        Line2D([0], [0], color=BLUE, lw=5, solid_capstyle="round", label=r"Link 1 ($L_1$)"),
        Line2D([0], [0], color=SLATE, lw=5, solid_capstyle="round", label=r"Link 2 ($L_2$)"),
        Line2D([0], [0], marker="o", linestyle="None", color=NAVY,
               markeredgecolor=WHITE, markersize=8, label="Base / end-effector"),
        Line2D([0], [0], marker="o", linestyle="None", color=BLUE,
               markeredgecolor=WHITE, markersize=8, label="Joint 1"),
    ]
    legend_style(ax, handles, loc="lower right", fontsize=9.0,
                 borderpad=0.8, labelspacing=0.65)

    add_title(ax, "Two-link planar arm: forward kinematics",
              r"$\theta_2$ is measured relative to link 1")

    autosize_figure(fig, ax, unit=1.6, title_h=1.1)
    save_figure(fig, "two_link_arm.png")


# ---------------------------------------------------------------------
# Figure 5: Transform order
# ---------------------------------------------------------------------
def fig5_transform_order():
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    p = np.array([1.0, 0.0])
    t = np.array([2.0, 0.0])
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    ax = axes[0]
    setup_2d(ax, (-0.6, 3.3), (-0.6, 3.4))
    rp = R @ p
    r_p_plus_t = rp + t
    arrow(ax, (0, 0), p, SLATE, label=r"$\mathbf{p}$", label_offset=(0.12, -0.22))
    arrow(ax, (0, 0), rp, BLUE, label=r"$\mathbf{R}\mathbf{p}$", label_offset=(-0.3, 0.1))
    arrow(ax, (0, 0), r_p_plus_t, NAVY, lw=2.4,
          label=r"$\mathbf{R}\mathbf{p}+\mathbf{t}$",
          label_offset=(0.25, 0.22), fontsize=12)
    add_title(ax, "Rotate, then translate", r"$\mathbf{p}'=\mathbf{R}\mathbf{p}+\mathbf{t}$")

    ax = axes[1]
    setup_2d(ax, (-0.6, 3.3), (-0.6, 3.4))
    p_plus_t = p + t
    r_pplus = R @ p_plus_t
    arrow(ax, (0, 0), p, SLATE, label=r"$\mathbf{p}$", label_offset=(0.12, -0.22))
    arrow(ax, (0, 0), p_plus_t, BLUE, label=r"$\mathbf{p}+\mathbf{t}$",
          label_offset=(0.2, -0.25), fontsize=12.5)
    arrow(ax, (0, 0), r_pplus, NAVY, lw=2.4,
          label=r"$\mathbf{R}(\mathbf{p}+\mathbf{t})$",
          label_offset=(-0.35, 0.25), fontsize=12)
    add_title(ax, "Translate, then rotate", r"$\mathbf{p}'=\mathbf{R}(\mathbf{p}+\mathbf{t})$")

    autosize_figure(fig, axes, unit=1.1, title_h=1.05)
    save_figure(fig, "transform_order.png")


# ---------------------------------------------------------------------
# Figure 6: Homogeneous transform
# ---------------------------------------------------------------------
def fig6_homogeneous_transform():
    fig, ax = plt.subplots(constrained_layout=True)
    # 原先范围到 [4.25, 3.6]，现在收窄到 [3.0, 2.4]
    setup_2d(ax, (-0.3, 3.0), (-0.3, 2.4))

    arrow(ax, (0, 0), (1.15, 0), NAVY, lw=1.6,
          label=r"$x_0$", label_offset=(0.0, -0.20), fontsize=12)
    arrow(ax, (0, 0), (0, 1.15), NAVY, lw=1.6,
          label=r"$y_0$", label_offset=(-0.20, 0.0), fontsize=12)

    origin1 = np.array([2.0, 1.0])
    theta = np.deg2rad(40)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    x1 = R @ np.array([1.0, 0.0])
    y1 = R @ np.array([0.0, 1.0])

    arrow(ax, (0, 0), origin1, SLATE, lw=1.8,
          label=r"$\mathbf{t}$", label_offset=(-0.18, 0.18))
    joint(ax, origin1, 70, BLUE)
    arrow(ax, origin1, origin1 + x1, BLUE, lw=1.6,
          label=r"$x_1$", label_offset=(0.12, -0.16), fontsize=12)
    arrow(ax, origin1, origin1 + y1, BLUE, lw=1.6,
          label=r"$y_1$", label_offset=(-0.18, 0.12), fontsize=12)

    p_local = np.array([0.8, 0.6])
    p_world = origin1 + R @ p_local
    reference_line(ax, origin1, p_world, color=PALE, lw=1.3)
    joint(ax, p_world, 95, NAVY)
    ax.text(p_world[0] + 0.12, p_world[1] + 0.12,
            r"$\tilde{\mathbf{p}}$", color=NAVY, fontsize=16)

    add_title(ax, "Homogeneous coordinate transform",
              r"$\tilde{\mathbf{p}}_0=\mathbf{R}_{01}\mathbf{p}_1+\mathbf{t}_{01}$")

    autosize_figure(fig, ax, unit=1.6, title_h=1.0)
    save_figure(fig, "homogeneous_transform.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    generators = [
        fig1_vector_add_sub,
        fig2_dot_projection,
        fig3_rotation_2d,
        fig4_two_link_arm,
        fig5_transform_order,
        fig6_homogeneous_transform,
    ]
    for generate in generators:
        generate()
    print("\nAll six figures were generated.")