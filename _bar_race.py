# -*- coding: utf-8 -*-
"""
动态排名条形图（Bar Chart Race）· 平滑升降版 v2
数据：20 名员工在 Q1–Q5 的综合考核得分演变

v2 修正：
  1) y 由弹簧阻尼驱动 -> 名次交换时柱子逐渐上升/下降，不再瞬移
  2) 名次号改用"视觉顺序"生成 -> 自上而下永远是 1,2,3… 不再错乱
  3) 相邻柱过近时文字自动淡出 -> 换位过程中标签不再糊成一团
"""

import os, glob, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.patheffects as pe
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "动画演示")
os.makedirs(OUT_DIR, exist_ok=True)

# ==================== 动画调参区 ====================
STIFF = 0.18        # 弹簧刚度：越大追得越快，越小升降越从容
DAMP = 0.80         # 阻尼：越大越稳，越小越"弹"
MAX_VEL = 0.35      # 单帧最大位移（格），限制冲刺速度
FRAMES_PER = 26     # 每个季度区间的过渡帧数
SETTLE_FRAMES = 26  # 末尾静止帧，让弹簧收敛到位
FPS = 20
TOP_N = 12          # 可见槽位数
BAR_H = 0.66        # 柱高
FADE_NEAR = 0.92    # 相邻柱距小于此值时，文字开始淡出
FADE_MIN = 0.32     # 淡出下限透明度
SETTLE_ERR = 0.22   # 距目标槽位误差小于此值才显示名次号
# ====================================================


def find_cjk_font():
    preferred = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simkai.ttf",
    ]
    found = [p for p in preferred if os.path.exists(p)]
    if not found:
        for f in glob.glob(r"C:\Windows\Fonts\*"):
            n = os.path.basename(f).lower()
            if n.endswith((".ttf", ".ttc", ".otf")) and \
               any(k in n for k in ("msyh", "simhei", "simsun", "deng", "kai", "yahei")):
                found.append(f)
    for path in found:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            if any(f.name == name for f in font_manager.fontManager.ttflist):
                return name, path
        except Exception:
            continue
    return None, None


CJK_NAME, CJK_PATH = find_cjk_font()
print(f"中文字体: {CJK_NAME}  ({CJK_PATH})")
if CJK_NAME:
    plt.rcParams["font.family"] = [CJK_NAME]
plt.rcParams["axes.unicode_minus"] = False


EMPLOYEES = [
    "沈亦舟", "苏晚棠", "陆见深", "温以宁", "顾清和",
    "林砚白", "江照野", "许青梧", "裴修远", "宋知微",
    "谢听澜", "傅明川", "叶疏桐", "韩慕之", "卫子谦",
    "程若雪", "罗景行", "钟灵韵", "梁墨言", "方序安",
]
PERIODS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
N_EMP = len(EMPLOYEES)

rnd = random.Random(20250)
base = {n: rnd.uniform(66, 90) for n in EMPLOYEES}
trend = {n: rnd.uniform(-4.0, 5.0) for n in EMPLOYEES}
data = np.zeros((N_EMP, len(PERIODS)))
for i, n in enumerate(EMPLOYEES):
    for t in range(len(PERIODS)):
        v = base[n] + trend[n] * t + rnd.gauss(0, 2.2)
        data[i, t] = float(np.clip(v, 55, 100))

cmap = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(N_EMP)]


def ease(t):
    return t * t * (3 - 2 * t)


def on_color(rgba):
    """按背景亮度选择前景色，保证条形上的文字可读。"""
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#FFFFFF" if lum < 0.62 else "#2B2B2B"


def bold(color, lw=0.8):
    """SimHei 无粗体字面，用描边模拟加粗。"""
    return [pe.withStroke(linewidth=lw, foreground=color)]


def visual_ranks(y, visible):
    """只在实际画出来的柱子之间排名次：
    保证屏幕上的编号连续，且自上而下严格递增，不出现 9 号柱标成 10。"""
    r = np.zeros(len(y), dtype=int)
    for k, i in enumerate(reversed(visible)):   # 最高的排第 1
        r[i] = k
    return r


FIG_W, FIG_H = 10.0, 6.4
DPI = 82
Y_BOTTOM = -1.15
X_LEFT = -9.0

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.subplots_adjust(left=0.115, right=0.965, top=0.875, bottom=0.075)

TOTAL_FRAMES = (len(PERIODS) - 1) * FRAMES_PER + 1 + SETTLE_FRAMES
print(f"总帧数: {TOTAL_FRAMES}  (过渡 {TOTAL_FRAMES - SETTLE_FRAMES} + 收敛 {SETTLE_FRAMES})")

y_pos = np.zeros(N_EMP)
y_vel = np.zeros(N_EMP)
initialized = False
y_history, rank_history = [], []
frames = []

for f in range(TOTAL_FRAMES):
    # ---- 得分插值 ----
    tg = f / FRAMES_PER
    p0 = int(min(tg, len(PERIODS) - 2))
    local = ease(min(max(tg - p0, 0.0), 1.0))
    vals = data[:, p0] * (1 - local) + data[:, p0 + 1] * local

    # ---- 真实名次 -> 目标槽位 ----
    order = np.argsort(vals)[::-1]
    rank_of = np.empty(N_EMP, dtype=int)
    rank_of[order] = np.arange(N_EMP)
    target_y = (TOP_N - 1) - rank_of.astype(float)

    # ---- 弹簧阻尼：逐渐上升 / 下降 ----
    if not initialized:
        y_pos = target_y.copy()
        y_vel[:] = 0.0
        initialized = True
    else:
        y_vel = (y_vel + (target_y - y_pos) * STIFF) * DAMP
        y_vel = np.clip(y_vel, -MAX_VEL, MAX_VEL)
        y_pos = y_pos + y_vel

    y_history.append(y_pos.copy())
    rank_history.append(rank_of.copy())

    # ---- 视觉名次 + 邻距（决定文字淡出）----
    order_by_y = np.argsort(y_pos)          # 升序，稍后按此绘制
    visible = [i for i in order_by_y if y_pos[i] > Y_BOTTOM - 0.7]
    visible_set = set(visible)
    vrank = visual_ranks(y_pos, visible)
    gap_to_neighbor = np.full(N_EMP, 99.0)
    for k in range(len(order_by_y)):
        i = order_by_y[k]
        if k > 0:
            gap_to_neighbor[i] = min(gap_to_neighbor[i],
                                     abs(y_pos[i] - y_pos[order_by_y[k - 1]]))
        if k < len(order_by_y) - 1:
            gap_to_neighbor[i] = min(gap_to_neighbor[i],
                                     abs(y_pos[order_by_y[k]] - y_pos[i]))

    def text_alpha(i):
        g = gap_to_neighbor[i]
        if g >= FADE_NEAR:
            return 1.0
        return max(FADE_MIN, (g - 0.35) / (FADE_NEAR - 0.35))

    # 名次号只在柱子"落位"后才显示：
    # 滑行途中名次本身是模糊的（旧名次与新名次之间），显示任何一个都会
    # 与旁边的分数字段自相矛盾，因此干脆等它停稳再亮出来。
    err = np.abs(y_pos - target_y)
    rank_alpha = np.clip(1.0 - err / SETTLE_ERR, 0.0, 1.0)

    # ---- 绘制 ----
    ax.clear()
    ax.set_xlim(X_LEFT, 108)
    ax.set_ylim(Y_BOTTOM, TOP_N - 0.05)

    for idx in order_by_y:
        if idx not in visible_set:
            continue
        y = y_pos[idx]
        v, c = vals[idx], colors[idx]
        fg = on_color(c)
        al = text_alpha(idx)

        ax.barh(y, v, height=BAR_H, color=c,
                edgecolor="white", linewidth=1.1, zorder=3 + (y * 0.001), clip_on=True)

        ax.text(-2.2, y, f"{vrank[idx] + 1}", ha="right", va="center",
                fontsize=12, color="#9E2B25", zorder=6, clip_on=True,
                alpha=min(al, rank_alpha[idx]),
                path_effects=bold("#9E2B25"))
        ax.text(v - 1.8, y, EMPLOYEES[idx], ha="right", va="center",
                fontsize=12.5, color=fg, zorder=6, clip_on=True, alpha=al,
                path_effects=bold(fg))
        ax.text(v + 1.6, y, f"{v:.1f}", ha="left", va="center",
                fontsize=11.5, color="#2B2B2B", zorder=6, clip_on=True, alpha=al,
                path_effects=bold("#2B2B2B"))

    ax.set_yticks([])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=10, colors="#6B6B6B")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)

    tgc = min(tg, len(PERIODS) - 1)
    label = PERIODS[int(round(tgc))] if abs(tgc - round(tgc)) < 1e-9 \
        else f"{PERIODS[p0]} → {PERIODS[p0 + 1]}"

    ax.text(0, 1.115, "员工综合考核得分 · 动态排名",
            transform=ax.transAxes, fontsize=19,
            color="#1F3864", va="bottom", path_effects=bold("#1F3864", 1.0))
    ax.text(1.0, 1.115, label, transform=ax.transAxes, fontsize=16,
            color="#9E2B25", va="bottom", ha="right",
            path_effects=bold("#9E2B25", 0.9))
    ax.text(0, 1.035, "满分 100　·　业绩 40% + 能力 30% + 态度 20% + 出勤 10%",
            transform=ax.transAxes, fontsize=9.5, color="#8A8A8A", va="bottom")

    fig.canvas.draw()
    frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB"))

    if f % 30 == 0 or f == TOTAL_FRAMES - 1:
        print(f"  渲染 {f + 1}/{TOTAL_FRAMES}")

plt.close(fig)

# ---------------- 诊断 ----------------
Y = np.array(y_history)
delta = np.abs(np.diff(Y, axis=0))
print(f"\n[平滑度诊断]")
print(f"  单帧最大位移 : {delta.max():.3f} 格  (旧版瞬移 ≈ 1.000, 上限 {MAX_VEL})")
print(f"  平均单帧位移 : {delta.mean():.4f} 格")
print(f"  结论: {'平滑升降 ✅' if delta.max() <= MAX_VEL + 1e-6 else '仍有跳变 ❌'}")

R = np.array(rank_history)
churn = np.abs(np.diff(R, axis=0)).sum(axis=1)
peak = int(np.argmax(churn))
print(f"\n[换位过程取样] churn 峰值在第 {peak} 帧")
for k, fr in enumerate([max(0, peak - 4), peak, min(len(frames) - 1, peak + 4)], 1):
    p = os.path.join(OUT_DIR, f"换位过程_{k}.png")
    frames[fr].save(p)
    print(f"  帧 #{fr} -> {os.path.basename(p)}")

# ---------------- 输出 ----------------
gif_path = os.path.join(OUT_DIR, "员工考核动态排名.gif")
frames[0].save(gif_path, save_all=True, append_images=frames[1:],
               duration=int(1000 / FPS), loop=0, optimize=True)
png_path = os.path.join(OUT_DIR, "末帧预览.png")
frames[-1].save(png_path)

fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=100)
fig2.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.10)
xs = np.arange(len(PERIODS))
for idx in np.argsort(data[:, -1])[::-1][:6]:
    ax2.plot(xs, data[idx], marker="o", linewidth=2.3,
             color=colors[idx], label=EMPLOYEES[idx])
ax2.set_xticks(xs)
ax2.set_xticklabels(PERIODS, fontsize=11)
ax2.set_ylabel("综合得分", fontsize=11)
ax2.set_title("前六名员工得分趋势（Q1–Q5）", fontsize=15,
              color="#1F3864", pad=14, path_effects=bold("#1F3864", 1.0))
ax2.grid(linestyle=":", alpha=0.4)
ax2.legend(ncol=2, fontsize=10, frameon=False)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
fig2.savefig(os.path.join(OUT_DIR, "得分趋势.png"))
plt.close(fig2)

size_mb = os.path.getsize(gif_path) / 1024 / 1024
print(f"\nGIF : {gif_path}")
print(f"      {len(frames)} 帧, {FPS}fps, {size_mb:.1f} MB")
print(f"PNG : {png_path}")
