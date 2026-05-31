import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

def check_password():
    """簡易パスワード認証"""

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    st.title("Password Required")
    st.write("このアプリは授業中に指定されたタイミングで使用します。")

    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if password == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Password is incorrect.")

    return False


if not check_password():
    st.stop()

# ------------------------------------------------------------
# Matplotlibの日本語文字化け対策
# ------------------------------------------------------------
jp_font_candidates = [
    "Yu Gothic",
    "Meiryo",
    "MS Gothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
]

available_fonts = {f.name for f in fm.fontManager.ttflist}

for font in jp_font_candidates:
    if font in available_fonts:
        plt.rcParams["font.family"] = font
        break

plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------
# Streamlit設定
# ------------------------------------------------------------
st.set_page_config(
    page_title="すべり面探索アプリ",
    layout="wide"
)

st.title("円弧すべり面探索アプリ")
st.caption(
    "多数の円弧すべり面を試し、フェレニウス法による安全率が最小となるすべり面を探索します。"
)

# ------------------------------------------------------------
# 入力
# ------------------------------------------------------------
st.sidebar.header("斜面・地盤条件")

H = st.sidebar.slider("斜面高さ H (m)", 5.0, 30.0, 10.0, 0.5)
slope_angle_deg = st.sidebar.slider("斜面角度 β (度)", 15.0, 60.0, 30.0, 1.0)

gamma = st.sidebar.slider("単位体積重量 γ (kN/m³)", 12.0, 22.0, 18.0, 0.5)
c = st.sidebar.slider("粘着力 c (kPa)", 0.0, 50.0, 10.0, 1.0)
phi_deg = st.sidebar.slider("内部摩擦角 φ (度)", 0.0, 45.0, 30.0, 1.0)

n_slices = st.sidebar.slider("スライス数", 10, 80, 30, 5)

st.sidebar.header("探索範囲")

x_min, x_max = st.sidebar.slider(
    "円中心 x の探索範囲 (m)",
    min_value=-30.0,
    max_value=50.0,
    value=(-15.0, 25.0),
    step=1.0,
)

y_min, y_max = st.sidebar.slider(
    "円中心 y の探索範囲 (m)",
    min_value=-40.0,
    max_value=40.0,
    value=(-20.0, 10.0),
    step=1.0,
)

r_min, r_max = st.sidebar.slider(
    "半径 R の探索範囲 (m)",
    min_value=5.0,
    max_value=100.0,
    value=(10.0, 40.0),
    step=1.0,
)

nx = st.sidebar.slider("x方向の探索分割数", 5, 50, 20, 1)
ny = st.sidebar.slider("y方向の探索分割数", 5, 50, 20, 1)
nr = st.sidebar.slider("半径方向の探索分割数", 5, 50, 15, 1)

# ------------------------------------------------------------
# 斜面形状
# 左が低く、右が高い斜面
#
# x <= 0      : 法尻側水平地盤 y = 0
# 0 < x < L   : 斜面 y = x tanβ
# x >= L      : 天端側水平地盤 y = H
# ------------------------------------------------------------
beta = np.radians(slope_angle_deg)
phi = np.radians(phi_deg)

L = H / np.tan(beta)


def ground_y(x):
    """
    地表面のy座標を返す。
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)

    y[x <= 0] = 0.0

    mask = (x > 0) & (x < L)
    y[mask] = x[mask] * np.tan(beta)

    y[x >= L] = H

    return y


# ------------------------------------------------------------
# 円と地表面の交点探索
# ------------------------------------------------------------
def find_circle_ground_intersections(xc, yc, R):
    """
    円と地表面の交点を数値的に探す。

    有効な場合:
        [(x_left, y_left), (x_right, y_right)]

    無効な場合:
        None

    ここでは、地表面との交点がちょうど2点の円だけを有効とする。
    """
    xs = np.linspace(-2 * H, L + 2 * H, 3000)
    ys = ground_y(xs)

    f = (xs - xc) ** 2 + (ys - yc) ** 2 - R ** 2

    intersections = []

    for i in range(len(xs) - 1):
        if f[i] == 0:
            x_int = xs[i]
            y_int = ground_y(np.array([x_int]))[0]
            intersections.append((x_int, y_int))

        elif f[i] * f[i + 1] < 0:
            x_int = xs[i] - f[i] * (xs[i + 1] - xs[i]) / (f[i + 1] - f[i])
            y_int = ground_y(np.array([x_int]))[0]
            intersections.append((x_int, y_int))

    # 重複に近い交点を整理
    intersections = sorted(intersections, key=lambda p: p[0])

    cleaned = []
    for p in intersections:
        if not cleaned:
            cleaned.append(p)
        else:
            if abs(p[0] - cleaned[-1][0]) > 1.0e-3:
                cleaned.append(p)

    intersections = cleaned

    # 交点がちょうど2点でなければ除外
    if len(intersections) != 2:
        return None

    x_left, y_left = intersections[0]
    x_right, y_right = intersections[1]

    # 交点間距離が小さすぎる円は除外
    if x_right - x_left < 1.0:
        return None

    return [(x_left, y_left), (x_right, y_right)]


# ------------------------------------------------------------
# 円弧
# ------------------------------------------------------------
def circle_lower_y(x, xc, yc, R):
    """
    円の下側円弧のy座標を返す。
    """
    x = np.asarray(x)
    val = R ** 2 - (x - xc) ** 2

    if np.any(val < 0):
        return None

    return yc - np.sqrt(val)


# ------------------------------------------------------------
# 接線方向角によるすべり円判定
# ------------------------------------------------------------
def tangent_angle_deg_on_lower_arc(x, y, xc, yc):
    """
    円弧上の点における接線方向角 θ を求める。

    角度は、x軸正方向を0°として反時計回りに測る。
    返り値は 0°以上360°未満。

    下側円弧を左から右へ進む向きの接線方向を用いる。
    """
    rx = x - xc
    ry = y - yc

    # 下側円弧を左から右へ進む接線ベクトル
    tx = -ry
    ty = rx

    theta = np.degrees(np.arctan2(ty, tx))

    if theta < 0:
        theta += 360.0

    return theta


def is_valid_slip_circle_by_tangent_angle(xc, yc, R):
    """
    地表面との交点数と、交点での接線方向角により、
    すべり円として妥当かを判定する。

    左右反転後の斜面：
        左側：低い水平地盤
        右側：高い水平地盤

    判定条件：
        右側交点では β < θ <= 90°
        左側交点では 270° <= θ < 360°

    条件を満たさない円は探索候補から除外する。
    """
    intersections = find_circle_ground_intersections(xc, yc, R)

    if intersections is None:
        return False

    (x_left, y_left), (x_right, y_right) = intersections

    theta_left = tangent_angle_deg_on_lower_arc(x_left, y_left, xc, yc)
    theta_right = tangent_angle_deg_on_lower_arc(x_right, y_right, xc, yc)

    beta_deg = slope_angle_deg

    right_is_valid = beta_deg < theta_right <= 90.0
    left_is_valid = 270.0 <= theta_left < 360.0

    if not right_is_valid:
        return False

    if not left_is_valid:
        return False

    return True


# ------------------------------------------------------------
# フェレニウス法による安全率計算
# ------------------------------------------------------------
def calculate_fellenius_fs(xc, yc, R):
    """
    フェレニウス法による安全率を計算する。

    不自然なすべり円は None を返し、探索候補から除外する。
    """
    if not is_valid_slip_circle_by_tangent_angle(xc, yc, R):
        return None

    intersections = find_circle_ground_intersections(xc, yc, R)

    if intersections is None:
        return None

    (x1, y1), (x2, y2) = intersections

    xs = np.linspace(x1, x2, n_slices + 1)

    resisting = 0.0
    driving = 0.0

    for i in range(n_slices):
        xa = xs[i]
        xb = xs[i + 1]
        xm = 0.5 * (xa + xb)
        b = xb - xa

        y_top = ground_y(np.array([xm]))[0]
        y_base_array = circle_lower_y(np.array([xm]), xc, yc, R)

        if y_base_array is None:
            return None

        y_base = y_base_array[0]

        # すべり面が地表面より上にある場合は除外
        if y_base >= y_top:
            return None

        height = y_top - y_base

        # 厚さが小さすぎるスライスを含む場合は除外
        if height <= 0:
            return None

        W = gamma * b * height

        # 円弧の接線勾配
        # y = yc - sqrt(R^2 - (x - xc)^2)
        # dy/dx = (x - xc) / sqrt(R^2 - (x - xc)^2)
        sqrt_val = np.sqrt(R ** 2 - (xm - xc) ** 2)

        if sqrt_val <= 0:
            return None

        dydx = (xm - xc) / sqrt_val
        alpha = np.arctan(dydx)

        # スライス底面長
        cos_alpha = np.cos(alpha)

        if cos_alpha <= 0:
            return None

        l = b / cos_alpha

        # フェレニウス法
        # N_i ≒ W_i cosα_i
        N = W * cos_alpha

        resisting += c * l + N * np.tan(phi)

        # 左右反転しても滑動力を正として扱う
        driving += W * abs(np.sin(alpha))

    if driving <= 0:
        return None

    FS = resisting / driving

    # 極端な値は候補から除外
    if FS <= 0 or FS > 20:
        return None

    return FS


# ------------------------------------------------------------
# 初期表示用の斜面図
# ------------------------------------------------------------
def draw_base_slope(ax):
    x_plot = np.linspace(-2 * H, L + 2 * H, 500)
    y_plot = ground_y(x_plot)

    ax.plot(x_plot, y_plot, linewidth=3, label="Ground Surface")
    ax.fill_between(x_plot, y_plot, -1.5 * H, alpha=0.2)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Slope")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-1.2 * H, max(1.5 * H, y_max + 5))


# ------------------------------------------------------------
# アプリ本体
# ------------------------------------------------------------
total_candidates = nx * ny * nr

st.write(
    f"探索候補数：{total_candidates} 個 "
    f"（x方向 {nx} × y方向 {ny} × 半径方向 {nr}）"
)

if st.button("すべり面探索を実行"):
    results = []

    invalid_count = 0
    valid_circle_count = 0

    xcs = np.linspace(x_min, x_max, nx)
    ycs = np.linspace(y_min, y_max, ny)
    Rs = np.linspace(r_min, r_max, nr)

    progress = st.progress(0)
    count = 0

    best = None

    for xc in xcs:
        for yc in ycs:
            for R in Rs:
                FS = calculate_fellenius_fs(xc, yc, R)

                if FS is None:
                    invalid_count += 1
                else:
                    valid_circle_count += 1

                    result = {
                        "xc": xc,
                        "yc": yc,
                        "R": R,
                        "FS": FS,
                    }

                    results.append(result)

                    if best is None or FS < best["FS"]:
                        best = result

                count += 1
                progress.progress(count / total_candidates)

    if best is None:
        st.error(
            "有効なすべり面が見つかりませんでした。"
            "探索範囲、円中心、半径の範囲を見直してください。"
        )

        st.write(f"除外された円：{invalid_count} 個")
        st.write(f"有効な円：{valid_circle_count} 個")

    else:
        st.success("探索が完了しました。")

        st.write(f"有効な円：{valid_circle_count} 個")
        st.write(f"除外された円：{invalid_count} 個")

        col1, col2 = st.columns([1, 1.4])

        with col1:
            st.subheader("最小安全率")

            st.metric("最小安全率 Fmin", f"{best['FS']:.3f}")

            st.write(f"円中心 x: {best['xc']:.2f} m")
            st.write(f"円中心 y: {best['yc']:.2f} m")
            st.write(f"半径 R: {best['R']:.2f} m")

            df = pd.DataFrame(results).sort_values("FS")

            st.subheader("安全率の小さい順")
            st.dataframe(df.head(20), use_container_width=True)

        with col2:
            st.subheader("探索結果の図")

            fig, ax = plt.subplots(figsize=(9, 7))

            x_plot = np.linspace(-2 * H, L + 2 * H, 500)
            y_plot = ground_y(x_plot)

            ax.plot(x_plot, y_plot, linewidth=3, label="Ground Surface")
            ax.fill_between(x_plot, y_plot, -1.5 * H, alpha=0.2)

            # 候補すべり面を薄く表示
            if len(results) > 0:
                step = max(1, len(results) // 100)

                for r in results[::step]:
                    inter = find_circle_ground_intersections(
                        r["xc"], r["yc"], r["R"]
                    )

                    if inter is None:
                        continue

                    (x1, y1), (x2, y2) = inter

                    xs = np.linspace(x1, x2, 100)
                    ys = circle_lower_y(xs, r["xc"], r["yc"], r["R"])

                    if ys is not None:
                        ax.plot(xs, ys, linewidth=0.5, alpha=0.25)

            # 最小安全率のすべり面
            inter = find_circle_ground_intersections(
                best["xc"], best["yc"], best["R"]
            )

            if inter is not None:
                (x1, y1), (x2, y2) = inter

                xs = np.linspace(x1, x2, 300)
                ys = circle_lower_y(xs, best["xc"], best["yc"], best["R"])

                ax.plot(
                    xs,
                    ys,
                    linewidth=4,
                    label="Slope Circle to minimize Fs",
                )

                # 交点
                ax.scatter(
                    [x1, x2],
                    [y1, y2],
                    s=70,
                    label="Intersections",
                )

            # 円中心
            ax.scatter(
                best["xc"],
                best["yc"],
                s=90,
                label="Center of Circle",
            )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title("Searchin for a Splope Circle to minimize Fs")
            ax.grid(True)
            ax.legend()

            ax.set_xlim(-2 * H, L + 2 * H)
            ax.set_ylim(-1.2 * H, max(1.5 * H, y_max + 5))

            st.pyplot(fig)

        st.info(
            "このアプリでは、円中心と半径を格子状に変化させ、多数の円弧すべり面を試しています。"
            "ただし、地表面との交点数や交点での接線方向角が条件を満たさない円は、"
            "安全率計算の前に探索候補から除外しています。"
        )

else:
    st.info("左側の条件を設定し、「すべり面探索を実行」を押してください。")

    fig, ax = plt.subplots(figsize=(9, 6))
    draw_base_slope(ax)
    st.pyplot(fig)

    st.markdown(
        """
        ### このアプリで行っていること

        1. 円中心と半径を少しずつ変えながら、多数の円を作る  
        2. 地表面と2点で交わらない円を除外する  
        3. 交点での接線方向角が不自然な円を除外する  
        4. 残った円についてフェレニウス法の安全率を計算する  
        5. 安全率が最小となる円弧を表示する  
        """
    )
