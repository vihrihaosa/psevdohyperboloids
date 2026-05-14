# -*- coding: utf-8 -*-
"""
ФИНАЛЬНЫЙ СКРИПТ.

Рядные псевдопараболоиды n-го порядка по архитектуре финального скрипта
рядных псевдогиперболоидов, но с другой базовой образующей.

БАЗОВАЯ ГЕОМЕТРИЯ ПСЕВДОПАРАБОЛОИДА
------------------------------------
Вертикальный тип строится из двух одинаковых зеркальных параболических ветвей,
соединённых в общей вершине на линии фокусов:

    y(x) = sqrt(4 f |x|) = 2 sqrt(f |x|),        |x| <= a,
    a = R^2 / (4 f).

Ось вращения проходит параллельно оси x на расстоянии R от линии фокусов.
Радиальная дистанция от оси вращения до образующей:

    d_v(x) = R - 2 sqrt(f |x|),                  |x| <= a.

Горизонтальный тип — та же параболическая образующая, повернутая на 90°:

    d_h(u) = (R - |u|)^2 / (4 f),                |u| <= R.

РЕКУРСИЯ ПОРЯДКА
----------------
Порядок 2:

    [0, d]

Следующий порядок для каждого интервала [lo, hi]:

    [R_k - hi, R_k - lo] и [R_k + lo, R_k + hi],

с последующим Merge радиальных интервалов. Это означает:
- сами порождающие тороидальные компоненты не удаляются;
- удаляется только дублирование объёма в местах пересечения / вложения;
- итоговая фигура есть объединение всех объёмов;
- если между компонентами есть пустой зазор, скрипт ничего не дорисовывает.

ПОДДЕРЖИВАЕТСЯ
--------------
- вертикальный тип;
- горизонтальный тип;
- все порядки от 2 до n;
- 2D меридиональные сечения общего объёма;
- 3D поверхности общего объёма вращения;
- стек / ряд m одинаковых экземпляров на общей оси;
- параметр h: h > 0 разносит ряды, h = 0 даёт касание, h < 0 даёт перекрытие;
- защита от искусственных прямых соединений в 2D и 3D.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# НАСТРОЙКИ ОТОБРАЖЕНИЯ
# ------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "font.family": "DejaVu Sans",
})

SURFACE_ALPHA = 0.42
MERGE_TOL = 1.0e-9
NPHI = 220

# ВАЖНО: заливки по умолчанию нет. fill_between / fill_betweenx не используются,
# чтобы не появлялись искусственные замыкающие отрезки.
FILL_2D_AREAS = False


# ------------------------------------------------------------
# БАЗОВАЯ ПАРАБОЛИЧЕСКАЯ ГЕОМЕТРИЯ
# ------------------------------------------------------------
def parabola_a(f: float, R: float) -> float:
    """Полудлина вертикального типа / экваториальный радиус горизонтального типа."""
    if f <= 0 or R <= 0:
        raise ValueError("Параметры f и R должны быть положительными")
    return (R * R) / (4.0 * f)


def vertical_distance(abs_s: np.ndarray, f: float, R: float) -> np.ndarray:
    """
    Вертикальный псевдопараболоид:
        d_v(|s|) = R - 2*sqrt(f*|s|),  0 <= |s| <= a.
    """
    abs_s = np.asarray(abs_s, dtype=float)
    d = R - 2.0 * np.sqrt(np.maximum(0.0, f * abs_s))
    return np.maximum(d, 0.0)


def horizontal_distance(u: np.ndarray, f: float, R: float) -> np.ndarray:
    """
    Горизонтальный псевдопараболоид:
        d_h(u) = (R - |u|)^2 / (4*f),  |u| <= R.
    """
    u = np.asarray(u, dtype=float)
    d = ((R - np.abs(u)) ** 2) / (4.0 * f)
    return np.maximum(d, 0.0)


def _ensure_zero_in_axis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if np.any(np.isclose(arr, 0.0, atol=1e-14)):
        return arr
    arr2 = np.concatenate([arr, np.array([0.0])])
    arr2.sort()
    return arr2


def vertical_base_grid(f: float, R: float, npts: int = 900) -> Dict[str, np.ndarray]:
    """
    Базовая сетка вертикального типа.

    Ось фигуры: s in [-a, a].
    Радиальный профиль: d_v(s).
    """
    a = parabola_a(f, R)
    s = np.linspace(-a, a, npts)
    s = _ensure_zero_in_axis(s)
    d = vertical_distance(np.abs(s), f, R)
    return {
        "axis": s,
        "d": d,
        "L": float(a),
        "a": float(a),
        "max_distance": float(np.nanmax(d)),
    }


def horizontal_base_grid(f: float, R: float, npts: int = 900) -> Dict[str, np.ndarray]:
    """
    Базовая сетка горизонтального типа.

    Ось фигуры: u in [-R, R].
    Радиальный профиль: d_h(u).
    """
    a = parabola_a(f, R)
    u = np.linspace(-R, R, npts)
    u = _ensure_zero_in_axis(u)
    d = horizontal_distance(u, f, R)
    return {
        "axis": u,
        "d": d,
        "L": float(R),
        "a": float(a),
        "max_distance": float(np.nanmax(d)),
    }


# ------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ------------------------------------------------------------
def morphology_class_from_base(base_max_distance: float, offsets: Sequence[float]) -> str:
    if not offsets:
        return "только базовый 2-й порядок"
    R_star = max(offsets)
    if base_max_distance > R_star:
        return "с внутренним пересечением / вложением"
    if math.isclose(base_max_distance, R_star, rel_tol=1e-12, abs_tol=1e-12):
        return "предельное касание"
    return "кольцевой / разнесённый режим до Merge"


def validate_offsets(base_distance: np.ndarray, offsets: Sequence[float]) -> List[str]:
    """
    Не блокирующая проверка. Строить всё равно можно.
    Сообщения носят информационный характер.
    """
    msgs: List[str] = []
    current_max = float(np.max(base_distance))
    for i, offset in enumerate(offsets, start=1):
        if offset < current_max:
            msgs.append(
                f"На шаге R{i}: R{i} = {offset:.6f} < {current_max:.6f}. "
                f"Это режим пересечения / вложения компонент; нужен Merge общего объёма."
            )
        elif math.isclose(offset, current_max, rel_tol=1e-12, abs_tol=1e-12):
            msgs.append(
                f"На шаге R{i}: R{i} = {offset:.6f} равно {current_max:.6f}. "
                f"Это предельное касание компонент."
            )
        current_max = offset + current_max
    return msgs


def stack_shifts(axis_half_length: float, h: float, m: int) -> np.ndarray:
    """
    Сдвиги центров m одинаковых экземпляров вдоль общей оси вращения.

    Полная длина одного экземпляра вдоль общей оси = 2 * axis_half_length.
    Шаг между центрами соседних экземпляров:
        step = 2 * axis_half_length + h

    h > 0  — зазор;
    h = 0  — касание;
    h < 0  — перекрытие / Merge.
    """
    if int(m) != m or m < 1:
        raise ValueError("Параметр m должен быть целым числом >= 1")
    m = int(m)
    step = 2.0 * float(axis_half_length) + float(h)
    return -np.arange(m, dtype=float) * step


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_or_show(fig, path: Path | None, show: bool):
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ------------------------------------------------------------
# РЕКУРСИВНЫЕ РАДИАЛЬНЫЕ ИНТЕРВАЛЫ
# ------------------------------------------------------------
def build_recursive_interval_arrays(base_distance: np.ndarray,
                                    offsets: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Радиальные интервалы для одного порядка.

    Порядок 2:
        [0, d]

    Следующий шаг для каждого интервала [lo, hi]:
        [R_k - hi, R_k - lo] и [R_k + lo, R_k + hi].

    Затем интервалы объединяются в общем Merge. Это не удаляет порождающие
    компоненты как объекты построения, а убирает только дублирование объёма.
    """
    zero = np.zeros_like(base_distance, dtype=float)
    intervals = [(zero.copy(), np.asarray(base_distance, dtype=float).copy())]
    for Rk in offsets:
        nxt: List[Tuple[np.ndarray, np.ndarray]] = []
        for lo, hi in intervals:
            a = np.maximum(Rk - hi, 0.0)
            b = np.maximum(Rk - lo, 0.0)
            c = np.maximum(Rk + lo, 0.0)
            d = np.maximum(Rk + hi, 0.0)
            nxt.append((a, b))
            nxt.append((c, d))
        intervals = nxt
    return intervals


@dataclass
class Piece:
    axis: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


# ------------------------------------------------------------
# MERGE ОБЩЕГО ОБЪЁМА
# ------------------------------------------------------------
def _interp_piece_to_global_axis(piece: Piece, global_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = piece.axis
    lo = piece.lo
    hi = piece.hi
    lo_i = np.interp(global_axis, x, lo)
    hi_i = np.interp(global_axis, x, hi)
    mask = (global_axis >= x.min() - 1e-12) & (global_axis <= x.max() + 1e-12)
    lo_i[~mask] = np.nan
    hi_i[~mask] = np.nan
    return lo_i, hi_i


def _merge_scalar_intervals(intervals: List[Tuple[float, float]], tol: float = MERGE_TOL) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    data = [
        (float(min(a, b)), float(max(a, b)))
        for a, b in intervals
        if np.isfinite(a) and np.isfinite(b) and b > a + tol
    ]
    if not data:
        return []
    data.sort(key=lambda t: (t[0], t[1]))
    merged = [list(data[0])]
    for a, b in data[1:]:
        if a <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _build_merged_components_from_pieces(pieces: List[Piece],
                                         n_global: int | None = None) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    if not pieces:
        raise ValueError("Нет компонент для построения общего объёма")

    all_axis = np.concatenate([p.axis for p in pieces])
    global_axis = np.unique(np.round(all_axis, 12))
    global_axis.sort()

    if n_global is not None and n_global > global_axis.size:
        dense = np.linspace(global_axis.min(), global_axis.max(), n_global)
        global_axis = np.unique(np.concatenate([global_axis, dense]))
        global_axis.sort()

    # Если между рядами / компонентами есть реальный пустой зазор, нельзя
    # соединять края зазора линиями или поверхностями. Добавляем контрольную
    # точку в середину каждого пустого промежутка; там интервалов нет, и график
    # разорвётся сам.
    cover = sorted((float(np.min(p.axis)), float(np.max(p.axis))) for p in pieces)
    merged_cover: List[List[float]] = []
    for a0, b0 in cover:
        if not merged_cover or a0 > merged_cover[-1][1] + MERGE_TOL:
            merged_cover.append([a0, b0])
        else:
            merged_cover[-1][1] = max(merged_cover[-1][1], b0)

    gap_points: List[float] = []
    for left, right in zip(merged_cover[:-1], merged_cover[1:]):
        if right[0] > left[1] + MERGE_TOL:
            gap_points.append(0.5 * (left[1] + right[0]))
    if gap_points:
        global_axis = np.unique(np.concatenate([global_axis, np.array(gap_points, dtype=float)]))
        global_axis.sort()

    interpolated = [_interp_piece_to_global_axis(piece, global_axis) for piece in pieces]

    merged_per_sample: List[List[Tuple[float, float]]] = []
    max_count = 0
    for j in range(global_axis.size):
        scalars: List[Tuple[float, float]] = []
        for lo_i, hi_i in interpolated:
            lo = lo_i[j]
            hi = hi_i[j]
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo + MERGE_TOL:
                scalars.append((max(0.0, lo), max(0.0, hi)))
        merged = _merge_scalar_intervals(scalars, tol=MERGE_TOL)
        merged_per_sample.append(merged)
        max_count = max(max_count, len(merged))

    lo_components = [np.full(global_axis.shape, np.nan, dtype=float) for _ in range(max_count)]
    hi_components = [np.full(global_axis.shape, np.nan, dtype=float) for _ in range(max_count)]

    for j, merged in enumerate(merged_per_sample):
        for k, (a0, b0) in enumerate(merged):
            lo_components[k][j] = a0
            hi_components[k][j] = b0

    return global_axis, lo_components, hi_components


def _contiguous_segments(mask: np.ndarray, axis: np.ndarray | None = None) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).copy()
    if mask.size == 0:
        return []

    # Защита от соединения краёв реального пустого промежутка.
    if axis is not None and axis.size == mask.size and axis.size >= 3:
        dif = np.diff(axis)
        pos = dif[np.isfinite(dif) & (dif > 0)]
        if pos.size:
            med = float(np.median(pos))
            jump_thr = max(1e-12, 4.0 * med)
            jump_breaks = np.where(dif > jump_thr)[0]
            for j in jump_breaks:
                if 0 <= j < mask.size - 1:
                    mask[j] = False
                    mask[j + 1] = False

    segments: List[Tuple[int, int]] = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            if i - start >= 2:
                segments.append((start, i))
            start = None
    if start is not None and mask.size - start >= 2:
        segments.append((start, mask.size))
    return segments


def _split_on_artificial_jumps(radius: np.ndarray) -> List[Tuple[int, int]]:
    """
    Разрыв только при настоящем скачке радиуса, возникающем из-за смены индекса
    объединённого интервала. Параболические вершины не считаются ошибкой.
    """
    radius = np.asarray(radius, dtype=float)
    n = radius.size
    if n < 2:
        return [(0, n)] if n > 0 else []
    dr = np.abs(np.diff(radius))
    valid = dr[np.isfinite(dr)]
    if valid.size == 0:
        return [(0, n)]

    finite_r = radius[np.isfinite(radius)]
    if finite_r.size == 0:
        return [(0, n)]
    r_range = float(np.nanmax(finite_r) - np.nanmin(finite_r))
    med = float(np.median(valid))

    # Жёсткий порог: не режем естественную параболическую вершину из-за большой
    # производной, но режем скачки от переиндексации Merge-компонент.
    jump_thr = max(1e-8, 12.0 * med, 0.08 * r_range)
    break_ids = np.where(dr > jump_thr)[0] + 1
    if break_ids.size == 0:
        return [(0, n)]

    segments: List[Tuple[int, int]] = []
    st = 0
    for br in break_ids:
        if br - st >= 2:
            segments.append((st, int(br)))
        st = int(br)
    if n - st >= 2:
        segments.append((st, n))
    return segments


def _plot_boundary_curve(ax, radius: np.ndarray, axis: np.ndarray, lw: float):
    for st, en in _split_on_artificial_jumps(radius):
        ax.plot(radius[st:en], axis[st:en], lw=lw)


# ------------------------------------------------------------
# ПОРОЖДАЮЩИЕ КОМПОНЕНТЫ ДЛЯ ОДНОГО ПОРЯДКА
# ------------------------------------------------------------
def vertical_order_union_components(f: float,
                                    R: float,
                                    offsets: Sequence[float],
                                    npts: int,
                                    m: int,
                                    h: float) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], float, np.ndarray]:
    base = vertical_base_grid(f, R, npts=npts)
    shifts = stack_shifts(float(base["L"]), h, m)
    interval_base = build_recursive_interval_arrays(base["d"], offsets)

    pieces: List[Piece] = []
    for shift in shifts:
        for lo, hi in interval_base:
            pieces.append(Piece(base["axis"] + shift, lo, hi))

    global_axis, lo_components, hi_components = _build_merged_components_from_pieces(
        pieces,
        n_global=max(5000, 6 * npts, 1200 * (len(offsets) + 1)),
    )
    return global_axis, lo_components, hi_components, float(base["L"]), shifts


def horizontal_order_union_components(f: float,
                                      R: float,
                                      offsets: Sequence[float],
                                      npts: int,
                                      m: int,
                                      h: float) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], float, np.ndarray]:
    base = horizontal_base_grid(f, R, npts=npts)
    shifts = stack_shifts(float(base["L"]), h, m)
    interval_base = build_recursive_interval_arrays(base["d"], offsets)

    pieces: List[Piece] = []
    for shift in shifts:
        for lo, hi in interval_base:
            pieces.append(Piece(base["axis"] + shift, lo, hi))

    global_axis, lo_components, hi_components = _build_merged_components_from_pieces(
        pieces,
        n_global=max(5000, 6 * npts, 1200 * (len(offsets) + 1)),
    )
    return global_axis, lo_components, hi_components, float(base["L"]), shifts


# ------------------------------------------------------------
# ПОВЕРХНОСТИ ВРАЩЕНИЯ
# ------------------------------------------------------------
def revolve_about_x(axis_coord: np.ndarray, radius: np.ndarray, nphi: int = NPHI):
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    S, Phi = np.meshgrid(axis_coord, phi, indexing="ij")
    RR = np.tile(radius[:, None], (1, phi.size))
    X = S
    Y = RR * np.cos(Phi)
    Z = RR * np.sin(Phi)
    return X, Y, Z


def revolve_about_y(axis_coord: np.ndarray, radius: np.ndarray, nphi: int = NPHI):
    phi = np.linspace(0.0, 2.0 * np.pi, nphi)
    U, Phi = np.meshgrid(axis_coord, phi, indexing="ij")
    RR = np.tile(radius[:, None], (1, phi.size))
    X = RR * np.cos(Phi)
    Y = U
    Z = RR * np.sin(Phi)
    return X, Y, Z


@dataclass
class PlotBounds3D:
    xmin: float = math.inf
    xmax: float = -math.inf
    ymin: float = math.inf
    ymax: float = -math.inf
    zmin: float = math.inf
    zmax: float = -math.inf

    def update(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
        self.xmin = min(self.xmin, float(np.nanmin(X)))
        self.xmax = max(self.xmax, float(np.nanmax(X)))
        self.ymin = min(self.ymin, float(np.nanmin(Y)))
        self.ymax = max(self.ymax, float(np.nanmax(Y)))
        self.zmin = min(self.zmin, float(np.nanmin(Z)))
        self.zmax = max(self.zmax, float(np.nanmax(Z)))

    def is_valid(self) -> bool:
        return all(math.isfinite(v) for v in (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax))


def set_axes_equal_real_3d(ax, bounds: PlotBounds3D):
    if not bounds.is_valid():
        return
    xmid = 0.5 * (bounds.xmin + bounds.xmax)
    ymid = 0.5 * (bounds.ymin + bounds.ymax)
    zmid = 0.5 * (bounds.zmin + bounds.zmax)
    half = 0.5 * max(bounds.xmax - bounds.xmin, bounds.ymax - bounds.ymin, bounds.zmax - bounds.zmin)
    if half <= 0:
        half = 1.0
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_zlim(zmid - half, zmid + half)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _plot_surface_piece_x(ax,
                          axis_seg: np.ndarray,
                          radius_seg: np.ndarray,
                          bounds: PlotBounds3D,
                          stride: int = 2):
    X, Y, Z = revolve_about_x(axis_seg, radius_seg)
    ax.plot_surface(
        X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
        linewidth=0, edgecolor="none", alpha=SURFACE_ALPHA, antialiased=False, shade=True,
    )
    bounds.update(X, Y, Z)


def _plot_surface_piece_y(ax,
                          axis_seg: np.ndarray,
                          radius_seg: np.ndarray,
                          bounds: PlotBounds3D,
                          split_positions: Sequence[float] = (),
                          stride: int = 2):
    # Разбиваем в положениях центров рядов, чтобы не сглаживать реальные вершины.
    split_ids = set()
    for pos in split_positions:
        idx = int(np.argmin(np.abs(axis_seg - pos)))
        if 1 <= idx < axis_seg.size - 1 and np.isclose(axis_seg[idx], pos, atol=1e-10):
            split_ids.add(idx)
    starts = [0] + sorted(split_ids)
    ends = sorted(split_ids) + [axis_seg.size - 1]

    for st, en in zip(starts, ends):
        seg_axis = axis_seg[st:en + 1]
        seg_rad = radius_seg[st:en + 1]
        if seg_axis.size < 2:
            continue
        X, Y, Z = revolve_about_y(seg_axis, seg_rad)
        ax.plot_surface(
            X[::stride, ::stride], Y[::stride, ::stride], Z[::stride, ::stride],
            linewidth=0, edgecolor="none", alpha=SURFACE_ALPHA, antialiased=False, shade=True,
        )
        bounds.update(X, Y, Z)


def _surface_subsegments(axis_seg: np.ndarray,
                         radius_seg: np.ndarray,
                         split_positions: Sequence[float] = ()) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Разбивает граничную кривую поверхности на реальные подучастки, чтобы
    не сглаживать и не склеивать внутренние параболические образующие при Merge.

    Учитываются:
    - искусственные скачки радиуса из-за переиндексации Merge-компонент;
    - реальные точки смены ряда / вершины по split_positions.
    """
    axis_seg = np.asarray(axis_seg, dtype=float)
    radius_seg = np.asarray(radius_seg, dtype=float)
    if axis_seg.size < 2:
        return []

    cut_ids = {0, axis_seg.size}

    for st, en in _split_on_artificial_jumps(radius_seg):
        cut_ids.add(st)
        cut_ids.add(en)

    for pos in split_positions:
        idx0 = int(np.argmin(np.abs(axis_seg - pos)))
        if 1 <= idx0 < axis_seg.size - 1:
            cut_ids.add(idx0)
            cut_ids.add(idx0 + 1)

    cuts = sorted(i for i in cut_ids if 0 <= i <= axis_seg.size)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        if b0 - a0 >= 2:
            aa = axis_seg[a0:b0]
            rr = radius_seg[a0:b0]
            if aa.size >= 2 and np.all(np.isfinite(rr)):
                out.append((aa, rr))
    return out


def _plot_surface_boundary_x(ax,
                             axis_seg: np.ndarray,
                             radius_seg: np.ndarray,
                             bounds: PlotBounds3D,
                             stride: int = 2):
    for aa, rr in _surface_subsegments(axis_seg, radius_seg, split_positions=()):
        _plot_surface_piece_x(ax, aa, rr, bounds, stride=stride)


def _plot_surface_boundary_y(ax,
                             axis_seg: np.ndarray,
                             radius_seg: np.ndarray,
                             bounds: PlotBounds3D,
                             split_positions: Sequence[float] = (),
                             stride: int = 2):
    for aa, rr in _surface_subsegments(axis_seg, radius_seg, split_positions=split_positions):
        _plot_surface_piece_y(ax, aa, rr, bounds, split_positions=split_positions, stride=stride)


# ------------------------------------------------------------
# БОКОВАЯ 2D-ВИЗУАЛИЗАЦИЯ НА 3D
# ------------------------------------------------------------
def _plot_side_section_on_vertical_3d(ax,
                                      global_axis: np.ndarray,
                                      lo_components: List[np.ndarray],
                                      hi_components: List[np.ndarray],
                                      bounds: PlotBounds3D):
    """
    Показывает яркое меридиональное 2D-сечение вертикального типа В ПЛОСКОСТИ ОСИ
    ВРАЩЕНИЯ 3D: Y = 0. Внешние границы выделяются одним ярким цветом,
    внутренние — другим, чтобы сразу видеть полный объединённый объём.
    """
    y_plane = 0.0
    outer_color = 'magenta'
    inner_color = 'lime'

    for lo, hi in zip(lo_components, hi_components):
        mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
        for st, en in _contiguous_segments(mask, global_axis):
            xx = global_axis[st:en]
            lo_seg = lo[st:en]
            hi_seg = hi[st:en]

            for a0, b0 in _split_on_artificial_jumps(hi_seg):
                x = xx[a0:b0]
                z = hi_seg[a0:b0]
                if x.size >= 2:
                    y = np.full_like(x, y_plane)
                    ax.plot(x, y, z, color=outer_color, linewidth=3.0, alpha=1.0)
                    ax.plot(x, y, -z, color=outer_color, linewidth=3.0, alpha=1.0)
            if np.nanmax(lo_seg) > MERGE_TOL:
                for a0, b0 in _split_on_artificial_jumps(lo_seg):
                    x = xx[a0:b0]
                    z = lo_seg[a0:b0]
                    if x.size >= 2:
                        y = np.full_like(x, y_plane)
                        ax.plot(x, y, z, color=inner_color, linewidth=2.6, alpha=1.0)
                        ax.plot(x, y, -z, color=inner_color, linewidth=2.6, alpha=1.0)


def _plot_side_section_on_horizontal_3d(ax,
                                        global_axis: np.ndarray,
                                        lo_components: List[np.ndarray],
                                        hi_components: List[np.ndarray],
                                        bounds: PlotBounds3D):
    """
    Показывает яркое меридиональное 2D-сечение горизонтального типа В ПЛОСКОСТИ ОСИ
    ВРАЩЕНИЯ 3D: X = 0. Внешние границы выделяются одним ярким цветом,
    внутренние — другим, чтобы сразу видеть полный объединённый объём.
    """
    x_plane = 0.0
    outer_color = 'magenta'
    inner_color = 'lime'

    for lo, hi in zip(lo_components, hi_components):
        mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
        for st, en in _contiguous_segments(mask, global_axis):
            yy = global_axis[st:en]
            lo_seg = lo[st:en]
            hi_seg = hi[st:en]

            for a0, b0 in _split_on_artificial_jumps(hi_seg):
                y = yy[a0:b0]
                z = hi_seg[a0:b0]
                if y.size >= 2:
                    x = np.full_like(y, x_plane)
                    ax.plot(x, y, z, color=outer_color, linewidth=3.0, alpha=1.0)
                    ax.plot(x, y, -z, color=outer_color, linewidth=3.0, alpha=1.0)
            if np.nanmax(lo_seg) > MERGE_TOL:
                for a0, b0 in _split_on_artificial_jumps(lo_seg):
                    y = yy[a0:b0]
                    z = lo_seg[a0:b0]
                    if y.size >= 2:
                        x = np.full_like(y, x_plane)
                        ax.plot(x, y, z, color=inner_color, linewidth=2.6, alpha=1.0)
                        ax.plot(x, y, -z, color=inner_color, linewidth=2.6, alpha=1.0)


# ------------------------------------------------------------
# 2D СЕЧЕНИЯ ОБЩЕГО ОБЪЁМА
# ------------------------------------------------------------
def _plot_component_2d(ax, axis: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """
    Рисует только реальные границы объединённого объёма.

    Никаких служебных линий, никаких искусственных замыканий, никаких
    fill_between. Если компонент разорван, линия разрывается.
    """
    mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
    for st, en in _contiguous_segments(mask, axis):
        yy = axis[st:en]
        lo_seg = lo[st:en]
        hi_seg = hi[st:en]

        _plot_boundary_curve(ax, hi_seg, yy, lw=1.8)
        _plot_boundary_curve(ax, -hi_seg, yy, lw=1.8)

        if np.nanmax(lo_seg) > MERGE_TOL:
            _plot_boundary_curve(ax, lo_seg, yy, lw=1.5)
            _plot_boundary_curve(ax, -lo_seg, yy, lw=1.5)


def _axis_limits_from_components(axis: np.ndarray, hi_components: List[np.ndarray]) -> Tuple[float, float, float]:
    rmax = 0.0
    for hi in hi_components:
        if np.any(np.isfinite(hi)):
            rmax = max(rmax, float(np.nanmax(hi)))
    if rmax <= 0:
        rmax = 1.0
    amin = float(np.nanmin(axis))
    amax = float(np.nanmax(axis))
    return rmax, amin, amax


def build_recursive_boundary_descriptors(offsets: Sequence[float]):
    """
    Строит точные дескрипторы границ радиальных интервалов без Merge.

    Дескриптор границы:
        ("const", t)            -> r(s) = t
        ("parab", c, sigma)     -> r(s) = c + sigma * d_base(s)

    где d_base(s) — базовая параболическая образующая текущего типа.

    ВАЖНО:
    - фокусы вычисляются ТОЛЬКО из этих фактических параболических кривых;
    - никакой отдельной "рекурсии фокусов" нет;
    - любая новая параболическая кривая получается тем же преобразованием,
      что и сама граница, и её фокус переносится синхронно с ней.
    """
    intervals = [(("const", 0.0), ("parab", 0.0, +1.0))]
    for Rk in offsets:
        nxt = []
        for lo, hi in intervals:
            nxt.append((_desc_sub_const(Rk, hi), _desc_sub_const(Rk, lo)))
            nxt.append((_desc_add_const(Rk, lo), _desc_add_const(Rk, hi)))
        intervals = nxt
    return intervals


def _desc_add_const(C: float, desc):
    if desc[0] == "const":
        return ("const", float(C + desc[1]))
    _, c, sigma = desc
    return ("parab", float(C + c), float(sigma))


def _desc_sub_const(C: float, desc):
    if desc[0] == "const":
        return ("const", float(C - desc[1]))
    _, c, sigma = desc
    return ("parab", float(C - c), float(-sigma))


def _split_curve_at_axis_positions(x: np.ndarray,
                                   y: np.ndarray,
                                   split_positions: Sequence[float],
                                   tol: float = 1e-10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Разбивает уже построенную 2D-кривую по осевым положениям, где базовая
    формула с |.| меняет ветвь. Это не дорисовывание, а только разбиение
    уже существующей линии на параболические участки.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return []

    cut_ids = {0, x.size}
    for pos in split_positions:
        idx = int(np.argmin(np.abs(y - pos)))
        if 1 <= idx < x.size - 1 and abs(float(y[idx] - pos)) <= max(tol, 1e-7 * max(1.0, np.nanmax(np.abs(y)))):
            cut_ids.add(idx)
            cut_ids.add(idx + 1)

    cuts = sorted(cut_ids)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for a0, b0 in zip(cuts[:-1], cuts[1:]):
        if b0 - a0 >= 8:
            out.append((x[a0:b0], y[a0:b0]))
    return out


def _quadratic_focus_vertical(x: np.ndarray, y: np.ndarray):
    """
    Фокус параболы, заданной фактически построенными точками y = A*x^2+B*x+C.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 8 or (np.nanmax(x) - np.nanmin(x)) < 1e-9:
        return None

    A, B, C = np.polyfit(x, y, 2)
    if abs(A) < 1e-12:
        return None

    y_fit = A*x*x + B*x + C
    scale = max(1.0, float(np.nanmax(y) - np.nanmin(y)))
    rmse = float(np.sqrt(np.mean((y - y_fit)**2))) / scale
    if rmse > 2.5e-5:
        return None

    xv = -B / (2.0*A)
    yv = C - B*B / (4.0*A)
    yf = yv + 1.0 / (4.0*A)
    return float(xv), float(yf)


def _quadratic_focus_horizontal(x: np.ndarray, y: np.ndarray):
    """
    Фокус параболы, заданной фактически построенными точками x = A*y^2+B*y+C.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 8 or (np.nanmax(y) - np.nanmin(y)) < 1e-9:
        return None

    A, B, C = np.polyfit(y, x, 2)
    if abs(A) < 1e-12:
        return None

    x_fit = A*y*y + B*y + C
    scale = max(1.0, float(np.nanmax(x) - np.nanmin(x)))
    rmse = float(np.sqrt(np.mean((x - x_fit)**2))) / scale
    if rmse > 2.5e-5:
        return None

    yv = -B / (2.0*A)
    xv = C - B*B / (4.0*A)
    xf = xv + 1.0 / (4.0*A)
    return float(xf), float(yv)


def _adaptive_focus_from_drawn_curve(x: np.ndarray,
                                     y: np.ndarray,
                                     orientation: str,
                                     min_points: int = 18,
                                     depth: int = 0,
                                     max_depth: int = 7) -> List[Tuple[float, float]]:
    """
    Вычисляет фокусы только по фактически построенной 2D-линии.

    Если линия не является одной параболой, она делится на участки.
    Это нужно для Merge-случаев, когда видимая граница может переходить
    с одной параболической кривой на другую.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < min_points:
        return []

    if orientation == "vertical":
        focus = _quadratic_focus_vertical(x, y)
    else:
        focus = _quadratic_focus_horizontal(x, y)

    if focus is not None:
        return [focus]

    if depth >= max_depth or x.size < 2*min_points:
        return []

    mid = x.size // 2
    return (
        _adaptive_focus_from_drawn_curve(x[:mid], y[:mid], orientation, min_points, depth+1, max_depth)
        + _adaptive_focus_from_drawn_curve(x[mid:], y[mid:], orientation, min_points, depth+1, max_depth)
    )


def _collect_drawn_boundary_curves(axis: np.ndarray,
                                   lo_components: List[np.ndarray],
                                   hi_components: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Возвращает ровно те 2D-линии, которые рисует _plot_component_2d:
    внешние границы ±hi и внутренние границы ±lo.
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for lo, hi in zip(lo_components, hi_components):
        mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
        for st, en in _contiguous_segments(mask, axis):
            yy = axis[st:en]
            lo_seg = lo[st:en]
            hi_seg = hi[st:en]

            for rad in (hi_seg, -hi_seg):
                for a0, b0 in _split_on_artificial_jumps(rad):
                    if b0 - a0 >= 8:
                        curves.append((np.asarray(rad[a0:b0], dtype=float),
                                       np.asarray(yy[a0:b0], dtype=float)))

            if np.nanmax(lo_seg) > MERGE_TOL:
                for rad in (lo_seg, -lo_seg):
                    for a0, b0 in _split_on_artificial_jumps(rad):
                        if b0 - a0 >= 8:
                            curves.append((np.asarray(rad[a0:b0], dtype=float),
                                           np.asarray(yy[a0:b0], dtype=float)))
    return curves


def _focus_points_from_drawn_2d(axis: np.ndarray,
                                lo_components: List[np.ndarray],
                                hi_components: List[np.ndarray],
                                shifts: np.ndarray,
                                orientation: str) -> List[Tuple[float, float]]:
    """
    Главная функция фокусов.

    1. Берёт только фактически построенные на 2D рисунке линии.
    2. Делит их на параболические участки.
    3. Вычисляет фокус каждого участка по его собственному уравнению.
    """
    curves = _collect_drawn_boundary_curves(axis, lo_components, hi_components)
    points: List[Tuple[float, float]] = []

    for x, y in curves:
        # Разбиваем в осевых центрах рядов, где формула с abs меняет ветвь.
        pieces = _split_curve_at_axis_positions(x, y, split_positions=shifts)
        if not pieces:
            pieces = [(x, y)]

        for px, py in pieces:
            points.extend(_adaptive_focus_from_drawn_curve(px, py, orientation=orientation))

    # Удаляем дубли.
    unique = sorted({(round(float(x), 10), round(float(y), 10)) for x, y in points},
                    key=lambda p: (p[1], p[0]))
    return [(float(x), float(y)) for x, y in unique]


def _plot_focus_points_2d(ax,
                          points: List[Tuple[float, float]],
                          rmax: float):
    if not points:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.scatter(xs, ys, s=24, marker='o', color='darkgreen', zorder=7,
               label='Фокусы фактически построенных 2D-парабол')

    dx = 0.018 * max(rmax, 1.0)
    for x, y in points:
        if abs(x) <= 1e-12:
            ax.text(x + dx, y, 'F', fontsize=8, color='darkgreen',
                    va='center', ha='left')
        else:
            ha = 'left' if x > 0 else 'right'
            x_text = x + dx if x > 0 else x - dx
            ax.text(x_text, y, 'F', fontsize=8, color='darkgreen',
                    va='center', ha=ha)



# ------------------------------------------------------------
# АННОТАЦИИ ДЛЯ 2D РИСУНКОВ 2-ГО ПОРЯДКА
# ------------------------------------------------------------
def _draw_common_rotation_axis_label(ax, axis_min: float, axis_max: float):
    """Центральная пунктирная ось вращения r=0 для 2D."""
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.1, zorder=2)
    ax.text(0.03, 0.97, 'Ось вращения',
            transform=ax.transAxes, ha='left', va='top', fontsize=9)


def _annotate_vertical_order2_2d(ax,
                                 f: float,
                                 R: float,
                                 a: float,
                                 focus_points: List[Tuple[float, float]],
                                 rmax: float,
                                 axis_min: float,
                                 axis_max: float):
    """
    Вертикальный псевдопараболоид 2-го порядка:
    - центральная пунктирная ось вращения;
    - пунктирная фокальная ось образующей;
    - размер R между этими двумя пунктирными линиями.
    """
    _draw_common_rotation_axis_label(ax, axis_min, axis_max)

    x_focus_axis = float(R)

    ax.axvline(x_focus_axis, color='black', linestyle='--', linewidth=1.05, zorder=2)
    ax.text(x_focus_axis + 0.035 * max(rmax, 1.0),
            axis_min + 0.72 * (axis_max - axis_min),
            'Фокальная ось\nобразующей',
            ha='left', va='center', fontsize=9)

    yR = axis_min + 0.18 * (axis_max - axis_min)
    ax.annotate('', xy=(x_focus_axis, yR), xytext=(0.0, yR),
                arrowprops=dict(arrowstyle='<->', lw=1.15, color='black'))
    ax.text(0.5 * x_focus_axis, yR + 0.035 * (axis_max - axis_min),
            'R', ha='center', va='bottom', fontsize=12)
    ax.text(0.5 * x_focus_axis, yR - 0.025 * (axis_max - axis_min),
            'расстояние от оси вращения до фокальной оси',
            ha='center', va='top', fontsize=8)

    near = [(x, y) for x, y in focus_points if abs(x - x_focus_axis) <= 0.08 * max(R, 1.0)]
    if near:
        near.sort(key=lambda p: p[1])
        y_low = near[0][1]
        y_high = near[-1][1]
        ax.text(x_focus_axis - 0.04 * max(rmax, 1.0), y_high,
                '+f', ha='right', va='center', fontsize=9)
        ax.text(x_focus_axis - 0.04 * max(rmax, 1.0), y_low,
                '−f', ha='right', va='center', fontsize=9)


def _annotate_horizontal_order2_2d(ax,
                                   f: float,
                                   R: float,
                                   a: float,
                                   focus_points: List[Tuple[float, float]],
                                   rmax: float,
                                   axis_min: float,
                                   axis_max: float):
    """
    Горизонтальный псевдопараболоид 2-го порядка:
    - центральная горизонтальная пунктирная ось симметрии y=0;
    - горизонтальная фокальная ось образующей y=R;
    - размер R между y=0 и y=R.
    """
    # Центральная вертикальная линия остаётся только как геометрическая ось,
    # без подписи "вертикальная ось симметрии".
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.1, zorder=2)

    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.05, zorder=2)
    ax.text(0.98, 0.52, 'центральная горизонтальная\nось симметрии y = 0',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)

    ax.axhline(R, color='black', linestyle='--', linewidth=1.05, zorder=2)
    ax.text(0.98, 0.88, 'Фокальная ось\nобразующей y = R',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)

    xR = -0.82 * max(rmax, a, 1.0)
    ax.annotate('', xy=(xR, R), xytext=(xR, 0.0),
                arrowprops=dict(arrowstyle='<->', lw=1.15, color='black'))
    ax.text(xR - 0.03 * max(rmax, 1.0), 0.5 * R,
            'R', ha='right', va='center', fontsize=12)
    ax.text(xR + 0.02 * max(rmax, 1.0), 0.5 * R,
            'расстояние от оси симметрии\nдо оси фокусов',
            ha='left', va='center', fontsize=8)

    ax.annotate('', xy=(a, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle='<->', lw=1.0, color='black'))
    ax.text(0.5 * a, 0.05 * max(R, 1.0),
            'a = R²/(4f)', ha='center', va='bottom', fontsize=9)


def _hide_3d_service_axes(ax):
    """
    Возвращает на 3D координатную сетку, оси и tick marks, чтобы можно было
    визуально контролировать полный объём и внутренние образующие.
    """
    ax.set_axis_on()
    ax.grid(True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_union_section_vertical(order: int,
                                global_axis: np.ndarray,
                                lo_components: List[np.ndarray],
                                hi_components: List[np.ndarray],
                                shifts: np.ndarray,
                                used_offsets: Sequence[float],
                                f: float,
                                R: float,
                                m: int,
                                h: float,
                                outpath: Path | None = None,
                                show: bool = True):
    fig, ax = plt.subplots(figsize=(9, 10))
    for lo, hi in zip(lo_components, hi_components):
        _plot_component_2d(ax, global_axis, lo, hi)

    rmax, _, _ = _axis_limits_from_components(global_axis, hi_components)
    focus_points = _focus_points_from_drawn_2d(
        axis=global_axis,
        lo_components=lo_components,
        hi_components=hi_components,
        shifts=shifts,
        orientation="vertical",
    )
    _plot_focus_points_2d(ax, focus_points, rmax=rmax)

    if focus_points:
        rmax = max(rmax, max(abs(x) for x, _ in focus_points))

    _, axis_min, axis_max = _axis_limits_from_components(global_axis, hi_components)
    if order == 2:
        _annotate_vertical_order2_2d(
            ax=ax, f=f, R=R, a=parabola_a(f, R), focus_points=focus_points,
            rmax=rmax, axis_min=axis_min, axis_max=axis_max,
        )

    ax.set_xlim(-1.05 * rmax, 1.18 * rmax if order == 2 else 1.05 * rmax)
    ax.set_title(f"Псевдопараболоид: вертикальный тип, порядок {order}, 2D, f={f}, R={R}, m={m}, h={h}")
    ax.set_xlabel("Радиальная координата")
    ax.set_ylabel("Координата вдоль общей оси")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, outpath, show)


def plot_union_section_horizontal(order: int,
                                  global_axis: np.ndarray,
                                  lo_components: List[np.ndarray],
                                  hi_components: List[np.ndarray],
                                  shifts: np.ndarray,
                                  used_offsets: Sequence[float],
                                  f: float,
                                  R: float,
                                  m: int,
                                  h: float,
                                  outpath: Path | None = None,
                                  show: bool = True):
    fig, ax = plt.subplots(figsize=(10, 8))
    for lo, hi in zip(lo_components, hi_components):
        _plot_component_2d(ax, global_axis, lo, hi)

    rmax, _, _ = _axis_limits_from_components(global_axis, hi_components)
    focus_points = _focus_points_from_drawn_2d(
        axis=global_axis,
        lo_components=lo_components,
        hi_components=hi_components,
        shifts=shifts,
        orientation="horizontal",
    )
    _plot_focus_points_2d(ax, focus_points, rmax=rmax)

    if focus_points:
        rmax = max(rmax, max(abs(x) for x, _ in focus_points))

    _, axis_min, axis_max = _axis_limits_from_components(global_axis, hi_components)
    if order == 2:
        _annotate_horizontal_order2_2d(
            ax=ax, f=f, R=R, a=parabola_a(f, R), focus_points=focus_points,
            rmax=rmax, axis_min=axis_min, axis_max=axis_max,
        )

    ax.set_xlim(-1.10 * max(rmax, parabola_a(f, R)), 1.10 * max(rmax, parabola_a(f, R)))
    ax.set_title(f"Псевдопараболоид: горизонтальный тип, порядок {order}, 2D, f={f}, R={R}, m={m}, h={h}")
    ax.set_xlabel("Радиальная координата")
    ax.set_ylabel("Координата вдоль общей оси")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    _save_or_show(fig, outpath, show)


# ------------------------------------------------------------
# 3D ПОВЕРХНОСТИ ОБЩЕГО ОБЪЁМА
# ------------------------------------------------------------
def plot_union_surface_vertical(order: int,
                                global_axis: np.ndarray,
                                lo_components: List[np.ndarray],
                                hi_components: List[np.ndarray],
                                f: float,
                                R: float,
                                m: int,
                                h: float,
                                outpath: Path | None = None,
                                show: bool = True,
                                stride: int = 2):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    bounds = PlotBounds3D()

    for lo, hi in zip(lo_components, hi_components):
        mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
        for st, en in _contiguous_segments(mask, global_axis):
            axis_seg = global_axis[st:en]
            lo_seg = lo[st:en]
            hi_seg = hi[st:en]
            _plot_surface_boundary_x(ax, axis_seg, hi_seg, bounds, stride=stride)
            if np.nanmax(lo_seg) > MERGE_TOL:
                _plot_surface_boundary_x(ax, axis_seg, lo_seg, bounds, stride=stride)

    _plot_side_section_on_vertical_3d(ax, global_axis, lo_components, hi_components, bounds)
    set_axes_equal_real_3d(ax, bounds)
    _hide_3d_service_axes(ax)
    ax.set_title(f"Псевдопараболоид: вертикальный тип, порядок {order}, 3D, f={f}, R={R}, m={m}, h={h}")
    fig.tight_layout()
    _save_or_show(fig, outpath, show)


def plot_union_surface_horizontal(order: int,
                                  global_axis: np.ndarray,
                                  lo_components: List[np.ndarray],
                                  hi_components: List[np.ndarray],
                                  shifts: np.ndarray,
                                  f: float,
                                  R: float,
                                  m: int,
                                  h: float,
                                  outpath: Path | None = None,
                                  show: bool = True,
                                  stride: int = 2):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    bounds = PlotBounds3D()

    for lo, hi in zip(lo_components, hi_components):
        mask = np.isfinite(lo) & np.isfinite(hi) & (hi > lo + MERGE_TOL)
        for st, en in _contiguous_segments(mask, global_axis):
            axis_seg = global_axis[st:en]
            lo_seg = lo[st:en]
            hi_seg = hi[st:en]
            _plot_surface_boundary_y(ax, axis_seg, hi_seg, bounds, split_positions=shifts, stride=stride)
            if np.nanmax(lo_seg) > MERGE_TOL:
                _plot_surface_boundary_y(ax, axis_seg, lo_seg, bounds, split_positions=shifts, stride=stride)

    _plot_side_section_on_horizontal_3d(ax, global_axis, lo_components, hi_components, bounds)
    set_axes_equal_real_3d(ax, bounds)
    _hide_3d_service_axes(ax)
    ax.set_title(f"Псевдопараболоид: горизонтальный тип, порядок {order}, 3D, f={f}, R={R}, m={m}, h={h}")
    fig.tight_layout()
    _save_or_show(fig, outpath, show)


# ------------------------------------------------------------
# ДОПОЛНИТЕЛЬНАЯ ДИАГНОСТИКА ОБРАЗУЮЩИХ
# ------------------------------------------------------------
def plot_base_generatrices(f: float,
                           R: float,
                           outpath: Path | None = None,
                           show: bool = True,
                           npts: int = 1200):
    """
    Исправленные базовые образующие 2-го порядка.

    Показываются только половинки образующих — по одной стороне от оси вращения,
    то есть именно зеркальные два сегмента парабол без дублирования второй
    половины полного 2D-сечения.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # ---------------- Вертикальный тип ----------------
    ax = axes[0]
    a = parabola_a(f, R)
    s = np.linspace(-a, a, max(1200, npts))
    d = vertical_distance(np.abs(s), f, R)

    # Только правая половина образующей: два зеркальных сегмента парабол.
    ax.plot(d, s, color='navy', linewidth=2.0, label='Образующая')

    # Фокусы образующей.
    focus_points_v = [(R, -f), (R, +f)]
    _plot_focus_points_2d(ax, focus_points_v, rmax=max(R, float(np.nanmax(d))))

    # Оптимизированные служебные линии.
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.05, zorder=1)
    ax.text(0.03, 0.97, 'Ось вращения', transform=ax.transAxes,
            ha='left', va='top', fontsize=9)

    ax.axvline(R, color='black', linestyle='--', linewidth=1.0, zorder=1)
    ax.text(R + 0.25, 0.72 * a, 'Фокальная ось\nобразующей',
            ha='left', va='center', fontsize=9)

    yR = -a - 0.9
    ax.annotate('', xy=(R, yR), xytext=(0.0, yR),
                arrowprops=dict(arrowstyle='<->', lw=1.15, color='black'))
    ax.text(0.5 * R, yR + 0.28, 'R', ha='center', va='bottom', fontsize=12)

    ax.text(R - 0.35, +f, '+f', ha='right', va='center', fontsize=9)
    ax.text(R - 0.35, -f, '−f', ha='right', va='center', fontsize=9)

    ax.set_xlim(-0.8, 1.28 * R)
    ax.set_ylim(-a - 1.3, a + 1.0)
    ax.set_title(f'Базовые образующие 2-го порядка: вертикальный тип')
    ax.set_xlabel('Радиальная координата')
    ax.set_ylabel('Координата вдоль общей оси')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=9)

    # ---------------- Горизонтальный тип ----------------
    ax = axes[1]
    u = np.linspace(-R, R, max(1200, npts))
    d = horizontal_distance(u, f, R)
    a_h = float(np.nanmax(d))

    # Только правая половина образующей.
    ax.plot(d, u, color='darkred', linewidth=2.0, label='Образующая')

    # Фокусы образующей.
    focus_points_h = [(f, -R), (f, +R)]
    _plot_focus_points_2d(ax, focus_points_h, rmax=max(a_h, f))

    # Служебные линии по отработанной схеме 2D.
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.05, zorder=1)
    ax.axhline(0.0, color='black', linestyle='--', linewidth=1.0, zorder=1)
    ax.text(0.98, 0.52, 'центральная горизонтальная\nось симметрии y = 0',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8)

    ax.axhline(R, color='black', linestyle='--', linewidth=1.0, zorder=1)
    ax.text(0.98, 0.88, 'Фокальная ось\nобразующей y = R',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)

    xR = -0.72
    ax.annotate('', xy=(xR, R), xytext=(xR, 0.0),
                arrowprops=dict(arrowstyle='<->', lw=1.15, color='black'))
    ax.text(xR - 0.18, 0.5 * R, 'R', ha='right', va='center', fontsize=12)

    # Размер a показываем компактно вдоль y=0.
    ax.annotate('', xy=(a_h, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle='<->', lw=1.0, color='black'))
    ax.text(0.5 * a_h, 0.36, 'a = R²/(4f)', ha='center', va='bottom', fontsize=9)

    ax.set_xlim(-1.05, 1.18 * a_h)
    ax.set_ylim(-1.18 * R, 1.18 * R)
    ax.set_title(f'Базовые образующие 2-го порядка: горизонтальный тип')
    ax.set_xlabel('Радиальная координата')
    ax.set_ylabel('Координата вдоль общей оси')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right', fontsize=9)

    fig.suptitle(f'Псевдопараболоиды 2-го порядка: базовые образующие, f={f}, R={R}')
    fig.tight_layout()
    _save_or_show(fig, outpath, show)


# ------------------------------------------------------------
# ОСНОВНАЯ ФУНКЦИЯ
# ------------------------------------------------------------
def run_pseudoparaboloids(f: float = 2.0,
                          R: float = 8.0,
                          offsets: Sequence[float] = (7.0, 14.0),
                          geometry_type: str = "both",   # vertical / horizontal / both
                          mode: str = "all",             # section / surface / all / profile
                          outdir: str = "pseudoparaboloids_notebook_output",
                          npts: int = 900,
                          h: float = -2.0,
                          m: int = 3,
                          show: bool = True,
                          make_profile: bool = True) -> Dict[str, object]:
    """
    Главная функция.

    Строит итоговый общий объём для каждого порядка от 2 до n.
    n = len(offsets) + 2.

    Параметры:
        f             фокусное расстояние параболических ветвей;
        R             вертикальная высота / радиус смещения оси вращения;
        offsets       R1, R2, ..., R_{n-2}; задают рекурсивные уровни;
        geometry_type "vertical", "horizontal" или "both";
        mode          "section", "surface", "all" или "profile";
        m             число рядов;
        h             зазор / касание / перекрытие между рядами;
        show          показывать графики на экране;
        make_profile  сохранить отдельную схему образующих.
    """
    offsets = list(offsets)
    outdir_path = Path(outdir)
    _ensure_dir(outdir_path)

    if geometry_type not in {"vertical", "horizontal", "both"}:
        raise ValueError("geometry_type должен быть 'vertical', 'horizontal' или 'both'")
    if mode not in {"section", "surface", "all", "profile"}:
        raise ValueError("mode должен быть 'section', 'surface', 'all' или 'profile'")
    if f <= 0 or R <= 0:
        raise ValueError("Параметры f и R должны быть положительными")
    if int(m) != m or m < 1:
        raise ValueError("Параметр m должен быть целым числом >= 1")

    a = parabola_a(f, R)
    base_v = vertical_base_grid(f, R, npts=npts)
    base_h = horizontal_base_grid(f, R, npts=npts)

    result: Dict[str, object] = {
        "f": float(f),
        "R": float(R),
        "a": float(a),
        "offsets": offsets,
        "n_order": len(offsets) + 2,
        "m": int(m),
        "h": float(h),
        "vertical_base_max_distance": float(base_v["max_distance"]),
        "horizontal_base_max_distance": float(base_h["max_distance"]),
        "vertical_morphology": morphology_class_from_base(float(base_v["max_distance"]), offsets),
        "horizontal_morphology": morphology_class_from_base(float(base_h["max_distance"]), offsets),
        "volume_mode": "union_of_all_parabolic_toroidal_components",
        "saved_files": [],
        "warnings": [],
    }

    print("ПСЕВДОПАРАБОЛОИДЫ n-го ПОРЯДКА")
    print(f"f = {f}")
    print(f"R = {R}")
    print(f"a = R^2/(4f) = {a:.6f}")
    print(f"offsets = {offsets}")
    print(f"Порядок n = {len(offsets) + 2}")
    print(f"m = {m}")
    print(f"h = {h}")
    print(f"Вертикальный базовый максимум d = {base_v['max_distance']:.6f}")
    print(f"Горизонтальный базовый максимум d = {base_h['max_distance']:.6f}")
    print("Режим построения = общий объём как объединение всех компонент")
    print("Правило: удаляется только общая перекрывающаяся часть объёма;")
    print("сами порождающие параболические тороиды как объекты построения не удаляются.")
    print("Искусственные прямые соединения не дорисовываются.")
    print(f"Папка вывода = {outdir_path.resolve()}")

    if make_profile or mode == "profile":
        path = outdir_path / f"base_generatrices_f{f}_R{R}.png"
        plot_base_generatrices(f=f, R=R, outpath=path, show=show, npts=max(1200, npts))
        result["saved_files"].append(str(path))
        if mode == "profile":
            return result

    if geometry_type in {"vertical", "both"}:
        result["warnings"].extend([f"vertical: {w}" for w in validate_offsets(base_v["d"], offsets)])

        for p in range(2, len(offsets) + 3):
            used_offsets = offsets[:max(0, p - 2)]
            global_axis, lo_components, hi_components, _, shifts = vertical_order_union_components(
                f=f, R=R, offsets=used_offsets, npts=npts, m=m, h=h,
            )
            if mode in {"section", "all"}:
                path = outdir_path / f"vertical_order_{p:02d}_section_m{m}_h{h}.png"
                plot_union_section_vertical(
                    order=p,
                    global_axis=global_axis,
                    lo_components=lo_components,
                    hi_components=hi_components,
                    shifts=shifts,
                    used_offsets=used_offsets,
                    f=f,
                    R=R,
                    m=m,
                    h=h,
                    outpath=path,
                    show=show,
                )
                result["saved_files"].append(str(path))
            if mode in {"surface", "all"}:
                path = outdir_path / f"vertical_order_{p:02d}_surface_m{m}_h{h}.png"
                plot_union_surface_vertical(
                    order=p,
                    global_axis=global_axis,
                    lo_components=lo_components,
                    hi_components=hi_components,
                    f=f,
                    R=R,
                    m=m,
                    h=h,
                    outpath=path,
                    show=show,
                )
                result["saved_files"].append(str(path))

    if geometry_type in {"horizontal", "both"}:
        result["warnings"].extend([f"horizontal: {w}" for w in validate_offsets(base_h["d"], offsets)])

        for p in range(2, len(offsets) + 3):
            used_offsets = offsets[:max(0, p - 2)]
            global_axis, lo_components, hi_components, _, shifts = horizontal_order_union_components(
                f=f, R=R, offsets=used_offsets, npts=npts, m=m, h=h,
            )
            if mode in {"section", "all"}:
                path = outdir_path / f"horizontal_order_{p:02d}_section_m{m}_h{h}.png"
                plot_union_section_horizontal(
                    order=p,
                    global_axis=global_axis,
                    lo_components=lo_components,
                    hi_components=hi_components,
                    shifts=shifts,
                    used_offsets=used_offsets,
                    f=f,
                    R=R,
                    m=m,
                    h=h,
                    outpath=path,
                    show=show,
                )
                result["saved_files"].append(str(path))
            if mode in {"surface", "all"}:
                path = outdir_path / f"horizontal_order_{p:02d}_surface_m{m}_h{h}.png"
                plot_union_surface_horizontal(
                    order=p,
                    global_axis=global_axis,
                    lo_components=lo_components,
                    hi_components=hi_components,
                    shifts=shifts,
                    f=f,
                    R=R,
                    m=m,
                    h=h,
                    outpath=path,
                    show=show,
                )
                result["saved_files"].append(str(path))

    if result["warnings"]:
        print("\nПРЕДУПРЕЖДЕНИЯ:")
        for w in result["warnings"]:
            print(" -", w)

    print("\nСОХРАНЁННЫЕ ФАЙЛЫ:")
    for filename in result["saved_files"]:
        print(" -", filename)

    return result


# ------------------------------------------------------------
# ПАРАМЕТРЫ ПО УМОЛЧАНИЮ
# ------------------------------------------------------------
if __name__ == "__main__":
    # Пример из исходного документа псевдопараболоидов:
    # f = 2, R = 8, a = R^2/(4f) = 8.
    f = 2.0
    R = 8.0

    # offsets задают максимальный порядок: n = len(offsets) + 2.
    # Для n=4 нужны два смещения: R1, R2.
    offsets = [7.0, 14.0]

    geometry_type = "both"     # "vertical", "horizontal", "both"
    mode = "all"               # "section", "surface", "all", "profile"
    outdir = "pseudoparaboloids_notebook_output"
    npts = 900
    m = 3
    h = -2.0
    show = True

    results = run_pseudoparaboloids(
        f=f,
        R=R,
        offsets=offsets,
        geometry_type=geometry_type,
        mode=mode,
        outdir=outdir,
        npts=npts,
        m=m,
        h=h,
        show=show,
        make_profile=True,
    )
