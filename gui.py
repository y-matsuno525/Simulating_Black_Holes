"""gui.py — 黒ホール格子シミュレーションの統合 GUI（Tkinter, 標準ライブラリのみ）。

3 タブ + 共通ログで、計算・論文図生成・論文ビルドを一気通貫で操作する:
  [計算]      config.json の主要パラメータを編集し run_sim.py を実行（outputs/ に出力）
  [論文図]    paper_figures/make_*.py を実行し、paper_figures/generated/ の図をプレビュー
  [論文ビルド] paper/main.tex を pdflatex/bibtex でビルドし main.pdf を開く

各処理は別スレッドの subprocess で実行し、stdout/stderr を下部ログにストリーム表示する。
依存追加なし（画像プレビューは Pillow があれば使用、無ければ Tk PhotoImage）。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable
GENERATED = REPO / "paper_figures" / "generated"

# [計算] タブで編集する config.json のキー（ラベル, キー, 既定取得元）
CONFIG_FIELDS = [
    ("L (格子点数)", "L"),
    ("p", "p"),
    ("m", "m"),
    ("scenario", "scenario"),
    ("t_f (終了時刻)", "t_f"),
    ("dt_scale", "dt_scale"),
    ("beta_amplitude", "beta_amplitude"),
    ("beta_center_fraction", "beta_center_fraction"),
    ("j0_fraction", "j0_fraction"),
    ("sigma_fraction", "sigma_fraction"),
    ("run_name", "run_name"),
]
SCENARIOS = ["BH_chi_minus", "BH_chi_plus", "WH_chi_plus", "null"]

FIGURE_SCRIPTS = [
    ("全図生成 (make_all)", "paper_figures/make_all.py"),
    ("分散 dispersion", "paper_figures/make_dispersion.py"),
    ("ダブラーFFT", "paper_figures/make_doubler_fft.py"),
    ("BH 時空図", "paper_figures/make_bh_panels.py"),
    ("WH 時空図", "paper_figures/make_wh_panels.py"),
    ("表面重力 sg", "paper_figures/make_surface_gravity.py"),
    ("[解析] 計量幾何", "analysis/majorana_metric.py"),
    ("[解析] logL", "analysis/stagnation_logL.py"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Black Hole Lattice Simulator — 統合GUI")
        self.geometry("1000x760")
        self._log_q: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._action_buttons: list[tk.Widget] = []
        self._preview_img = None  # 参照保持

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=False, padx=6, pady=6)
        self._build_calc_tab(nb)
        self._build_figure_tab(nb)
        self._build_build_tab(nb)

        # 共通ログ
        logframe = ttk.LabelFrame(self, text="ログ")
        logframe.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.log = ScrolledText(logframe, height=14, state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

        self.after(80, self._drain_log)

    # ---------- タブ構築 ----------
    def _build_calc_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="計算")
        self.cfg_vars: dict[str, tk.StringVar] = {}
        try:
            base = json.loads((REPO / "config.json").read_text(encoding="utf-8"))
        except Exception:
            base = {}
        for i, (label, key) in enumerate(CONFIG_FIELDS):
            ttk.Label(tab, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            var = tk.StringVar(value=str(base.get(key, "")))
            self.cfg_vars[key] = var
            if key == "scenario":
                ttk.Combobox(tab, textvariable=var, values=SCENARIOS, width=28).grid(
                    row=i, column=1, sticky="w", padx=8)
            else:
                ttk.Entry(tab, textvariable=var, width=30).grid(
                    row=i, column=1, sticky="w", padx=8)
        ttk.Label(tab, text="※ config.json は書き換えず、上書き値で実行します",
                  foreground="gray").grid(row=len(CONFIG_FIELDS), column=0, columnspan=2,
                                          sticky="w", padx=8, pady=6)
        btn = ttk.Button(tab, text="シミュレーション実行 (run_sim.py)", command=self._run_simulation)
        btn.grid(row=len(CONFIG_FIELDS) + 1, column=0, columnspan=2, pady=8)
        self._action_buttons.append(btn)

    def _build_figure_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="論文図")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=6, pady=6)
        for label, script in FIGURE_SCRIPTS:
            b = ttk.Button(left, text=label, width=24,
                           command=lambda s=script: self._run_script(s))
            b.pack(fill="x", pady=2)
            self._action_buttons.append(b)
        ttk.Separator(left).pack(fill="x", pady=6)
        rb = ttk.Button(left, text="生成図を再読込", command=self._refresh_figures)
        rb.pack(fill="x", pady=2)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.fig_list = tk.Listbox(right, height=8)
        self.fig_list.pack(fill="x")
        self.fig_list.bind("<<ListboxSelect>>", self._on_fig_select)
        self.preview = ttk.Label(right, text="(図を選択するとプレビュー)", anchor="center")
        self.preview.pack(fill="both", expand=True, pady=6)
        self._refresh_figures()

    def _build_build_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="論文ビルド")
        ttk.Label(tab, text="paper/main.tex を pdflatex→bibtex→pdflatex×2 でビルドします。",
                  ).pack(anchor="w", padx=10, pady=8)
        b1 = ttk.Button(tab, text="論文をビルド (main.pdf 生成)", command=self._build_paper)
        b1.pack(anchor="w", padx=10, pady=4)
        b2 = ttk.Button(tab, text="main.pdf を開く", command=self._open_pdf)
        b2.pack(anchor="w", padx=10, pady=4)
        self._action_buttons.append(b1)

    # ---------- ログ ----------
    def _logln(self, text):
        self._log_q.put(text)

    def _drain_log(self):
        try:
            while True:
                line = self._log_q.get_nowait()
                self.log.configure(state="normal")
                self.log.insert("end", line)
                self.log.see("end")
                self.log.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(80, self._drain_log)

    # ---------- subprocess 実行（別スレッド） ----------
    def _set_busy(self, busy):
        self._busy = busy
        state = "disabled" if busy else "normal"
        for b in self._action_buttons:
            try:
                b.configure(state=state)
            except tk.TclError:
                pass

    def _run_cmd(self, args, label):
        if self._busy:
            self._logln("[GUI] 実行中です。完了までお待ちください。\n")
            return
        self._set_busy(True)
        self._logln(f"\n===== {label} =====\n$ {' '.join(str(a) for a in args)}\n")

        def worker():
            try:
                proc = subprocess.Popen(
                    args, cwd=str(REPO), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1,
                )
                for line in proc.stdout:
                    self._log_q.put(line)
                proc.wait()
                self._log_q.put(f"[GUI] 終了コード {proc.returncode}\n")
            except Exception as e:
                self._log_q.put(f"[GUI] エラー: {e}\n")
            finally:
                self.after(0, lambda: self._set_busy(False))
                self.after(0, self._refresh_figures)

        threading.Thread(target=worker, daemon=True).start()

    # ---------- アクション ----------
    def _run_simulation(self):
        overrides = {}
        for key, var in self.cfg_vars.items():
            raw = var.get().strip()
            if raw == "":
                continue
            overrides[key] = self._coerce(key, raw)
        ov_path = REPO / "_gui_overrides.json"
        ov_path.write_text(json.dumps(overrides, ensure_ascii=False), encoding="utf-8")
        self._run_cmd([PYTHON, "run_sim.py", str(ov_path.name)], "シミュレーション実行")

    @staticmethod
    def _coerce(key, raw):
        if key == "scenario":
            return None if raw == "null" else raw
        if key == "run_name":
            return None if raw in ("", "None", "null") else raw
        try:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    def _run_script(self, script):
        self._run_cmd([PYTHON, script], f"実行: {script}")

    def _build_paper(self):
        self._run_cmd([PYTHON, "-c", _BUILD_SNIPPET], "論文ビルド")

    def _open_pdf(self):
        pdf = REPO / "paper" / "main.pdf"
        if pdf.exists():
            try:
                os.startfile(str(pdf))  # type: ignore[attr-defined]
            except Exception as e:
                self._logln(f"[GUI] PDF を開けません: {e}\n")
        else:
            self._logln("[GUI] main.pdf がありません。先にビルドしてください。\n")

    # ---------- 図プレビュー ----------
    def _refresh_figures(self):
        if not hasattr(self, "fig_list"):
            return
        self.fig_list.delete(0, "end")
        if GENERATED.exists():
            for p in sorted(GENERATED.glob("*.png")):
                self.fig_list.insert("end", p.name)

    def _on_fig_select(self, _evt):
        sel = self.fig_list.curselection()
        if not sel:
            return
        name = self.fig_list.get(sel[0])
        self._show_image(GENERATED / name)

    def _show_image(self, path):
        try:
            try:
                from PIL import Image, ImageTk
                im = Image.open(path)
                im.thumbnail((780, 460))
                self._preview_img = ImageTk.PhotoImage(im)
            except ImportError:
                img = tk.PhotoImage(file=str(path))
                # 大きすぎる場合は間引き
                factor = max(1, img.width() // 780, img.height() // 460)
                if factor > 1:
                    img = img.subsample(factor, factor)
                self._preview_img = img
            self.preview.configure(image=self._preview_img, text="")
        except Exception as e:
            self.preview.configure(image="", text=f"プレビュー失敗: {e}")


# 論文ビルドを Python から実行するスニペット（pdflatex/bibtex を順に呼ぶ）
_BUILD_SNIPPET = (
    "import subprocess, sys, shutil;"
    "from pathlib import Path;"
    "d=Path('paper');"
    "tex=shutil.which('pdflatex');"
    "bib=shutil.which('bibtex');"
    "print('pdflatex:',tex,'bibtex:',bib);"
    "(_ for _ in ()).throw(SystemExit('pdflatex が見つかりません')) if tex is None else None;"
    "subprocess.run([tex,'-interaction=nonstopmode','main.tex'],cwd=str(d));"
    "subprocess.run([bib,'main'],cwd=str(d)) if bib else print('bibtex なし: 参考文献はスキップ');"
    "subprocess.run([tex,'-interaction=nonstopmode','main.tex'],cwd=str(d));"
    "subprocess.run([tex,'-interaction=nonstopmode','main.tex'],cwd=str(d));"
    "print('build done -> paper/main.pdf')"
)


if __name__ == "__main__":
    App().mainloop()
