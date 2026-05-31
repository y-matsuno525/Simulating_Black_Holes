# Simulating Black Holes

1 次元格子上の BdG ハミルトニアンを使って、ブラックホール/ホワイトホール的な horizon 背景での波束の時間発展を数値計算するリポジトリです。

現在の実行入口は `main.py` です。設定は `config.json` に出し、計算本体は `simulation.py`、既定値と設定検証は `config.py` に分けています。古い実験用スクリプト、未使用テスト、`my_func/` 内の補助モジュールは削除済みです。

## 構成

| パス | 説明 |
| --- | --- |
| `main.py` | 設定を読み込み、シミュレーションを起動する薄い入口です。 |
| `config.json` | 普段編集する設定ファイルです。物理パラメータ、初期状態、出力 ON/OFF をここで変えます。 |
| `config.py` | 既定設定、設定ファイル読み込み、別名変換、派生量計算をまとめています。 |
| `simulation.py` | BdG 行列生成、対角化、初期波束、演算子生成、時間発展、図・GIF・解析出力を行う計算本体です。 |
| `outputs/` | 実行ごとの結果保存先です。Git 管理外です。 |
| `.gitignore` | 実験結果、画像、キャッシュを Git に入れないための除外設定です。 |

## 実行方法

初回だけ依存ライブラリを入れます。

```bash
python3 -m pip install -r requirements.txt
```

その後、シミュレーションを実行します。

```bash
python3 main.py
```

計算は `L` や `t_f` によって重くなります。設定は `config.json` を編集して変更します。

必要な主なライブラリ:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow` または matplotlib の GIF 保存に必要な PillowWriter 環境

数値計算の健全性チェックと演算子ベクトル化の回帰テストを同梱しています。

```bash
python -m unittest discover -s tests
```

push / PR では GitHub Actions（`.github/workflows/test.yml`）で自動実行されます。

## 主な設定

`config.json` でよく触る項目です。

| キー | 説明 |
| --- | --- |
| `L` | 格子点数。BdG 行列サイズは `2L x 2L` です。 |
| `l` | 物理空間の長さ。標準では `2π`。 |
| `p`, `m` | BdG ハミルトニアンに入る係数です。現在は位置依存 `p(j)` は使いません。 |
| `scenario` | `"BH_chi_plus"`, `"WH_chi_plus"`, `"BH_chi_minus"` から選びます。 |
| `t_i`, `t_f`, `dt_scale` | 時間範囲と時間刻みを決めます。実際の `dt` は `dt_scale * (300/L)` です。 |
| `PBC` | 周期境界条件を使う場合は `True`。 |
| `beta_profile` | `"pos"`, `"center"`, `"flat"` から選びます。 |
| `surface_gravity_beta` | `true` にすると他の beta 設定より優先して `tanh(surface_gravity_beta_width*(j-L/2)epsilon)+1` を使い、`scenario` は `"BH_chi_minus"`、`j0_fraction` は `0.48`、`sigma_fraction` は `0.003` になります。 |
| `surface_gravity_beta_width` | `surface_gravity_beta` 用の tanh 幅です。既定は `0.1`。 |
| `beta_width`, `beta_amplitude`, `beta_center_fraction` | horizon 位置付き beta profile の形を決めます。 |
| `sigma_fraction` | 初期 Gaussian packet の幅を決めます。初期位置と進行方向は既定値を使うため、通常は指定不要です。 |
| `output_base_dir` | run ごとの出力を置く親ディレクトリです。既定は `outputs`。 |
| `run_name` | 出力ディレクトリ名です。`null` の場合は時刻・シナリオ名・L から自動生成されます。`scenario: null` の手動指定では `manual_<chirality>_beta_<beta_sign>` が入ります。 |

よく使う状況は、`scenario` だけで用意します。内部では `chirality` と `beta_sign` に展開されます。

| 状況 | `scenario` | 内部の `chirality` | 内部の `beta_sign` |
| --- | --- | --- | --- |
| BH に chi+ を置く | `"BH_chi_plus"` | `"chi_plus"` | `"minus"` |
| WH に chi+ を置く | `"WH_chi_plus"` | `"chi_plus"` | `"plus"` |
| BH に chi- を置く | `"BH_chi_minus"` | `"chi_minus"` | `"plus"` |

シナリオを使わずに手動指定したい場合は、`scenario` を `null` にして `chirality` と `beta_sign` を直接指定します。

```json
"scenario": null,
"chirality": "chi_plus",
"beta_sign": "minus"
```

## 残した解析機能

古いスクリプトから、必要なものだけ `simulation.py` に統合しています。

| 機能 | 出力 |
| --- | --- |
| BdG mode function の保存 | `outputs/<run_name>/figures/mode_function_p=<p>_k=<n>.png` |
| 測地線データ生成 | `outputs/<run_name>/geodesic.dat`, `outputs/<run_name>/figures/geodesic.png` |
| `H_m` 幅の指数フィット | `outputs/<run_name>/figures/surface_gravity_fit.png` と標準出力の fit 係数 |
| 停滞時間の計算 | 標準出力に `線形部分通過時間` と `停滞時間` を表示 |

対応する設定:

| キー | 説明 |
| --- | --- |
| `mode_function_count` | 保存する mode function の本数。不要なら `0`。 |
| `geodesic_points` | 初期波束中心から伸ばす測地線の保存点数。 |
| `surface_gravity_output_path` | surface gravity fit 図の保存先。 |
| `stagnation_position` | 位置平均がこの物理座標を超えた時刻から、`H_p` sigma 最小時刻までを停滞時間として計算します。 |
| `fft_observables` | FFT を保存・描画する observable 名のリストです。例: `["H_p"]`。 |
| `fft_remove_spatial_mean` | FFT 前に各時刻 profile の空間平均を引く場合は `true`。 |
| `animation_marker_fractions_pbc` | PBC の GIF に将来 horizon 補助線を戻すときの格子位置マーカーです。値は `j/L` の割合です。 |
| `animation_marker_fractions_open` | 開境界の GIF 用の格子位置マーカーです。値は `j/L` の割合です。 |
| `mode_function_marker_fractions` | BdG mode function 図に引く物理位置マーカーです。値は `x/l` の割合です。 |

重い出力は `outputs` で個別に止められます。

| キー | 説明 |
| --- | --- |
| `outputs.show_beta_profile` | notebook 表示用の `beta_profile.csv` を保存します。 |
| `outputs.heatmaps` | `H_p`, `H_m`, `H_pm`, `c_dag_c` の heatmap を保存します。 |
| `outputs.gifs` | 時間発展 GIF を保存します。 |
| `outputs.mode_functions` | BdG mode function を保存します。 |
| `outputs.geodesic` | `geodesic.dat` と確認図を生成します。 |
| `outputs.surface_gravity_fit` | `H_m_sigmas` の指数フィットを行います。 |
| `outputs.fft` | 各時刻の FFT スペクトル CSV、heatmap、GIF を生成します。 |

## 通常の出力

`main.py` 実行後、`outputs/<run_name>/` に以下が生成されます。`config.json` もコピーされるので、後から実行条件を追えます。

| 出力 | 説明 |
| --- | --- |
| `config.json` | その run で実際に使った設定。 |
| `summary.json` | 主要パラメータ、sigma 最小時刻、停滞時間などの要約。 |
| `x_ave.txt` | 時刻と `H_m` profile から計算した平均位置。 |
| `H_m_sigmas.txt` | 時刻と `H_m` の幅変化。 |
| `geodesic.dat` | heatmap に重ねる測地線データ。 |
| `H_p_val.csv`, `H_m_val.csv`, `H_pm_val.csv`, `c_dag_c_val.csv` | heatmap/GIF に使う時系列 profile。先頭列が `time`、以降が格子点です。 |
| `H_p_0.csv`, `H_m_0.csv`, `H_pm_0.csv`, `c_dag_c_0.csv` | 各 profile の初期時刻データ。 |
| `figures/H_p.png`, `figures/H_m.png`, `figures/H_pm.png` | 各エネルギー密度の時間発展 heatmap。 |
| `figures/c_dag_c.png` | `c_j^\dagger c_j` の時間発展 heatmap。 |
| `figures/H_p.gif`, `figures/H_m.gif`, `figures/H_pm.gif` | 時刻ごとの profile を見る GIF。 |
| `figures/mode_function_p=<p>_k=<n>.png` | BdG mode function。 |
| `figures/geodesic.png` | 生成した測地線の確認図。 |
| `figures/surface_gravity_fit.png` | `H_m` 幅の指数フィット図。 |
| `<name>_fft_k.csv`, `<name>_fft_val.csv` | `fft_observables` に指定した observable の FFT 波数と時系列スペクトル。 |
| `figures/<name>_fft.png`, `figures/<name>_fft.gif` | FFT スペクトルの時間発展 heatmap と GIF。 |

## データ管理方針

Git に入れるもの:

- 計算コード: `main.py`, `config.py`, `simulation.py`
- 再現用の標準設定: `config.json`
- 説明文書: `README.md`

Git に入れないもの:

- `outputs/`
- `figure/`, `data/`
- `*.png`, `*.gif`
- 実行で生成される `*.txt`, `*.dat`, `*.csv`
- `__pycache__/`, `*.pyc`

## 現在使わないもの

次の機能は統一方針により削除済み、または `main.py` には入れていません。

- 位置依存 `p(j)` と `diff_p(j)`
- `ll` / `ul` の beta profile
- K-form 系の別定式化
- 未使用の固有ベクトル補助関数
- 古いテストスクリプト
- `cj c_j^\dagger` などの未使用演算子チェック

## 論文 (PRB 草稿) の統合と再現

論文 `paper/main.tex` が参照する計算・データ・図生成・解析を本リポジトリに集約しました。
詳しい作業記録は `CONSOLIDATION_PROGRESS.md` を参照してください。

### ディレクトリ

| パス | 内容 |
| --- | --- |
| `paper/` | 論文一式（`main.tex`, `books.bib`, `draft_jp.txt`, `figure/`=論文本体の図8枚）。 |
| `paper_data/` | 5 つの旧フォルダ（master_thesis, master_thesis_p=0, thesis, old(L=500)）から集約した計算 CSV・geodesic・σ。**約700MBのため git 管理外**（ディスク上のみ）。 |
| `paper_figures/` | 論文図を保存データから再生成するスクリプト群。出力は `paper_figures/generated/`。 |
| `analysis/` | 補助解析（分散関係、PG 計量幾何、停滞時間の logL スケーリング）。 |
| `gui.py` | 計算・図生成・論文ビルドを行う Tkinter 統合 GUI。 |
| `run_sim.py` | `config.json` を壊さず上書きパラメータで `simulation.py` を実行する入口。 |

### 図と生成元の対応

| `paper/figure` の図 | 生成スクリプト | 主なデータ／計算 |
| --- | --- | --- |
| `dispersion.png`, `p0_dispersion.png` | `paper_figures/make_dispersion.py` | `analysis/dispersion_relation.py`（解析式 εE±=β sin k ± √(p²(1−cos k)²+sin²k)） |
| `doubler_fft.png` | `paper_figures/make_doubler_fft.py` | β=1.2 分散 + 内部 run の空間 FFT |
| `p=0_BH.png`, `p=1_BH.png` | `paper_figures/make_bh_panels.py` | `paper_data/p{0,1}/lr_p, ur_p_2` + geodesic |
| `WH_p=0.png`, `WH_p=1.png` | `paper_figures/make_wh_panels.py` | `paper_data/p{0,1}/ur_m, ur_p` + `analysis/stagnation_logL.py` |
| `sg.png` | `paper_figures/make_surface_gravity.py` | `paper_data/H_m_sigmas_p{0,1}.txt` の指数フィット |

> 生成図は保存データからの「同等再現図」です（論文 PDF の各パネルは特定の初期条件 run を
> 使っており、レイアウトは一致しません）。論文本体 `paper/figure/` は上書きせず保持しています。

### 再現手順

```bash
# 1) 論文図を一括再生成（paper_figures/generated/ に出力）
python paper_figures/make_all.py

# 2) 論文をビルド（要 TeX: pdflatex/bibtex）
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# 3) もしくは GUI から計算・図生成・ビルドをまとめて操作
python gui.py
```

## 注意

- 実行結果は `outputs/<run_name>/` に分かれます。`run_name` を固定すると同じディレクトリを上書きします。
- `__pycache__/` は Python の自動生成キャッシュなので、保守対象ではありません。
- `paper_data/` は容量が大きいため git 管理外です。元データは各旧フォルダにも残しています。

## License

本リポジトリは MIT License で公開しています。詳細は [LICENSE](LICENSE) を参照してください。
