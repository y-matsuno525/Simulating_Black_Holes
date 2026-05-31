# 改善作業 進捗記録

プロジェクト診断（問題点・改善案の網羅）に基づく改善の実施ログ。
診断の全体像は別途プラン文書を参照。フェーズ単位で着手し、切りのいい所でコミット・プッシュする。

## フェーズ1: 基盤整備 ✅ 完了

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| A1 | `LICENSE`（MIT）追加 + README に License セクション | ✅ |
| A2 | `requirements.txt` にバージョン下限を固定、Python 3.9+ を明記 | ✅ |
| A3 | `.github/workflows/test.yml` で push/PR 時に unittest 自動実行（3.10–3.12, MPLBACKEND=Agg） | ✅ |
| B1 | `replot_paper_figs.py` の個人パスを除去（`__file__` 基準 + 環境変数 `SBH_ARCHIVE_BASE`）。レガシーツールである旨を明記 | ✅ |
| 付随 | README の古い絶対パスリンク修正 | ✅ |

検証: `python -m unittest discover -s tests -p "test_*.py"` が PASS。

## フェーズ2: Single Source of Truth 化（再現性）✅ 完了

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| B2 | 論文図の horizon 値を `paper_figures/common.py` に一本化（`replot` は import）。実行パイプライン側は `config.compute_horizon_positions_from_config` が SSoT である旨を明記 | ✅ |
| B3 | 正準 BH β プロファイルを `analysis/dispersion_relation.py`（`beta_profile_x`/`beta_horizon_x`）に一本化。`majorana_metric` / `make_dispersion` は import に変更 | ✅ |
| B4 | 図出力先を `paper_figures/generated/` に統一（`replot` の出力先も変更） | ✅ |
| B5 | `_gui_overrides.json` の撤廃（`tempfile` で一時ディレクトリに書き出し） | ✅ |

検証: unittest 4件 PASS、`majorana_metric` / `dispersion_relation` / `make_dispersion` / `make_bh_panels` / `make_wh_panels` / `replot_paper_figs` が完走し図が `generated/` に出力。horizon 値 BH=212.81 / WH=87.19 で一致。

## フェーズ3: 計算コアのリファクタ（一部）✅ / ⏸

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| C1 | `build_operator_lists()` を O(L^4)→O(L^3) にベクトル化（外積+対角項。新ヘルパー `_cj_dag_cj_op`/`_cj1_cj_op`/`_cj1_dag_cj_op`）。L=120 で約0.1秒 | ✅ |
| C3 | 時間発展ループの n ループを位相ベクトル積に置換（`exp(-iE t)` を一括適用） | ✅ |
| C5/C6 | 散在していた許容誤差を定数化（`ATOL_HERMITIAN`/`IMAG_WARN_THRESHOLD`/`PH_PAIR_ATOL`） | ✅ |
| C7 | ベクトル化ヘルパー・`build_operator_lists` に型ヒント付与（全面付与は今後） | 一部 |
| C2 | BdG 行列のベクトル化 | ⏸ 見送り（O(L^2) で非支配的・β リンク平均/PBC 角の索引が繊細。効果<リスク） |
| C4 | グローバル変数を `SimulationParams` に集約 | ⏸ 見送り（~40関数に波及する大規模改修。現行テスト4件+回帰では安全に検証しきれず別タスク化） |

検証: L=10(PBC)/L=12(open) で演算子リスト・エネルギー密度・真空値・固有値の全24配列がリファクタ前と一致（`np.allclose`）。unittest 4件 PASS。小規模エンドツーエンド実行（run_sim.py, L=12）も完走。

## フェーズ4: テスト・ドキュメント拡充 ✅（E1除く）

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| D4 | `tests/test_operators_regression.py` 追加。ベクトル化前の素朴な O(L^4) 実装を参照実装として保持し、ベクトル化版と小規模 L(=6, PBC両方)で完全一致を固定 | ✅ |
| A5 | `config.validate_config()` を追加し `prepare_config` で早期検証（L/l/p/m/t_f/PBC/dt_scale/各fraction の型・範囲）。不正値テストも追加 | ✅ |
| E1 | 英語 README（任意） | ⏸ 見送り（任意項目。日本語READMEは充実） |

検証: `python -m unittest discover -s tests` で 7 件（既存4 + 新規3）PASS。

---

## 残課題（別タスク化）

- **C4**: グローバル変数の `SimulationParams` 全面集約（~40関数に波及・大規模）。
- **C2**: `build_bdg_matrix` のベクトル化（非支配的・索引が繊細）。
- **E1**: 英語 README / API ドキュメント。

いずれも今回の検証範囲（unittest 7件 + 演算子回帰）では安全に担保しきれない、
または効果が小さいため見送り。着手時は本ファイルのフェーズ構成に追記する。
