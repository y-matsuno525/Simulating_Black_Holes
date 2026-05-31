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

## フェーズ3: 計算コアのリファクタ

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| C4 | グローバル変数を `SimulationParams` に集約 | 未着手 |
| C1 | `build_operator_lists()` のベクトル化 | 未着手 |
| C2/C3 | BdG 疎構造化 / 時間発展テーブル事前計算 | 未着手 |
| C5/C6/C7 | 定数・許容誤差の集約 + 型ヒント | 未着手 |

## フェーズ4: テスト・ドキュメント拡充

| 項目 | 内容 | 状態 |
| --- | --- | --- |
| D4 | 回帰テスト追加 | 未着手 |
| A5 | config スキーマ検証 | 未着手 |
| E1 | 英語 README（任意） | 未着手 |
