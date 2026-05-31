# 統合作業 進捗レポート

論文(main.tex)関連の計算・データ・図生成・論文一式を本リポジトリに統合し、Tkinter GUI から
一気通貫で実行可能にする作業の進捗。プラン: `~/.claude/plans/main-tex-...md`。

作業ブランチ: `paper-consolidation`

## ステータス凡例
- ⬜ 未着手 / 🟦 進行中 / ✅ 完了

## タスク一覧

| # | タスク | 状態 | コミット | 備考 |
|---|---|---|---|---|
| 1 | データ集約（CSV/geodesic→`paper_data/`） | ✅ | 69baf3d | 5フォルダ198CSV集約。686MBのためgitignore |
| 2 | 論文一式→`paper/` 取込 | ✅ | 69baf3d | main.tex/books.bib/draft/figure(8図) |
| 3 | Notebook図生成ロジック把握 | ✅ | - | CSV形式(ヘッダ無(999,300))・geodesic形式を確定 |
| 4 | `paper_figures/` スクリプト群作成 | ✅ | (本コミット) | common+make_*×5+make_all、全7図生成確認 |
| 5 | `analysis/` 補助計算スクリプト作成 | ✅ | 34736f4 | dispersion/majorana/stagnation 全て動作確認 |
| 6 | `gui.py`（Tkinter統合GUI）作成 | ✅ | (本コミット) | 3タブ+ログ+図プレビュー、run_sim.py、import確認 |
| 7 | .gitignore/README/requirements更新・検証 | 🟦 | - | gitignore✅、論文ビルド✅、README/requirements/テスト 残 |

## 主要な判明事項（実装中に確定）
- `*_val.csv` はヘッダ無し・time列無しの `(999,300)`。geodesic は `x,t` ヘッダ付き。
- 表面重力: master_thesis の `H_m_sigmas.txt` は実は p=0 データ。真の p=1 は std ブランチ
  `p=1_H_m_sigmas.txt`（負開始→ほぼ線形成長、波束変形に対応）。
- 分散の2バンド解析式 `εE±(k)=β sin k ± √(p²(1−cosk)²+sin²k)` を導出、数値対角化と一致(誤差0.025)。
  k=0傾き=β±1、k=πで p=0→ダブラーゼロモード/p=1→±2pギャップ（本文と整合）。
- 生成図は「保存データからの同等再現」。論文PDFの厳密パネル(j0=30等)とはレイアウトが異なるため、
  `paper/figure/`(論文本体)は上書きせず `paper_figures/generated/` へ出力。

## コミット履歴（このブランチ）
- 69baf3d: 論文一式取込・データ集約・gitignore方針
- (本コミット): paper_figures/ 図生成スクリプト群 + analysis/dispersion_relation.py

## 留意点 / 判明事項
- 複合図の元レイアウト生成コードは現存せず、Notebookセルから新規に起こす（完全一致でなく同等図が目標）。
- doubler_fft のFFT元run、WH_p=0 のlogL測定データの所在は実装時に確認。無ければ再生成。
- OGRePy はオプション依存（未導入環境では majorana_metric をスキップ可能に）。

## 検証結果
（実装後に追記）
