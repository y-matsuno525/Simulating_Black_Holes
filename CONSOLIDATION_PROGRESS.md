# 統合作業 進捗レポート

論文(main.tex)関連の計算・データ・図生成・論文一式を本リポジトリに統合し、Tkinter GUI から
一気通貫で実行可能にする作業の進捗。プラン: `~/.claude/plans/main-tex-...md`。

作業ブランチ: `paper-consolidation`

## ステータス凡例
- ⬜ 未着手 / 🟦 進行中 / ✅ 完了

## タスク一覧

| # | タスク | 状態 | コミット | 備考 |
|---|---|---|---|---|
| 1 | データ集約（CSV/geodesic→`paper_data/`） | ⬜ | - | 元フォルダは保持しコピー |
| 2 | 論文一式→`paper/` 取込 | ⬜ | - | main.tex/books.bib/draft/figure |
| 3 | Notebook図生成ロジック把握 | ⬜ | - | figure.ipynb×2, plot.ipynb 等 |
| 4 | `paper_figures/` スクリプト群作成 | ⬜ | - | common+make_*×5+make_all |
| 5 | `analysis/` 補助計算スクリプト作成 | ⬜ | - | dispersion/majorana/stagnation |
| 6 | `gui.py`（Tkinter統合GUI）作成 | ⬜ | - | 3タブ+ログ+図プレビュー |
| 7 | .gitignore/README/requirements更新・検証 | ⬜ | - | make_all/ビルド/テスト/GUI |

## コミット履歴（このブランチ）
（順次追記）

## 留意点 / 判明事項
- 複合図の元レイアウト生成コードは現存せず、Notebookセルから新規に起こす（完全一致でなく同等図が目標）。
- doubler_fft のFFT元run、WH_p=0 のlogL測定データの所在は実装時に確認。無ければ再生成。
- OGRePy はオプション依存（未導入環境では majorana_metric をスキップ可能に）。

## 検証結果
（実装後に追記）
