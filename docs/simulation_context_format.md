# シミュレーション相談用コンテキストの渡し方

このファイルは、BdG ブラックホール/ホワイトホール格子シミュレーションについて、AI や共同研究者に相談するときに渡す情報のフォーマットです。

目的は、単に「この結果が変」と伝えるのではなく、物理条件、数値条件、期待している振る舞い、実際の出力を同じ単位系で共有し、計算ミス・実装ミス・物理的解釈のずれを切り分けやすくすることです。

## 最小フォーマット

短い質問でも、最低限この形で渡すと議論しやすくなります。

```markdown
## 聞きたいこと
<知りたい変化、疑っている量、確認したい物理的意味を書く>

## 実行条件
- run: `<outputs/<run_name>>`
- L: `<格子点数>`
- l: `<物理長>`
- epsilon: `<l/L>`
- dt: `<時間刻み>`
- t_i, t_f: `<時間範囲>`
- PBC: `<true/false>`
- scenario: `<BH_chi_plus / WH_chi_plus / BH_chi_minus / null>`
- chirality: `<chi_plus / chi_minus>`
- beta_sign: `<plus / minus>`
- beta_profile: `<pos_horizon / centered_horizon / flat>`
- surface_gravity_beta: `<true/false>`
- p, m: `<値>`
- j0, sigma: `<初期波束中心と幅。格子単位>`

## 背景 beta
- beta 式: `<config.py の beta_expression_from_config 相当>`
- horizon positions j: `<horizon_positions.txt または summary.json>`
- horizon positions x: `<j * epsilon>`
- surface gravity estimate: `<kappa = |d beta/dx|_horizon>`

## 見ている observable
- observable: `<H_p / H_m / H_pm / c_dag_c / FFT など>`
- データ: `<CSV/TXT/PNG/GIF のパス>`
- vacuum subtraction: `<している/していない。現在の H_p, H_m, H_pm は真空差し引き後>`

## 期待
<物理的・数値的にこうなるはず、という予想を書く>

## 実際
<summary.json、標準出力、図、数値の特徴を書く>

## 判断してほしいこと
<計算式、実装、パラメータ選び、物理解釈のどれを見てほしいかを書く>
```

## 詳細フォーマット

重めの相談や、MCP サーバーに解析してもらう前提では、次の粒度まで渡すのが理想です。

```markdown
# Simulation Question

## Question
<例: L を増やしたときに H_p sigma 最小時刻と停滞時間が収束しているか知りたい>

## Physical Setup
- model: 1D lattice BdG Hamiltonian
- physical length l: `<値>`
- lattice sites L: `<値または掃引リスト>`
- lattice spacing epsilon: `<l/L>`
- boundary condition: `<open / periodic>`
- p: `<値>`
- m: `<値>`
- beta profile: `<式または種類>`
- horizon condition: `|beta| = 1`
- selected chirality: `<chi_plus / chi_minus>`
- scenario meaning:
  - BH_chi_plus: chi_plus packet in black-hole-like background
  - WH_chi_plus: chi_plus packet in white-hole-like background
  - BH_chi_minus: chi_minus packet in black-hole-like background

## Initial Packet
- j0: `<格子位置>`
- x0: `<j0 * epsilon>`
- sigma_lattice: `<sigma>`
- sigma_physical: `<sigma * epsilon>`
- initial direction sign: `<+1 right / -1 left>`
- note: `scenario != null` の場合、進行方向は chirality から決まる

## Numerics
- dt: `<値>`
- dt / epsilon: `<値>`
- time samples: `<len(times)>`
- matrix size: `<2L x 2L>`
- outputs enabled:
  - heatmaps: `<true/false>`
  - gifs: `<true/false>`
  - surface_gravity_fit: `<true/false>`
  - fft: `<true/false>`

## Diagnostics Already Checked
- BdG Hermitian: `<pass/fail/unknown>`
- particle-hole eigenvalue pairing: `<pass/fail/unknown>`
- particle-hole eigenvector reconstruction: `<pass/fail/unknown>`
- local energy density Hermitian: `<pass/fail/unknown>`
- max imaginary part of observables: `<値または unknown>`
- state norm conservation: `<値または unknown>`
- NaN/inf: `<なし/あり/unknown>`

## Output Summary
- run directory: `<outputs/<run_name>>`
- summary.json:
  - H_p_sigma_min_time: `<値>`
  - H_p_sigma_min: `<値>`
  - H_m_sigma_min_time: `<値>`
  - H_m_sigma_min: `<値>`
  - stagnation_time: `<値または null>`
  - horizon_positions_j: `<リスト>`
  - horizon_positions_x: `<リスト>`
- key files:
  - H_p: `<H_p_val.csv / figures/H_p.png>`
  - H_m: `<H_m_val.csv / figures/H_m.png>`
  - sigma: `<H_m_sigmas.txt>`
  - x average: `<x_ave.txt>`
  - FFT: `<*_fft_val.csv>`

## Expected Behavior
<例: L を増やしても物理幅、beta の物理形状、dt/epsilon を固定すれば、H_p_sigma_min_time は一定値に近づくはず>

## Observed Behavior
<例: L=100, 200, 300 で停滞時間が単調に伸びる。H_m_sigmas の初期値も L に依存している>

## Request
<例: 物理的な有限格子効果か、sigma_fraction の固定方法によるスケーリングミスか、実装ミスかを切り分けてほしい>
```

## 質問例と解答例

### 例 1: サイト数を増やしたときの変化

#### 入力例

```markdown
## 聞きたいこと
L を 100, 200, 300 に増やしたとき、停滞時間が収束しているか知りたい。

## 実行条件
- l: 6.283185307179586
- L: [100, 200, 300]
- scenario: BH_chi_minus
- surface_gravity_beta: true
- surface_gravity_beta_width: 1
- p: 0.0001
- m: 0.0001
- PBC: false
- dt_scale: 0.01
- dt: 0.03, 0.015, 0.01
- sigma_fraction: surface_gravity_beta により 0.003 に上書き

## 見ている observable
- H_p_sigma_min_time
- stagnation_time
- H_m_sigmas.txt

## 期待
物理条件を固定しているなら、L を増やすほど停滞時間はある値に近づくはず。

## 実際
L が大きいほど停滞時間が増えているように見える。

## 判断してほしいこと
これは有限サイズ効果か、sigma のスケーリングの問題か、実装ミスか。
```

#### 解答例

```markdown
まず確認すべき点は、`sigma_fraction` が `sigma = sigma_fraction * L` として格子幅を決めていることです。
`l` を固定して `L` を増やすと `epsilon = l/L` は小さくなりますが、`sigma * epsilon = sigma_fraction * l` なので、物理幅は固定されています。
したがって、`sigma_fraction` だけを見る限り、L 掃引時の初期波束幅スケーリングは自然です。

一方で `surface_gravity_beta_width` は `tanh(width * (j-L/2) * epsilon)` に入っているので、これも物理座標 x に対する幅として固定されています。
`dt = dt_scale * (300/L)` なので、`dt/epsilon` も L に対してほぼ一定です。

したがって、停滞時間の L 依存が強い場合、次の順で疑うのがよいです。

1. 有限サイズ・境界反射の影響
2. `stagnation_position` が物理座標として各 L で同じ場所を指しているか
3. `H_p_sigma_min_time` の定義が、同じ物理現象を拾っているか
4. `H_p_sigmas` の最小値がノイズや初期過渡を拾っていないか
5. 固有値ペア、Hermiticity、虚部、norm 保存に異常がないか

次にやるべき計算は、各 L の `summary.json`, `x_ave.txt`, `H_p_sigmas.txt`, `H_m_sigmas.txt` を同じ物理時刻軸で重ね、`x_ave` が境界に近づく前だけで比較することです。
```

### 例 2: `H_m` の量がおかしい

#### 入力例

```markdown
## 聞きたいこと
`H_m` の heatmap だけ振幅が極端に大きく、負の値も目立つ。計算式か符号を間違えていないか見たい。

## 実行条件
- run: outputs/20260513_013600_BH_chi_minus_L300
- L: 300
- scenario: BH_chi_minus
- surface_gravity_beta: true
- p: 0.0001
- m: 0.0001
- PBC: false

## 見ている observable
- H_m_val.csv
- H_m_0.csv
- figures/H_m.png

## 期待
H_m は horizon 付近で特徴的に変化するが、虚部はほぼ 0 で、端点の異常値は出ないはず。

## 実際
H_m の一部だけ桁が大きい。

## 判断してほしいこと
物理的にありえる負エネルギー密度か、実装ミスか。
```

#### 解答例

```markdown
`H_m` は真空期待値を差し引いた局所エネルギー密度なので、負の値そのものは異常とは限りません。
ただし、桁違いのスパイクがある場合は、物理的解釈の前に数値診断を通すべきです。

確認順は次の通りです。

1. `H_m` operator が各 site で Hermitian か
2. `compute_expectation_profile` 後の虚部最大値が十分小さいか
3. 開境界で端点 `j=0`, `j=L-1` がゼロ化されているか
4. `beta_half = beta(j+1/2)` が horizon 付近で期待通り `+1` または `-1` を横切るか
5. `H_m_core` の pairing 項の符号が `H_p_core` と対応しているか
6. 真空差し引き前の値と差し引き後の値を比較し、巨大な cancellation が起きていないか

このコードでは `build_energy_densities` 内で `H_m_j` の Hermiticity assert が入っているため、行列としての非 Hermitian は実行時に検出されるはずです。
次に必要なのは、時間発展後の期待値配列について `max(abs(imag(H_m)))`、`max(abs(real(H_m)))`、最大値の site/time を出す診断です。
```

### 例 3: surface gravity fit が理論と合わない

#### 入力例

```markdown
## 聞きたいこと
`H_m_sigmas` の指数フィットで得た成長率が、`kappa = |d beta/dx|_horizon` と合わない。

## 実行条件
- surface_gravity_beta: true
- surface_gravity_beta_width: 1
- beta: tanh(width * (j-L/2) * epsilon) + 1
- scenario: BH_chi_minus
- chirality: chi_minus
- selected horizon: beta = +1

## 出力
- figures/surface_gravity_fit.png
- H_m_sigmas.txt
- summary.json

## 期待
指数成長率 B が kappa に近い。

## 実際
B と kappa の relative error が大きい。

## 判断してほしいこと
fit 範囲、horizon 選択、sigma 定義、beta 微分のどれが怪しいか。
```

#### 解答例

```markdown
`surface_gravity_beta = true` の beta は `beta(x) = tanh(width * (x - l/2)) + 1` です。
`chi_minus` では target beta が `+1` なので、選ばれる horizon は beta が +1 になる位置、つまり tanh の中心付近です。
このとき理想的には `d beta / dx = width * sech^2(0) = width` なので、`surface_gravity_beta_width = 1` なら kappa はおよそ 1 です。

fit が合わない場合、最初に疑うのは fit 対象の時間範囲です。
`H_m_sigmas` の全時刻を単一指数で fit すると、初期過渡、境界効果、packet が horizon から離れた後の振る舞いまで混ざります。

次の診断が必要です。

1. horizon 近傍に packet がいる時間範囲だけで fit する
2. `sigma_delta_x = H_m_sigma * epsilon` に直して物理単位で比較する
3. `x_ave.txt` と `H_m_sigmas.txt` を重ね、指数成長区間を手動または自動で選ぶ
4. `beta_derivative_at_j` の差分幅 `physical_step` を変えて kappa の安定性を見る
5. `H_m_sigma` が負やゼロに近い区間を fit に入れていないか確認する
```

### 例 4: 設定が意図通り反映されているか

#### 入力例

```markdown
## 聞きたいこと
`config.json` では `chirality: chi_plus`, `beta_sign: minus` にしたが、実行ログでは `BH_chi_minus` になっている。これはバグか。

## 実行条件
- surface_gravity_beta: true
- scenario: BH_chi_minus
- chirality: chi_plus
- beta_sign: minus
- j0_fraction: 0.2
- sigma_fraction: 0.05

## 実際
実行時に `chirality`, `beta_sign`, `j0_fraction`, `sigma_fraction` が上書きされる。
```

#### 解答例

```markdown
これは現在の仕様です。
`surface_gravity_beta = true` の場合、`prepare_config` は surface gravity 比較用の基準設定として、`scenario = BH_chi_minus`, `j0_fraction = 0.48`, `sigma_fraction = 0.003` を優先します。
さらに `scenario = BH_chi_minus` により、`chirality = chi_minus`, `beta_sign = plus` が決まります。

バグかどうかの判断は、目的によります。
surface gravity の標準実験を再現したいなら正常です。
手動で chirality や初期位置を変えたいなら、`surface_gravity_beta` の自動上書きを無効にするか、手動実験用の別 profile として切り出すのがよいです。
```

## MCP に渡したい構造化データ

将来 MCP サーバー化するなら、自然言語だけでなく、各 run から次の JSON を返せると便利です。

```json
{
  "run_dir": "outputs/20260513_013600_BH_chi_minus_L300",
  "config": {
    "L": 300,
    "l": 6.283185307179586,
    "epsilon": 0.020943951023931952,
    "dt": 0.01,
    "scenario": "BH_chi_minus",
    "chirality": "chi_minus",
    "beta_profile": "pos_horizon",
    "surface_gravity_beta": true,
    "surface_gravity_beta_width": 1,
    "p": 0.0001,
    "m": 0.0001,
    "PBC": false,
    "j0": 144,
    "sigma": 0.9
  },
  "derived": {
    "dt_over_epsilon": 0.477464829275686,
    "sigma_physical": 0.01884955592153876,
    "horizon_positions_j": [150.0],
    "horizon_positions_x": [3.141592653589793],
    "surface_gravity_kappa": 1.0
  },
  "summary": {
    "H_p_sigma_min_time": null,
    "H_p_sigma_min": null,
    "H_m_sigma_min_time": null,
    "H_m_sigma_min": null,
    "stagnation_time": null
  },
  "diagnostics": {
    "bdg_hermitian": "unknown",
    "particle_hole_pairing": "unknown",
    "max_observable_imag": "unknown",
    "state_norm_drift": "unknown",
    "nan_or_inf": "unknown"
  }
}
```

## 判断時の優先順位

異常そうな結果を見たときは、次の順番で切り分けると混乱しにくいです。

1. 設定が意図通り prepare されているか
2. 単位が格子単位か物理単位か
3. horizon 位置と beta の符号が chirality と合っているか
4. Hermiticity と粒子正孔対称性が保たれているか
5. 時間発展の norm と虚部が正常か
6. observable の定義と真空差し引きが期待通りか
7. 境界、有限 L、fit 範囲などの数値解析上の問題か
8. 最後に物理的に新しい振る舞いとして解釈できるか
