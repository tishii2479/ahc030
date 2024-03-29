# 考察メモ

- 良い解が出る方針を探す

# 2/9

- **実行時間制限: 3sec**
- 再挑戦できることは使うのか？
    - コスト1大きすぎるか
    - 候補が絞れてきたら、もしかしたら挑戦した方が期待コストが低いかも
- 何回も同じところを検査するのは？
- 上限はN^2
- 情報理論の復号方法が参考になりそう
- 連結なのは使えるはず
- エラー率ごとのチューニングは必要
- v>1の時の影響は？
    - 複数マスがあるように見える
    - 結局最終的には正確にvが推定できている
    - ほんとか？要検討
    - 大きいマスを見つけると、複数重なっていることがわかる
- Sの大きさはどうすれば良い？
- エラーが0の時は？
- 連結という制約がない場合は？
- 01整数計画問題として定式化できないか？
- 最上位になるには尤度を考える必要はありそうだが、、
    - エラー率ごとのパラメータチューニングでなんとかなるか？20通りだし
- 形の情報は超重要
    - 最終的には配置がどうなっているかを元に推定できるはず
    - 一番可能性が高い配置を探すには？
- 配置を固定するとか？
    - 候補が複数あるなら、一番不確かさなところを調べれば良い
    - ただ、一マスだけは調べられない、、
- コストは`1/sqrt(k)`
- 「集合Sを調べる」は言い換えができないか？
    - k個のマスを調べたいなら、kxkのrankがkの行列を作れば良い
        - 1個の1ベクトルと、k-1個の1個だけ欠けているベクトルを作れば良い
        - エラーが面倒なら、k個の1個だけ欠けているベクトルでもそんなに変わらなそう
        - 自由度がkなので、これが最善そう
    - この時、スコアは`(k-1)/sqrt(k)`

$$
x \sim N \left( v(S) (1 - 2\epsilon) + k\epsilon, k\epsilon(1 - \epsilon) \right) \\
x' = \frac{x - k\epsilon}{1 - 2\epsilon} \\
x' \sim N \left( v(S), \frac{k\epsilon(1 - \epsilon)}{(1 - 2\epsilon)^2} \right)
$$

- ミノを考慮して調べるのが筋が良さそう
    - 全パターンは一様ランダムなので、尤度を調べられる？
    - max(M)=20なので無理そうだけど、近いことはできそう
    - Mの分布めっちゃ小さい
        - 平均ほぼ6、中央値も6
    - 左上のマスの位置がわかれば復元できる
- 入力ケースの観察
    - 何箇所かにかなり集中している
    - 油田の個数は多くない
- N^2/5 ~ N^2/2が埋まる
    - 最大でも半分くらいしか埋まらない
- kはどれくらいにするのかな
- なんか動かさないと肌感が掴めないな

- まずはシンプルな解法を考える
- 上限がN^2なので、直接掘るがないと厳しいな
- 集合と見ると、積や和などの演算ができる？
- 油田がある場所に当たりをつける -> 具体的に探す
- 特徴的な形状（横に4連続とか）を探す
    - ミノを動かして探す

# 2/10

- 失敗回数を0にしながら、コストを下げる必要がある

- 窓を動かして期待値を求める
    - ミノを動かす方が良い？
    - 調べたところは期待値を管理しておく
    - 望みが低いところは調べない？
        - 微改善
- 情報を集めてから、ミノの位置を決定する最適化問題を解く？
    - 自由度はO(M^N^2)
    - 解いたあとで間違っていたら？

- 情報の集め方

1. kxkのマスを動かして、x[i][j]に期待コストを加算していく、c[i][j] += 1する
2. x[i][j] / c[i][j]する
3. 周りを考慮して、x[i][j]を再計算する
    - どうやって？

- 確率が高いところから順に決めるでも十分良いスコアが出そう
    - 不正解になる確率が非常に高いが、、
- 形を工夫（ミノの形状）にすれば、誤検出は小さくなりそう
    - 完全一致が強い

- 不正解な場合はどうする？
    - 複数あり得る解を作って、分散が大きいところを調べる
    - ミノを考慮せずにvを作って、一致しないところを調べる
- x[i][j]の再計算
    - 01整数計画問題として解けそう
    - 線形緩和して良さそう
- 現状のx[i][j]の問題点
    - 値が大きいところ重視してしまう
    - 周囲にある油田の数に影響される
- ミノに当てはめるように候補を決めるか、理想的な場所を探すか
- 正方形だけだと多分だめ

- 工夫ポイント
    - 窓の形
        - ミノによくある形にした方が良さそう
        - 飛び飛びにしたらどうなる？
            - ほぼ一様になる
            - 自分が0か1かは、誤差になってしまう

- 方向性
    - 山登りの改善
    - 追加情報の取得
    - 最初の情報の取得方法
        - 逐次的にするか、など
- 勝手に段階に分けているが、これに固執しない方が良い
    - ミノの位置は特定する必要があるのか？
    - 流石にありそう
- とりあえず全ケースACを目指す

- 自由度がある場所が偏ってしまっている
    - 隣接しているところや、くの字になっているところとか
    - 安定していないところを調べるとか？
- 調べる必要ないところ（ほぼ0のところ）は調べない
    - 最初の調査でも、追加調査でも

- なぜ間違えているか
    - 誤差が大きくて誤って認識している
    - 最適化結果が正しくない
        - 誤差
        - 自由度
    - 誤差に対する対処
        - kの調整
    - 自由度に対する対処
        - Sの形状を変える
        - 縦横一直線とか？
        - 同じ制約になるべく登場しない方が良い
        - ランダムがいいんじゃないか
- 最適化の改善点
    - 明らかに孤立しているところは0に固定する
- 追加調査
    - 怪しいところから複数選ぶことを何回か繰り返す
    - 形状を変える
    - 境界を間違えがち

- 評価項目
    - 分布最適化の誤差（x_error）
        - 推定値がどれくらい外れているか
        - ||ans - x||_2
    - ミノの配置最適化の誤差（optimize_mino_pos_error）
        - 推定値がどれくらい外れているか + ミノに関する最適化
        - ||v - x||_2
    - 真の解からのズレ（error_count）
        - |v != ans|

- TODO:
    - investigateからxを消す

- 確定が欲しいか？
- 追加調査すべき箇所
    - 分散が大きいところ
        - 「ミノの数とxの値の差が大きいところ」になる？
        - |ミノを考慮した最適化 - xを直接最適化|
    - 自由度が大きいところ
        - 自由度が小さくなるように最初に調査したい
        - ばらけた方が強いか
    - 境界
- ランダム強いか、、

- 正規分布なので、尤度最大化できそう？
    - 対数尤度にするので、どれくらい良いかはわからん
    - kの大きさを変えた時にも近似的に計算できそう

- TODO
    - 山登りの改善
        - 近傍の改善
    - 追加情報の取得の工夫
    - 最初の情報の取得方法

- 追加情報取得の工夫
- きついケースの対策
    - 確定が欲しそう

# 2/11

- 情報取得の工夫
    - 最初は一箇所に固めて調査する
    - ミノがある箇所の周辺
        - 不確かなところを調査したい
        - 直前のミノの位置だけでなく、過去履歴も使用したい
    - ミノの形で調査する
    - 特殊な形（一直線など）で調査
- 制約式だけだと、地理的な条件を考慮していない
- 最初の調査は区別した方が良さそう
- 山登り高速化
    - 今の段階でどれくらいやるべきか
    - 方針が変わらないならやって良さそう
        - 絶対使いそうだし

- 愚直な計算量
    - O(QK + M)
- 差分計算の時間
    - O(|Q_i|*|M_i|)

- きついケース対策
    - 途中で答える間隔を減らして、20手くらい残して最後まで情報を収集する
    - 複数の解の候補を作る
        - 低温で焼きながら、良い解がでてきたら答えてみる
        - 多点スタート
    - 確定を使う
    - きついケースかどうか
        - 適当なロジスティック回帰で判定できそう
    - 2s経過したら、特殊モードに入るとか
    - k、step_sizeのスケジュール関数を作る

- 最適化の工夫
    - 制約式の数は減らせないか、、？
    - 最急降下に近いことはできないか？
    - そもそも山登り以外で解けないか？
    - 近傍足りていないのでは？
        - 隣にずらす、強そう

### TODO:

1. 最適化の改善
    - 必要なイテレーション数を減らしたい
    - 近傍の追加・工夫
        - どういう組でswapが起きているか
    - 初期解の工夫
2. （必要なら）きついケースの対策
    - 解を求めるときの工夫
    - 特殊ルールの追加は最後にしたい
3. 情報取得の工夫
    - ちゃんとばらつきを考慮できないか？
    - 途中
        - 境界だけにする
        - 割合を変化させて精度を見る
    - 最初
        - 工夫する余地ありそうだが、、
4. パラメータ調整

- 差分を意識して2つを動かす
    - 並行移動、2、3点swap
    - 重なりが大きくなるように、少しずれているのでは？
- どこかまで調査して、そこから最適化を続けるのが最適っぽい
    - ばらつきが評価できれば、期待値が計算できる？
        - 相当むずそう
    - 定期的に最適化するのはそこまで悪くないか、、
- 連続で最適化する場合は、多点山登りかre-annealingを試す
    - 解の重複チェックはする
    - 単純なkickでも良さそう

### 考察

- 最適化しながら情報を収集する方向性はあるのか
- kは可変にした方が良いのか
    - 確定は使った方が良いのか？
- どこかまで調査して、そこから最適化を続けるのが最適なのでは？
- 2*N^2まで使い切らない方が良いのでは？

# 2/12

- 近傍
    - action_slide_one
    - action_slide_two
    - action_move_one
    - action_swap_two
    - action_swap_three
- 改善点
    - 複数候補を試す
    - 重複が大きくなるようにswapする
        - そのような位置関係を記録しておく
    - 大きさが似たミノをswapする
        - 実際どうか確認する
    - 3点swap
        - 大きい1つと小さい2つをswap

- きついケースは試行回数が大事っぽい
    - m>10なら、特殊処理で良いかも
- iteration数が十分なら、action-swap-threeは入れた方が良い

- 山登りの質を上げれば、スコアが上がることはわかっている
    - 近傍の網羅性
    - 有効な近傍の割合
    - 高速化
- 情報量は十分
    - 改善する余地はある

- 情報量の改善
    - vの分散の大きさに比例させる
    - p ~ V(v_tij)

- セーフティの処理は必要そう、、
- セーフティ処理案
    1. query_countが制限の3/4を超えたら、焼きなまし
    2. いくつか確定する

- sub1: 思ったより相対スコア低いな、、
    - 10*1e9なので、10ケースで失敗しているっぽい
    - 貪欲にすれば、5Bくらいは伸びる

- 情報が足りていないのか、最適化が甘いのか
    - error_countとlossの相関を見る
    - ちゃんと相関している、最適化が甘そう

- イテレーション数増やしたい
- やっぱり適切な場所でちゃんと最適化するのが強そう

# 2/13

- アンサンブルが強い？
- 0の補正

- 方向性

2/14 エラーを（ほぼ）ゼロにする
2/15 パラメータを調整する
2/16〜 情報収集を工夫する

# 2/14

- 確定を使う
- 最適化しながら制約を追加する？

## 解法案

1. パラメータに応じて、調査回数を決定する
2. 初期調査をする
3. 最適化してvを求める
4. vのばらつきに応じて、いくつかのマスを確定する
5. 追加調査する
6. 最適化する
7. 可能性が高い解から出力する

## 現状

- 全ての制約を開示しても、最適解に辿り着けないケースがある
    - 最適化が甘い
        - 近傍が不十分
- 適切な調査回数の決め方がわからない

- 一回最適化が良いとは限らないよなー、、

- 12Gまで来たが、1位はかなり遠い

## 課題

- 簡単なケースで調査回数を過剰に使っている
    - 予測の精度を上げる
    - ステップの設定が適当
- 難しいケースの成功率が低い
    - 確定を使っていない
    - 最適化の高速化、精度が足りない
- kの設定が適当

- 焼きなまし
    - スコアのログ見る
    - 前回の解を渡す
    - 近傍を改善する
- 調査する形
    - 制約を作るときに、位置関係を考慮していない
    - 共起関係？

# 2/15

- ミノの重なりによって難易度が変わるので、パラメータだけから正確に予測するのは無理っぽい
- 実行時間伸ばすとまだ伸びる
- kの設定、ステップの設定を改善したい
- 最適化しなくても良い方法を探す、とか？
    - 自由度など
- 誤差を予測する？
- stepsの調整

- 2倍以下にする必要があるので、何かしら根本的な改善が必要、、

- 情報の集め方？
- 類似している形を見分けられていない
- パラメータの設定が肝な気がするが、、

- 解が出た時に、占いで大体確かめられる

# 2/16

- クエリ数の設定
- クエリサイズの設定
- 位置を推定しながら情報を収集する？
- 条件付き確率の更新？
    - P(T | Q)
        - 最初は一様分布

## スケジュール

~2/17 根本的な改善の模索
2/18 細かい改善
2/19 パラメータ調整

## 根本的な改善

- 制約一個増やしたときの解の差分更新はできないか？
- reanneal？
- 掃き出し法がやっぱり良いか、、？
- 得られた情報の有効活用が不十分か、、？
    - 調査数を決定するのに使っていない
- 最適化した時に、ありうる解の自由度を判定できないか？
- 収束したらOK？
    - 各マスの値を更新していく
- 隣接するものは近い値
- 周囲を集計して、0に近ければ無視して良い
- 分散ではなく、1っぽいところほど多くサンプリングする

- 特徴量
    - 重なりが問題なのか？
    - 類似したミノがある方が難しいのでは？
- フィードバックは占いだけ
    - 現在の解とどの程度差があるかを考える
- 解を複数持っておいて、尤度が高い解を出力する
- 入力だけから正確に予測ができるようになれば良いけど、、
- 1000個くらい解を持っておいて、判別ができるようなクエリを投げる
    - 尤度を計算しておく
    - 他と比べて尤度が高くなったら答えを投げる
- 簡単なケースでは、
    - 失敗が重たい
    - 過剰なクエリを使いたくない

- Q=800、N=10000、K=20でもO(NKQ)
- 序盤の解に答えが含まれているか確認する
- 1000itごとではなく、最小値を更新したら候補に追加する

- まず解を見つけるのが大変だが、、
- 焼きなまし以外の最適化手法はないか？
- 線形緩和？

## 解法

1. Q=base_query_countを計算する
2. q in [Q/2, Q/4, Q/4, Q/2, Q/2]について
    1. qを使って調査する
        - 調査しながら、必要であれば出力をする
    2. 最適化をして候補を選択する
    3. 必要であれば出力をする

- mが小さいデータだけで学習データを作る

```rust
answers: Vec<(f64, Vec<Vec<usize>>)>;
answers_set: HashSet<Vec<Vec<usize>>>,
```

- 確定を使う
- あまり伸びないでござる
- reannealingは試す価値あるか、、

- チューニング
    - クエリの大きさ
    - 焼きなましの温度
- 高速化は結構効く
- 答えるかどうかに、2番手の解との差を見る
- 前の候補を超えない限り答えない

# 2/18

- 高速化
    - calc_errorの割り算を削る
- チューニング
    - クエリの大きさ
    - 焼きなましの温度
    - ステップ数、ステップ幅

- 検証
    - 4点swapの追加
    - 2点swapの改善
- 情報収集の改善
    - ミノの形に応じて決定する
    - 差分があるような形を作る

- kはinput.nに応じて変動させた方が良いのか？

# 2/19

## チューニング

1. ステップ数
2. 調査周辺
3. 焼きなましの温度
4. クエリサイズ

- extremeの対策
- 保守的なやつも投げてみる

- 高速化
- create_queryの工夫
- f16
