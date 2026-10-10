# bu-fast

CodexがURL・自然言語の目標・既知の入力値を渡し、Decisionsが観測から次の操作を選びます。新しい入力文字列が必要なときは入力欄と現在の画面をCodexへ返します。Codexが文字列を渡すと同じタブ・履歴・予算で再開します。

```text
目標 → 観測 → Decisions → ネイティブ操作 → 観測
                ↓ 新しい入力が必要
              Codex → 入力値 → 再開
```

```sh
uv run bu-fast start --url 'https://対象サイト/' --goal '達成したいブラウザ操作' --input '既知の値'
uv run bu-fast resume .bu-fast/runs/RUN_ID/result.json --input 'Codexが補った値'
```

入口はstartとresumeだけです。Job・TaskContract・patch・全文details APIはありません。入出力はPydanticで検証します。Decisionsは既存の操作対象と値から選び、文字列・selector・コードは生成しません。入力不足以外のblocked・unknownは自動再開しません。

モデルはgpt-6-luna。CLIは.env.localの既存OPENAI_API_KEYを読みます。キーはログに保存しません。専用ブラウザを作成し、needs_inputとunknownでは保持します。DONEなどで終了すると閉じます。--visible、--executable-path、--timeout、--max-decisionsを指定できます。再開で時間と要求回数は補充しません。

stdoutは短い結果です。result_pathに状態、evidence_pathに時系列の観測・リンク・見出し・入力・操作receiptを保存します。DONEはモデルによるブラウザ作業の完了判断で、成果の正しさの独立検証ではありません。最終回答と出典付き成果はCodexが保存した観測から作成します。

ブラウザ操作はBrowser UseのCDP/Actorを使い、対象が変わった場合は再観測します。操作の効果が不明になった場合はunknownで止め、自動で同じ操作を送信しません。confidenceの共通閾値は設けません。候補が255を超えた場合は切り捨てずblockedにします。

プロセス中断後の保存runに未確定の操作receipt（started/unknown）が残る場合も、resumeを拒否します。

実サイト比較は `uv run python -m benchmarks.bu_fast.measure_goal_loop --output PATH`。API親controllerをCodexの代役として使う開発時の比較です。独立採点は操作達成と成果完成を分けます。以前の実装・測定原本は.bu-fast内に保持しています。

24試行の結果と条件は[計測報告](.bu-fast/evaluations/goal-loop-20261010-main/report.md)に保存しています。

## 実験の基準版

Gitの `bu-fast-baseline-v1` タグを基準にします。整理前のソースも直前のcheckpointコミットに残しています。既存のoriginはBrowser Use本体のリポジトリです。今回の履歴とタグはローカルに保存しています。

基準版には助言、試行間の経験メモ、並列推論を含めません。目標の粒度、Decisionsへの入力、操作候補、モデル、直近8操作、完了判断、停止条件は整理前と同じです。本文・見出し・リンクも成果の根拠として保存します。未使用の対象座標・操作URL・旧抽出設定を除去し、全状態の保存は要求開始・操作直前・操作receipt・終了時に絞っています。journalにはすべてのイベントを残します。

比較課題は `benchmarks/bu_fast/goal-tasks.json` に保存しています。過去に使用した開発課題で、未見評価ではありません。実行ログ、画像、過去の計測原本、ローカル環境ファイルはGit対象外です。独立採点には従来どおり外部BU Bench checkoutが必要です。

追加案ごとに基準タグからブランチを作ります。

```sh
git switch -c yu/experiment-advice bu-fast-baseline-v1
```

最初は「基準版」「基準版＋助言」「基準版＋経験メモ」「基準版＋並列処理」を個別に比較します。共通のブラウザ操作・証跡保存・採点条件を固定し、入力準備から成果完成までの時間、完成率、親とDecisionsの要求数・usage、返却量、補完操作を記録します。経験メモの作成と評価課題は分けます。有効な追加だけを後で組み合わせます。

整理後の実サイト速度は未計測です。以前の24試行をこの基準版の測定値として流用しません。

## 3案のレビュー後の共通計測

`benchmarks.bu_fast.measure_architectures` は別worktreeの基準版・助言・経験メモ・並列処理を同じ採点・モデル・総予算で実行します。並列処理には同じreaderを終了後に実行する逐次対照も設けます。過去の4開発課題×5条件×3回の60試行です。これはAPI親を使う開発比較で、実Codexや未見課題での有効性は別に確認します。

準備時はAPIもブラウザも呼ばず、各worktreeのimport先、Chrome、採点checkout、選択メモを確認し、コード・課題・メモのbytesと試行順をprotocolに固定します。親要求は準備・整合・助言・成果生成を合わせて12回、Decisionsは60回、成果生成を含む総期限は240秒です。並列親は最大4要求をこの12回から使います。操作停止と成果完成は別に採点し、親/DecisionsのHTTP要求・usage、総時間、操作時間、処理の重なった時間、停止理由と成果を保存します。メモ作成コストは別記し、不明値は0にしません。

並列と逐次ではキューの消費速度が違い、読まれる途中観測が変わる場合があります。要求数・採用根拠の差も確認し、速度差すべてを並列化効果とは扱いません。コードやメモを変えた場合は新しいprotocolを準備します。既存試行は上書き・自動再試行しません。

準備済みprotocolに対し、計測と独立採点を別々に実行します。

```sh
uv run --no-sync python -m benchmarks.bu_fast.measure_architectures --protocol /absolute/path/protocol.json --run
uv run --no-sync python -m benchmarks.bu_fast.measure_architectures --protocol /absolute/path/protocol.json --grade-only
```
