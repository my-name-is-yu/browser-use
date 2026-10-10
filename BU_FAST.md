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
