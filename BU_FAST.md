# bu-fast

自然言語の要求からブラウザを操作し、原文に基づく回答と要求されたファイルを保存します。サブコマンドなしで通常の実行経路を使います。

## セットアップ

Python 3.11以上、uv、ChromeまたはChromiumが必要です。ブラウザはBrowser Useが自動検出します。このcheckoutで実行します。

```sh
uv sync
```

`.env.local` に `OPENAI_API_KEY` を設定します。認証情報はGitに含めません。親モデルは `gpt-6.1-sol`、ブラウザの行動選択はDecisionsの `gpt-6-luna` を使います。

## 実行

```sh
uv run bu-fast --url 'https://対象サイト/' --goal '要求全文'
```

既知の入力文字列は `--input` を繰り返して渡します。複数の要求項目を明示する場合、`conditions` に `id` と `requirement` を持つPurposePlan JSONを `--plan PATH` で渡します。`inputs` も指定できます。

`--task PATH` では `id`、`start_url`、`goal`、`exact_input_values`、`required_fields` を持つWorkflowTask JSONを指定できます。`required_fields` は `id` と `requirement` の配列です。`--task` と `--url`・`--goal`・`--plan`・`--input` は併用できません。

Pythonからは `from browser_use.fast import WorkflowTask, WorkflowSettings, run_workflow` で同じ実行経路を使えます。CLIとPythonの既定値は共通で、総期限は180秒、親要求は24回、Decisions要求は90回です。`--timeout`、`--parent-turns`、`--max-decisions` で変更できます。ブラウザの既定値はBrowser Use本体から引き継ぎ、画面表示は環境変数 `BROWSER_USE_HEADLESS`、未指定ならディスプレイの有無で決まります。`--visible` で表示、`--headless` で非表示を明示し、`--executable-path` でChromeの実行ファイルを指定できます。Pythonでは `WorkflowSettings(browser_profile=BrowserProfile(...))` で本体のブラウザ設定を渡せます。`keep_alive=True` なら終了後もブラウザを残します。`--env-file` の既定は `.env.local`、`--output-dir` の既定は `.bu-fast/workflows` です。

## 実行経路と保存結果

`run_workflow → LedgerExplorer → DecisionRuntime` が、探索、必要な入力の補完、保存済み原文の検索、復旧、引用検証、回答とファイルの保存を同じ予算内で実行します。ブラウザは専用に起動します。

stdoutのJSONに `status`、`directory`、`final_answer`、`error` を返します。`status=complete` は引用検証とファイル保存が済んだ完成状態で、終了コード0です。情報不足は `incomplete`、失敗は `error`、期限切れは `timeout`、中断は `interrupted` です。完成以外は終了コード2、中断は終了コード130です。

## 今開いているChromeのタブを使う

Chromeの `chrome://inspect/#remote-debugging` で接続を許可した後、`--connect-chrome` を付けます。システムChromeの `DevToolsActivePort` から接続先を検出します。Chromeが接続許可を求めた場合はChrome上で承認します。

```sh
uv run bu-fast --connect-chrome \
  --url 'https://librosa.org/doc/latest/api/index.html' \
  --goal '同じタブでCore functionalityを開き、見出しとURLを報告して'
```

接続時の `--url` は、操作対象の既存タブの現在のURLと完全一致させます。タブを再読み込みせず、現在のページ状態から操作を始めます。一致するタブがない場合や複数ある場合は操作せず終了します。同じURLのタブが複数ある場合は、エラーに表示される対象IDを `--tab-id ID` で指定します。

別のChromeやリモートブラウザには、`--connect-chrome` の代わりに `--cdp-url HTTP_OR_WEBSOCKET_URL` を指定できます。接続時には `--visible`、`--headless`、`--executable-path` を併用できません。接続先のブラウザとタブは終了後も残します。Pythonでは `WorkflowSettings(cdp_url=..., tab_id=...)` で同じ接続経路を使えます。

成果物の保存先には `result.json`、検証済み成果の `delivery.json` と `final.txt` があり、`outputs` に `results.json`、`report.md` と要求されたCSVなどを保存します。表とCSVは同じセルから生成します。情報が不足する項目はnullにし、理由を残して未完了として保存します。

実行用の画像・状態JSON・復旧記録は、OSの一時領域に作る `bu-fast-workflow-*` に保存します。成果物側の `result.json` の `runtime_directory` で場所を確認できます。中断時の状態確認に使えるよう、終了直後には削除しません。一時領域の清掃後はこれらの記録を参照できなくなります。画像のPNG保存と状態JSON内の画像は従来どおりです。

詳細ログは `--log-dir PATH` を指定した場合だけ、`PATH/<成果物の実行フォルダ名>/` に保存します。Pythonでは `WorkflowSettings(log_dir=Path(...))` で指定できます。対象は観測・判断の `journal.jsonl`、親モデル応答、ツール応答、HTTP計測、試行回答、原文・引用の証拠台帳です。保存先は成果物側の `result.json` の `log_directory` に記録します。ログを保存しなくても、探索・証拠検索・引用検証・復旧・要求予算の制限は同じ処理を使います。

引用検証は、そのURLで取得した原文と要求項目の対応を確認します。リンク先URLだけでは引用元と認めません。未検証の試行は詳細ログを指定した場合に `attempted-delivery-*.json` に保存し、最終回答には出しません。意味的な正しさを保証するものではありません。入力値と表示結果は観測状態ごとに保持します。操作結果が不明な場合は同じ操作を自動で再送せず、復旧でも期限・予算・元の要求を維持します。

## 回帰確認

```sh
uv run --no-sync pytest tests/fast -q -o addopts='' -o log_cli=false
```

テストは通常CLI、既存Chromeの接続先検出・タブ選択・再読み込み防止・終了後の保持、本体の探索・復旧・停止、証拠保持、引用、欠損値、表・CSV保存を確認します。過去の比較実装・評価データ・旧CLIは保持していません。
