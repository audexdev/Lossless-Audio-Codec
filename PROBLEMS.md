# 問題点一覧

本プロジェクトのコードベースにおける問題点を以下に挙げます。

### 1. テスト環境の不備 (Test Environment Issue)
- `tests/test_e2e.cpp` および `test_all.sh` が、リポジトリに含まれていない `assets/` ディレクトリ内のWAVファイルに依存しています。
- これにより、`lac_tests` がアセットの読み込みに失敗し、テストスイート全体がアボートします (`Failed to load source WAV`)。

### 2. 並行処理の設計上の問題 (Unbounded Thread Creation)
- `src/codec/lac/encoder.cpp` および `decoder.cpp` において、`std::async` と `std::launch::async` を使用してブロックごとに非同期タスクを生成しています。
- 長い音声ファイルの場合、数千〜数万のタスク（およびスレッド）が同時に生成される可能性があり、システムのリソース枯渇やクラッシュを引き起こす恐れがあります。
- スレッドプールや `std::execution::par` などの適切な並行処理モデルへの変更が推奨されます。

### 3. WAVファイル解析の脆弱性 (Fragile WAV Parsing)
- `src/io/wav_io.cpp` の `read_wav` 関数は、`fmt ` チャンクが常に16バイト以上であることを前提としています（16バイト未満の場合、バッファオーバーランのリスクがあります）。
- `data` チャンクが `fmt ` チャンクより前に存在する場合、読み込みに失敗します（WAVフォーマットとしては稀ですが有効な構造です）。
- エラーハンドリングが不十分で、読み込み失敗時に詳細な理由が不明な場合があります。

### 4. メモリ効率 (High Memory Usage)
- 音声ファイル全体をメモリ上の `std::vector` に読み込んでから処理を行うため、ファイルサイズの数倍のメモリを消費します。
- 大容量ファイルの処理やストリーミング処理に適していません。

### 5. ビルドシステムの設定 (Build System Configuration)
- `CMakeLists.txt` において `file(GLOB_RECURSE LAC_SOURCES src/*.cpp)` を使用していますが、これは CMake のベストプラクティスに反します（ファイルの追加・削除が検知されず、再ビルドが必要な場合に見逃される可能性があります）。
- テストのビルドオプション (`LAC_BUILD_TESTS`) がデフォルトで OFF になっています。

### 6. コード品質とメンテナンス性 (Code Quality)
- `src/utils/endian.hpp` や `src/utils/logger.cpp` が空であり、実装が欠けているか不要なファイルが残っています。
- `src/codec/lac/decoder.cpp` 内のパーティションサイズ計算において、特定の条件下（極端に小さなブロックサイズ等）でロジックの脆弱性が懸念されます（現在はガードされていますが、潜在的なバグの温床となり得ます）。
