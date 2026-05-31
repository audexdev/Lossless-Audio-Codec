# Contributing

Thanks for taking the time to work on LAC. The project is experimental, but contributions should still be easy to review, reproduce, and maintain.

Please follow `CODE_OF_CONDUCT.md` in project discussions.

## Scope

LAC currently focuses on a compact C++20 lossless codec for supported PCM WAV inputs. Good contributions include:

- codec correctness fixes
- malformed-input hardening
- test coverage and fuzzing infrastructure
- documentation for the format, API, and maintenance workflow
- build, CI, and release hygiene
- targeted performance work with before/after notes

Large API or format changes should start as an issue so the compatibility impact can be discussed first.

## Development Setup

Requirements:

- CMake 3.16+
- a C++20 compiler
- pthread-compatible threading support

Configure and build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DLAC_BUILD_TESTS=ON
cmake --build build --parallel
```

Run tests:

```sh
ctest --test-dir build --output-on-failure
./build/lac_cli selftest
```

Generate a local compile database for clangd or other tools:

```sh
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json compile_commands.json
```

`compile_commands.json` and editor caches are local generated files and should not be committed.

## Pull Requests

Before opening a pull request:

- keep the change focused
- add or update tests for behavior changes
- update `docs/format.md` for bitstream changes
- update `SECURITY.md` or security notes for parser and decoder risk changes
- run the relevant local build and test commands
- explain correctness, security, fuzzing, release, and CI impact in the PR body

For parser, decoder, and CLI hardening work, include enough detail to reproduce the fixed behavior without attaching sensitive crash inputs publicly.

## Security Reports

Please follow `SECURITY.md` for crashes, hangs, memory safety issues, excessive allocation, and malformed-file handling bugs. Public issues should avoid exploit details when a private report is more appropriate.

## Maintainer Workflow

The project maintenance process is documented in `docs/maintainer-workflow.md`.
