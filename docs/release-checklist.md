# Release Checklist

Use this checklist before tagging a LAC release. The format and API are still experimental, so release review should make compatibility and known risks explicit.

## Pre-Release

- Confirm the release commit is on `main`.
- Confirm GitHub Actions CI is green for the release commit.
- Run the local Debug test suite.
- Run the local Release build.
- Run `./build/lac_cli selftest`.
- Review open correctness, security, fuzzing, and release-readiness issues.

## Version Metadata

- Confirm `README.md` lists the intended current release.
- Confirm `CMakeLists.txt` uses the intended project version.
- Confirm public version headers match the release version once the public API is populated.
- Confirm the Git tag matches the documented release version.

## Format And API

- Confirm any `.lac` bitstream behavior changes are documented in `docs/format.md`.
- Confirm unsupported WAV/PCM behavior is documented.
- Confirm public API changes are documented or explicitly marked experimental.
- Confirm compatibility-breaking changes are called out in release notes.

## Security And Hardening

- Review parser, decoder, and CLI hardening issues fixed in the release.
- Confirm security-sensitive fixes do not expose private crash inputs or exploit details.
- Confirm malformed-input regression tests were added for fixed decoder/parser bugs where practical.
- Confirm known unresolved risk areas are tracked in public issues or private advisories as appropriate.

## Release Notes

Release notes should include:

- user-visible codec or CLI changes
- format or compatibility changes
- correctness fixes
- security or hardening fixes
- test, fuzzing, and CI improvements
- known limitations

## After Tagging

- Verify the GitHub release points to the intended commit.
- Verify the release page links to relevant documentation.
- Open follow-up issues for deferred release-readiness work.
