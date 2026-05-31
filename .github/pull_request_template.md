## Summary

<!-- Describe the change and why it is needed. -->

## Correctness

- [ ] Roundtrip behavior is preserved for supported PCM WAV inputs.
- [ ] Codec, predictor, residual, stereo, or format behavior changes have targeted tests.
- [ ] Bitstream behavior changes are reflected in `docs/format.md`.
- [ ] Unsupported or out-of-scope behavior is documented or rejected explicitly.

## Security

- [ ] Malformed input, declared sizes, and allocation behavior were considered.
- [ ] Parser/decoder/CLI changes fail safely on invalid input.
- [ ] Security-sensitive details are kept out of public comments when private reporting is appropriate.

## Fuzzing Impact

- [ ] The change does not reduce fuzzability of parser or decoder surfaces.
- [ ] New parser/decoder edge cases have regression coverage or a fuzzing follow-up issue.
- [ ] Sanitizer behavior was considered for memory-safety-sensitive changes.

## Release Impact

- [ ] Version, release notes, public API, and documentation impact were considered.
- [ ] Compatibility changes are called out clearly.
- [ ] No generated files, local fixtures, editor caches, or build artifacts are committed.

## CI

- [ ] Relevant local build/test commands were run.
- [ ] GitHub Actions is expected to pass.

## Notes

<!-- Add reviewer context, limitations, or follow-up issue links. -->
