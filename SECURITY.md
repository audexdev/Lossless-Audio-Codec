# Security Policy

LAC is an experimental codec and parser for attacker-controlled audio/container input. Crashes, hangs, excessive memory allocation, and malformed-file decode behavior are security-relevant even when they occur in local CLI workflows.

## Supported Versions

Security fixes are currently targeted at:

- the `main` branch
- the latest tagged release

Older releases may not receive backports while the format and API remain experimental.

## Reporting a Vulnerability

Please do not publish exploitable crashes or proof-of-concept malformed files in a public issue first.

Preferred reporting path:

1. Use GitHub's private vulnerability reporting or security advisory flow for this repository when available.
2. If private reporting is unavailable, open a public issue with a high-level description only, without a minimized crashing file or exploit details.

Useful report details:

- LAC commit or release tag
- OS, compiler, and architecture
- command line used
- whether the issue affects WAV parsing, LAC container decoding, block decoding, or CLI file handling
- sanitizer output, if available
- minimized input, shared privately when possible

## Current Risk Areas

Known areas that deserve conservative treatment:

- WAV chunk parsing and declared-size handling
- LAC block table and total-sample validation
- block decoder metadata validation
- Rice and bitstream edge cases
- mid/side arithmetic on malformed or full-range inputs
- large-file memory pressure

## Scope

In scope:

- memory safety bugs
- unchecked allocation or denial of service from small malformed files
- parser hangs or unbounded decode work
- incorrect acceptance of malformed LAC/WAV structures
- crashes in supported CLI encode/decode paths

Out of scope:

- compression ratio disagreements
- unsupported WAV variants rejected cleanly
- behavior in locally modified builds that cannot be reproduced on `main`

## Response Expectations

This is a small maintainer-run project. The goal is to acknowledge valid reports, reproduce them, and land a fix or tracking issue as quickly as practical. Issues that require larger design work may be tracked publicly after exploit details are removed.
