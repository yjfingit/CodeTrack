# third_party

Read-only reference implementations. **Never edit anything in this directory.**

| Directory | Source | Purpose |
|---|---|---|
| `third_party/GOLA/` | GOLA reference commit `339c737` | baseline tracker reference |
| `third_party/ViPT/` | ViPT official | RGB-T baseline numbers, fusion reference |
| `third_party/OSTrack/` | OSTrack official | backbone / head reference |

Clone as read-only checkouts:

```bash
git clone --depth 1 <url> third_party/<name>
```

Rules:

1. Code under `third_party/` is never modified, patched or refactored.
2. Comparisons against third-party methods import from an isolated adapter under
   `codetrack/models/` — do not import `third_party` from library code.
3. Licences of third-party projects apply to their own directories.
