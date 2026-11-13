# GSM8K official test split

`test.jsonl` is a byte-for-byte copy of the official, non-Socratic GSM8K test
split from [openai/grade-school-math](https://github.com/openai/grade-school-math).
It contains 1,319 JSONL records, with `question` and `answer` fields.
The performance launcher sends only `question`; it does not grade answers.

- Upstream commit: `3101c7d5072418e28b9008a6636bde82a006892c`
- Downloaded: 2026-11-13
- [Original data](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl)
- Data bytes: `749738`
- Data SHA256: `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`
- [Original license](https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE)
- License SHA256: `86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b`

The original MIT license and copyright notice are preserved in `LICENSE`.
Do not correct spelling, normalize text, or otherwise edit the upstream data.

Paper: Cobbe et al., *Training Verifiers to Solve Math Word Problems* (2021),
[arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
