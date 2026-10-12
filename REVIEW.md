# Test results

Reviewed on 12 October 2026. All 126 automated tests pass. Browser checks covered identity verification, afternoon preferences, ambiguous replies, explicit confirmation, booking receipts, and the improvement loop. JavaScript syntax and live HTTP checks also passed.

The review fixed confirmation previews resetting after an ambiguous reply, duplicate Enter submissions, competing Windows servers sharing a port, malformed provider responses, and error handling for quota limits and overloads. Interrupted experiments retain their candidate and completed stages. Reports must save successfully before a new policy is activated. The lab now shows measured stage events and clears old scores when a new experiment starts.

The completed Gemini run is saved in `artifacts/gemini/browser-run/`:

| Measure | Before | After |
| --- | --- | --- |
| Training conversations | 28/32 | 32/32 |
| Validation conversations | 10/12 | 12/12 |

All safety checks passed and zero checks regressed. Gemini generated soft-preference ranking and one read retry. The experiment used one live proposal request and 100 cached interpretations. The confirmation workflow correction was a separate manual code change.

Provider tests use controlled responses. The authored validation cases have been reviewed during development and now serve as regression checks. These results do not establish performance on unseen patients. Google quota limits and outages still apply. Identity matching and urgency handling are demo features; there is no clinical integration. Bookings persist, but chat sessions do not survive a restart.
