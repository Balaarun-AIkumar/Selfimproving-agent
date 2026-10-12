# CareLoop

A patient scheduling agent with Gemini, SQLite bookings, and a loop that learns prompt and policy changes from failed conversations.

## Run

Requires Python 3.10+. There are no runtime packages to install.

Copy `.env.example` to `.env`, then add your `GEMINI_API_KEY` from [Google AI Studio](https://aistudio.google.com/api-keys). Keep `.env` private.

```sh
python -m scheduler serve --gemini
```

Open http://127.0.0.1:8000. To use the offline rules instead, omit `--gemini`. Gemini requires internet access and available API quota.

Ask for general care next Monday with Dr Rao, preferably in the afternoon. Use the fictional patient **Maya Chen, DOB 1990-04-12**, choose a time, and confirm the booking. All times are IST.

## Evaluate and improve

```sh
python -m scheduler improve --gemini --fresh
python -m unittest discover -s tests -v
```

The first command evaluates the initial policy, asks Gemini for changes based on failures, replays the same cases, and checks a separate validation split. A change is accepted only if training improves without safety failures or regressed checks. Reports are saved to `artifacts/gemini/latest/`.

In the browser, open **Improvement lab**, choose **Initial policy comparison**, and click **Run improvement loop**. Saved interpretations can reduce quota use; `--fresh` on the CLI bypasses that cache and takes longer.

The recorded run improved training **28/32 → 32/32** and validation **10/12 → 12/12**, with zero regressed checks. It used one live Gemini repair request and 100 cached interpretations. Validation cases have been reviewed and now serve as regression checks. The loop changes prompts and bounded policy settings, not model weights or executable code.

[Demo](artifacts/demo/careloop-gemini-demo.mp4) · [Demo guide](docs/DEMO.md) · [Design note](DESIGN.md) · [Test results](docs/REVIEW.md) · [Recorded evidence](artifacts/gemini/browser-run/report.md)

This is a local demo with fictional clinic data. Bookings persist across restarts; chat sessions do not. `.env`, runtime databases, logs, and private working files are excluded from the submission ZIP.
