# Demo

[Watch the edited recording](../artifacts/demo/careloop-gemini-demo.mp4). It combines the appointment walkthrough with the recorded improvement experiment. Idle time and unrelated windows have been removed; the evaluation results are unchanged.

To repeat it, run `python -m scheduler serve --gemini` and open http://127.0.0.1:8000.

1. Ask for general care next Monday with Dr Rao, preferably in the afternoon. Verify Maya Chen using DOB 1990-04-12 and choose a displayed slot.
2. At the booking preview, send "Sounds interesting.". The same appointment should remain pending. Send "yes" and show the committed receipt.
3. Open **Improvement lab**, select **Initial policy comparison**, and start the loop. Show the baseline failures, generated changes, replay, validation, and promotion decision in **Experiment log**.
4. Expand **soft afternoon** and compare the morning slot before the change with the afternoon slot after it. Finish on the scores and active-policy status.

The recorded experiment moved training from 28/32 to 32/32 and validation from 10/12 to 12/12, without regressed checks. It used one live Gemini repair request and 100 cached interpretations. This adapts prompts and policy settings; it does not retrain the model. The validation cases are regression checks after development review.

For submission, upload the MP4 to Google Drive, set **General access → Anyone with the link → Viewer**, and test the viewing link while signed out. Put that link in the form. Snipping Tool or OBS can record a replacement locally without a Loom subscription.
