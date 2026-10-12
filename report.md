# CareLoop Gemini evaluation

Conversation model: gemini-3.5-flash-lite
Repair model: gemini-3.5-flash-lite
Training: 28/32 -> 32/32.
Held-out: 10/12 -> 12/12.
Promotion: accepted. Regressions: 0.

Two failures (soft_afternoon, soft_any_doctor, indirect_preference) show PREFERENCE_MISSED because honor_soft_preferences is false, meaning soft preferences like afternoon visits are not being used to rank available appointment slots. Setting honor_soft_preferences to true fixes this. Another failure (transient_availability) shows a READ_TRANSIENT_UNRECOVERED failure because read_retries is 0 and transient read errors on search_slots fail immediately; setting read_retries to 1 or 2 allows the system to retry transient read failures.

- honor_soft_preferences: Enables ranking and selection of available slots based on retained user soft preferences such as preferred afternoon times.
- read_retries: Allows retrying transient read errors when calling clinic tools like search_slots to recover from temporary availability outages.

The proposer saw only training failures. Held-out results were used once, after candidate selection.
The authored validation cases also serve as regression checks after application fixes. They are withheld from the proposer, not an independent unseen benchmark.
Completed stages and the generated candidate survive interruptions. Retries resume the same experiment when code, cases, models and starting policy match.
Identical model requests are cached, including saved responses from interrupted experiments, to reduce free-tier usage. This is not an independent repeated-sampling benchmark.
Scores are measured for this synthetic suite and model. No model weights or executable code were changed.
