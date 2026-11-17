# Review guide for the QA evaluation set

Open `review_50.csv` (Google Sheets or Excel) and fill one row at a time.
Read the question, the answer and the evidence; open the URL only if unsure.

`verdict`:

* **ok**: all of the following hold
  1. the question is natural and self-contained (names the place / dish / festival);
  2. the evidence alone answers it, without outside knowledge;
  3. the answer is correct and complete according to the evidence;
  4. the question has one clear answer (not "list some …" with many valid answers).
* **fix**: the item is useful but needs an edit. Write the corrected text in
  `corrected_question` and/or `corrected_answer`. Do not change the evidence.
* **drop**: unanswerable from the evidence, wrong, ambiguous, or pure trivia
  that no traveller would ask.

`note`: optional, a few words on why (e.g. "answer not in evidence", "trivia",
"too much copied wording").

Decide on the label before looking at what a model answered; the goal is to
judge the question set, not the system.
