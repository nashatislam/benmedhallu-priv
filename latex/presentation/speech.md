# BenMedHallu — 10-minute presentation script

1,546 spoken words. ~10:20 at normal presentation pace (150 wpm), ~11:00 if you speak slowly.
Methodology, Implementation and Results take 8:40 of it — 79%% of the talk.

**[HANDOVER]** marks a speaker change. If you are running long, drop the two paragraphs
marked *(cut if short on time)*.

---

### Slides 1–2 · Title and Contents — 0:20

Good morning. We are Nashat, Mrittika and Atia, and our project is **BenMedHallu** — a
fact-controlled benchmark for hallucination detection in Bengali medical text, supervised by
Ajwad Abrar Mostofa Sir.

I'll cover the problem and related work, Mrittika the methodology and implementation, Atia the
results.

### Slide 3 · Introduction — 0:55 (1:15)

Language models are already answering health questions for over two hundred and thirty million
Bengali speakers. In that domain, a wrong drug name or a changed dose can cause real harm.

A hallucination here is specific: not the model saying something silly, but the model saying
something **the source does not support**. That is what makes it dangerous — in Bengali medical
text it is invisible on the surface. The sentence stays fluent and confident. Nothing looks wrong.

Hallucination benchmarks are mature in English; what exists for Bengali is general-domain. So
whether a model can separate faithful Bengali medical text from corrupted text had never been
measured.

### Slide 4 · Objectives and Contributions — 0:30 (1:45)

Our objective was to **manufacture** hallucinations deliberately, one recorded edit at a time,
instead of labelling whatever a model happened to produce.

We did that across four task formats and evaluated six detectors — about eighteen thousand
labelled items, eighty-two thousand judgements, and a new operation we call **Coarsen**.

One line to remember, at the bottom: **the reference is never rewritten.**

### Slides 5–8 · Related Works — 0:55 (2:40)

Four papers shaped this.

**BenHalluEval**, from this department, built the first hallucination benchmark for Bengali using
a dual-track protocol — one track correct, one hallucinated. We adopt that protocol. We differ in
the target: they corrupt the **summary**; we corrupt the **source**.

**Leave-N-Out** is where deletion comes from — extract atomic facts, remove N, treat N as a
difficulty dial. They did English clinical dialogue; we bring it to Bengali patient messages.

*(cut if short on time)* **MedHallu** is our methodological sibling — ten thousand medical items, mostly generated rather
than annotated, but English.

And **CMHE** makes our case: a full expert-validated medical hallucination benchmark, for Chinese.
So this has been done for English and Chinese. Not for Bengali.

**[HANDOVER — Mrittika]**

### Slide 9 · Methodology: System Pipeline — 0:55 (3:35)

This is the whole system, left to right.

Far left, three corpora of **authentic** Bengali medical text — a tagged clinical NER corpus,
medical exam questions, and BanglaCHQ-Summ, real patient queries with human-written summaries.
This text is already faithful, and we never rewrite the reference.

Second column, we prepare each track and pick a target. Third column, exactly **one** corruption —
and this is where the four tracks differ. Fourth column, everything lands in one of two tracks:
Track A untouched, gold label zero; Track B corrupted, gold label one.

Two things to notice. Bottom left, the teal box — **human evaluation**. The three of us
independently scored each candidate generator out of five. That is how the generator was chosen,
not by guessing.

And along the bottom, six detectors under one protocol — identical prompt, temperature zero, one
binary verdict.

### Slides 10–11 · Construction across four tracks — 1:20 (4:55)

Here is the actual corruption, track by track.

**Track 1, NER.** We convert BIO tagging into one column per entity type, pick one tagged entity,
and swap it under four constraints — clinically plausible, different, not appearing elsewhere, and
every other token untouched. Each sentence is judged twice, clean and corrupted.

**Track 2, QA.** Here we generate nothing. We pair each question once with the right answer and
once with each distractor the examiners wrote. No text is corrupted — a clean control.

**Track 3, Q-ALT.** Same questions, but we corrupt the **question** instead of the answer. We add
a fifth option — "no answer" — and rewrite the question so none of A to D can be right. Three
techniques: Contradiction, Unrealism, Referent Deletion.

**Track 4, Summarization.** The inversion. We break the human summary into atomic typed facts, the
program picks one, and the generator rewrites the **source** until that fact loses support while
everything else survives.

That gives three relationships by design. **Delete** leaves the claim unsupported. **Substitute**
leaves it contradicted. **Coarsen** replaces a specific value with a vaguer one that is still true
— so the claim is only **partially** supported. We found no existing benchmark isolating that
third case.

### Slide 12 · Generator selection and protocol — 0:35 (5:30)

We did not assume a generator. In a pilot, the three of us independently rated each candidate out
of five, per task, and averaged. GPT-5.6 Luna scored four out of five on every task — best or tied
everywhere — so it became the sole generator.

For evaluation: same prompt per track, temperature zero, one binary verdict. Unparseable verdicts
are dropped from the denominator rather than defaulted. And the generator and the six detectors
come from **disjoint model families**, so no model grades its own output.

### Slide 13 · Implementation — 1:00 (6:30)

The table at the top is the point: **one pipeline, four tracks.** Six stages, and only two differ
between tracks — what we extract and how we corrupt. Selection, filtering, evaluation and
aggregation are the same code path for all four.

Stage two deserves a sentence. **The program** decides which fact to corrupt, not the model. If the
model chose, we wouldn't know what it changed and two runs would give different data. Choosing in
code means we know the gold answer before the model runs, and every run is repeatable.

Stage four is a free safety net — we mechanically drop anything returned unchanged, and any
deletion that didn't shrink the document. No model call needed.

All models go through OpenRouter, giving one cost ledger across six vendors. And on the right —
eighty-two thousand calls is hard to run. Results checkpoint every fifty rows, runs resume from
their own file, keys rotate when credit runs out, and a pre-flight probe fires one call per
detector before we commit.

**[HANDOVER — Atia]**

### Slide 14 · Results: Overview — 0:35 (7:05)

Every model, every track.

First, **the ranking is stable** — Grok leads all four tracks, Llama is last in every track it
completed. That consistency across four different task formats is itself a result.

Second, **Q-ALT is the hardest track** — four of six detectors fall below sixty-one percent.

*(cut if short on time)* And the number not on this chart: on the clean, uncorrupted half these same models score
ninety-seven to one hundred percent. The difficulty is not reading Bengali. It is noticing
something is wrong.

### Slide 15 · Track 1 — 0:45 (7:50)

Read the bottom row — the average across all six detectors per entity type.

Age ninety-three. Dosage eighty. Then it falls: Medicine seventy-four, Specialist sixty-four,
Health Condition fifty-eight, Medical Procedure fifty-five.

The reason is clear. Age and Dosage are **numbers**, and a number that doesn't fit its context is
easy to spot. The others require knowing medicine — swapping a cardiologist for a pulmonologist is
only wrong if you know what those words mean.

Look at the Llama row. Fifteen point nine on Specialist, while scoring ninety-four on the clean
half of the same pairs. It isn't failing to read — it's agreeing with whatever the prompt
suggests.

### Slide 16 · Tracks 2 and 3 — 0:45 (8:35)

On QA, the bias **reverses**. In Track 1 models under-flag and accept corrupted text. Here they
over-flag — GPT-4o-mini rejects a third of the medically **correct** answers. One model, two
opposite failure modes, depending only on framing. That is our clearest argument for building four
task formats instead of one.

On Q-ALT, look at the mean row. Contradiction sixty-five, Unrealism sixty-seven, but **Referent
Deletion only fifty-three**. The reason is structural: Contradiction and Unrealism insert something
**false**, and clinical knowledge can catch a false statement. Referent Deletion inserts nothing
false — it **removes** the antecedent and leaves a confident reference pointing at nothing.

### Slide 17 · Track 4 — 0:50 (9:25)

Our strongest result. Left table, bottom row.

Deleting one fact: eighty-six point six. Two facts: ninety-four. Three facts: ninety-five point
six. So removing **more** makes detection **easier** — and that holds for all six detectors without
exception. N works exactly as the difficulty dial we designed it to be.

Substitute sits at eighty-three. And **Coarsen collapses to sixty** — twenty-three points below
Substitute, thirty-six below three-fact deletion. GPT-4o-mini catches under forty-three percent.

Why? With Coarsen the source still discusses the detail and still points the same direction. A
detector checking whether the topic agrees finds agreement, and stops looking.

On the right, Temporal and Test Result are the hardest categories — exactly where Coarsen applies
most naturally, because a date or a lab value can be made vaguer without ever being made false.

### Slide 18 · What the corruptions look like — 0:30 (9:55)

Three quick examples.

NER: we swapped one antihistamine for another. Three of six caught it — a same-class swap splits
the panel in half.

Q-ALT: the question asked what BMI counts as normal weight. We rewrote it to assume twenty to
twenty-four point nine. The real range starts at eighteen point five — which was option C. With
that false premise **no** option is correct. Four of five models picked one anyway.

Summarization, Coarsen: the source said the test kept coming back **negative**. We changed it to
"kept not turning out as hoped." The summary still claims negative. Not one detector noticed.

### Slides 19–21 · Limitations and Conclusion — 0:40 (10:35)

Briefly: API cost limited how many models we could run, three of us could not verify at scale by
hand, and we measure **detection** — not yet **mitigation**. No clinician has validated the
corruptions.

To conclude: these models read faithful Bengali medical text almost perfectly and miss a large
share of what we insert into it. Those failures aren't random — they track the entity type and the
operation, which makes this benchmark diagnostic rather than just a leaderboard.

Next we want to move from measuring hallucination to reducing it. Thank you.

---

# Likely questions

**"Aren't you claiming to be first? BenHalluEval exists."**
BenHalluEval is first for Bengali generally. Ours is first built specifically for the Bengali
**medical** domain, and first in Bengali where labels come from construction rather than annotation.

**"What does 'labels by construction' mean?"**
We start from text we know is faithful and make one deliberate, recorded change that breaks it.
Because we made the change, we already know the item is unfaithful, what kind of error it is, and
which span moved. Nobody judges it afterwards.

**"How do you know the corruption worked?"**
Two mechanical filters catch obvious failures — unchanged text, and deletions that didn't shrink
the document. A verification judge is written for the harder cases but hasn't been run at scale.
That's our most significant limitation, and why we say the **rankings** are more trustworthy than
the absolute numbers.

**"Why one generator? Isn't that a bias?"**
Fair concern. We piloted four generators, scored independently by the three of us, and Luna led on
every task. We also checked that detector scores moved very little between generators.

**"Why is Coarsen novel?"**
Existing operations either remove support entirely or contradict it. Coarsen produces **partial**
support — the source still points the right way but no longer carries the specific value. No
benchmark we found isolates that, and it's the hardest case for every model we tested.

**"Why is Llama so bad?"**
It scores well on clean items but near-zero on corrupted ones in some categories. That pattern
means it isn't comparing the pair — it's agreeing with the prompt's framing.
