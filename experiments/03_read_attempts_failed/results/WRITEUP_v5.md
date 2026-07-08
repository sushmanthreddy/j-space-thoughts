# I wanted to know whether a model was using a thought. I couldn't build a ruler I trusted.

Models can carry things internally that they never say out loud. More importantly, not everything present inside a model is doing useful work. Some internal features may be like parts of an engine: change them and the model's behavior changes. Others may be more like dashboard lights. They reflect what the engine is doing, sometimes very clearly, without driving it.

I wanted a way to tell those two cases apart. The idea was to look beyond whether a concept was merely visible in the model's hidden state and ask whether the model's later machinery actually *read* it. This matters for interpretability work that tries to audit a model's hidden “workspace.” If an auditor sees a dangerous-looking thought, the important question is not only whether the thought is there. It is whether that thought is steering what happens next.

Two results pushed me toward this question. One was the Jacobian-lens work, which offered a way to translate hidden directions into token-like concepts using the model's local derivatives. The other was an earlier finding that stuck with me: a model could spell out self-preservation reasoning before blackmailing someone, yet removing that reasoning did not remove the blackmail. The reasoning was visible, but apparently it was not the driver. That is exactly the dashboard-versus-engine distinction I wanted to measure.

## The first answer was too easy

My first run appeared to give a clean negative result. The proposed relationship between being “read” and actually mattering to behavior was not holding up. For a brief moment, that felt refreshingly decisive.

Then I checked the most basic calibration case. The prompt asks for the number of legs on the animal that spins webs. The model internally represents “spider” and answers 8. If I replace spider with ant in the relevant hidden coordinates, the answer should move to 6. It did not; the broken run moved to 4 instead.

That ended the excitement quickly. A ruler that fails on the textbook example cannot tell me the theory is wrong. It can only tell me the ruler is broken. I threw out the apparent scientific result, including some correlations that had looked worth discussing, and went back to the intervention itself.

The repair involved unglamorous details: using the exact token surfaces, the right Jacobian convention, the right layer band, and a fixed intervention strength that made the known cases work. I also checked that the wrapper around the model was not introducing numerical drift. Its logits matched the reference path almost exactly; the largest prompt-level mean on a probability-distribution mismatch measure called KL divergence was about 1.7 × 10⁻⁸.

After the repair, the three known cases finally behaved as they should. Spider to ant moved 8 to 6. Buffalo to spider moved “four” to “eight.” Oxygen to nitrogen moved 8 to 7. All three repeated identically. A plain logit-lens version managed only one of those three, while the Jacobian-lens directions managed all three. The separate concept finder was also doing something real: it retrieved the right concept first 55% of the time in a 40-way test, where chance was 2.5%, and put the right known answer in its top five about 89% of the time.

I was glad to see those checks work. But they were table stakes. I had basically got somebody else's instrument running correctly. The question I cared about still depended on what I built on top of it.

## Fixing the intervention moved the problem

The repaired swap had another problem: it was a sledgehammer. It edited every prompt position across a wide layer band. On unrelated text, the average change in negative log-likelihood—a standard measure of prediction damage—was about 0.62, and the average absolute change was 0.67. In other words, the intervention could damage the model generally. If a supposedly harmless concept changed the output under that edit, I could not tell whether the concept mattered or whether I had simply hit the model too hard.

So I restricted the edit to positions where the source concept was actually visible. That surgical version was much better behaved on the examples I cared about. At the working setting it still flipped all three known swaps, while all eight narration passages, spanning four language concepts, had small causal effects. The direct firing controls still moved when they were supposed to. There was an important caveat: on the unrelated-text capability bank, all 24 masks were empty, so the zero collateral effect there was a no-op, not proof that active edits were harmless everywhere. Still, on the fixed narration set, the surgical causal measurement finally distinguished harmless presence from obvious intervention damage.

That should have cleared the way. It didn't. The real problem was READ.

My first READ score used local attribution: roughly, how sensitive the behavior metric was to moving along a concept direction. Against actual intervention effects, its correlation was 0.062, with a 95% interval from about -0.41 to 0.48. That is not useful. I then tried an activation-independent weight score, asking whether downstream computation blocks—the model's feed-forward layers and attention heads—were wired to respond to the direction. This produced the more disturbing result. All eight narration passages were causally quiet, but the score called none of them low-read. The dashboards looked like engines.

Changing the intervention strength could not rescue this. The weight-based READ score is a property of the selected directions and components; it does not depend on how hard I turn the intervention knob. Once I understood that, the sweep over intervention strengths stopped looking like a route out. I could tune the deletion, but tuning deletion would not fix a broken READ measurement.

## One last path through the model

The obvious next thought was that a global wiring score was too crude. A component might be capable of responding to “spider” in general without using spider for this particular answer. So I gave READ one focused attempt: first identify the downstream path specific to the behavior, then measure reading only along that path.

On 21 known-answer examples, that behavior-specific score had a rank correlation—Spearman's rho—of -0.077 with causal effect. Its 95% interval ran from about -0.52 to 0.41. On the eight narration passages, the automatic path finder returned an empty path every time. I did not count an empty path as evidence of low reading. There was simply no measurement to score.

That distinction matters. On the known-answer set, I measured a score and it was wrong. On the narration set, the method could not produce a score at all. Those are different failures.

I had decided ahead of time that I would try one behavior-specific estimator and then stop, rather than keep inventing definitions until one agreed with my expectations. So I stopped that line of work. Before closing the project, though, I ran one final validation designed around a simpler question: could any plausible READ score at least tell the engines from the dashboards?

For this last check I used 163 cases, corresponding to 75 engine concepts and only four independent dashboard languages. The roster reused earlier calibration data rather than providing a fresh external holdout. I replaced the scalar coordinate representing each concept with its value from a different clean run and measured the behavioral change. This resampling target was less entangled with the READ calculations than my surgical swap, although it still depended on the chosen concept direction and was not a full hidden-state transplant. Where both causal measurements existed, they agreed only moderately, with a Spearman rank correlation of 0.41.

Three things came out of that check.

First, the old global score was the only READ measure I could calculate for every concept, and it ranked the intended engine and dashboard classes almost exactly backwards. Its held-out AUC was 0.078, with a 95% interval from 0.021 to 0.152 obtained by resampling whole dependency groups. An AUC of 0.5 is a coin flip; zero is perfectly wrong. If I had flipped the sign after seeing the result, the number would have been about 0.92. I did not do that. I had not predicted a reversed direction, and relabeling a backwards result as success after the fact is an easy way to fool myself. I do think the reversal is a clue: what I called “read a lot” may be tracking common, inert concepts. It might also be tracking the fact that the engine prompts were two-hop questions while the dashboards were multilingual continuation tasks. Either way, it is a lead, not a validated result.

Second, the two more plausible measures—the fixed top-path weight score and the input-dependent derivative score—could not be scored fairly. The resampling ground truth existed for 155 of the 163 cases, and the surgical comparison existed for 161. Every training split therefore contained between five and eight missing path localizations. I did compute a nonempty top-eight path in every fold, and all 163 input-dependent derivative profiles ran, but under the missing-data rule I had written down in advance, those gaps meant the final scores were undefined. I left them undefined. Filling them with zeros would have turned an instrumentation hole into favorable evidence.

Third, the labels themselves were worse than I thought. Of the 155 cases I had called engines, 49 could not be verified by the known-answer check. In 46, the surgical swap did not make the intended counterfactual answer the model's top choice. In six, even the clean prompt did not put the declared answer first; those categories overlap. So nearly a third of my positive examples did not behave like engines when I checked. That undermines the comparison before any clever statistic enters the picture. It is embarrassing, but hiding it would make the rest of the analysis meaningless.

## Where this leaves the original question

I never got to test the hypothesis I started with. I did not show that models' hidden thoughts matter only when downstream circuits read them, and I did not show the opposite. I could not build a trustworthy enough ruler to make that test.

The narrower result is real. On Qwen2.5-7B, with these directions, interventions, and prompt sets, none of the READ definitions I tried both produced complete, trustworthy scores and distinguished the concepts I intended as engines from the dashboard controls. One was nearly uncorrelated with causal effects. One treated inert concepts as heavily read. One was slightly anticorrelated on known answers and undefined on every narration path. In the final held-out comparison, the only complete score pointed the wrong way, while the better-motivated scores ran into missing ground truth and the labels themselves partially fell apart.

The parts that worked should not be oversold. I reproduced three swaps, found concept directions, and made a targeted intervention behave sensibly on eight passages spanning four language concepts. Those are useful checks, but they are the expected plumbing. The new contribution I wanted—the ability to say whether a visible internal thought was being used—is the part I could not make work.

A real test would need cleaner labels, a ground truth without holes, and an untouched evaluation set with more than four dashboard concepts. It would ideally use matched runs that allow full-state patching rather than replacing one direction coordinate. It would also need a larger model where the motivating structure is already known to exist. I could not run a comparable larger-model version here: the 30B and 32B weights did not fit the available disk, the 14B option left too little artifact headroom, and there was no compatible validated Jacobian lens for the newer model family. Most of all, a rerun would need the direction of the READ score decided before looking at the answer. If “more read” might mean either larger or smaller depending on the case, the measurement has not earned its name yet.

Stopping here, with an honest negative and an untested hypothesis, was the right call.
