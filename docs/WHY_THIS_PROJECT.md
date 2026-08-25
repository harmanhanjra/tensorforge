# WHY THIS PROJECT

Cycle 8 of the autonomous R&D engine. Niche: Neural Networks × Automatic Differentiation —
never touched by cycles 1–7 (security review, webhooks, onboarding, flaky tests, document
forensics, games, Redis clone). Difficulty 7, up from Cycle 3's 4.

## Why an autograd engine?
1. **Interview/job value**: Harman targets AI/ML/LLM/MLOps roles; "explain backprop" is
   table stakes, "have you built one" is differentiator. A working reverse-mode AD engine
   with broadcasting-correct reductions is exactly the depth interviewers probe.
2. **Real gap found in research**: educational engines (micrograd lineage) stop at scalars
   or ship unverified backward passes; HN threads (355-pt autodiff-from-scratch post) show
   sustained demand, and GitHub search showed nobody ships a numerical verification harness
   as a first-class CLI.
3. **Perfectly verifiable offline**: unlike API-dependent projects, correctness here is
   *provably* testable — analytic gradients must match finite differences to 1e-5. The
   anti-fabrication bar is structural, not procedural.

## The thesis
"The verification harness should be part of the engine." Every op in tensorforge is
permanently pinned against numerical truth by the same suite that tests it. Bugs found this
way during the build (accumulated-grad pollution across gradcheck calls, 1-D matmul
crashes, a silently wrong cross-entropy divisor) validate the design.
