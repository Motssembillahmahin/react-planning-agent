# Vision

## What this project proves
- A ReAct loop can be written as an explicit state machine whose termination is
  a structural property, not a hope: every path through the loop consumes a
  bounded budget (iterations, critic rejections), and a property test checks
  this over hundreds of random model/tool behaviours.
- Each failure mode of agent loops (runaway iteration, token/time overrun,
  repeating the same action, spinning without new information, broken tools)
  has a named guard that can be switched off, so its contribution is measurable
  by ablation.
- A cheap, deterministic self-critique step (grounding the final answer in the
  observed evidence) measurably reduces confidently wrong answers and turns
  them into abstentions.
- Failure is a value: the caller always gets a structured result (status, best
  answer, guard, reason, trace) instead of an exception.

## What it does not prove
- That the agent is *smart*. The default evaluation uses a scripted model that
  replays fixed outputs; it measures the control loop, not reasoning quality.
- That the critic detects falsehood. It checks grounding in tool output, so a
  tool that returns plausible wrong data passes the critic.
- That results transfer to real LLMs. `make eval-real` exists, but its numbers
  are not part of this repo's claims.
- Anything about adversarial robustness, prompt injection, or safety of
  arbitrary tools.
