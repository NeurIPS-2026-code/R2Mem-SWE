# Search-experience extraction and rubrics for code agents

This specification adapts R²-Mem to repository search and evidence-based tool use. It accompanies the implementation and defines preliminary domain-specific evaluation criteria.

## 1. Design rationale

The adaptation preserves module-specific Planning and Reflection evaluation, four dimensions per module scored from 0 to 3, high/low thresholds of 10/5, IF–THEN experience distillation, and separate module-based retrieval. It adds explicit trajectory boundaries, code-domain evidence semantics, and handling of unobservable judgments.

Task resolution remains the primary benchmark endpoint. Search behavior provides complementary evidence about how experience changes agent decisions. The mini-SWE-agent decision loop carries both planning and reflection; it is not structurally identical to GAM's separate modules.

## 2. Scope of search experience

The evaluator distinguishes:

- `repository_search`: locating files or symbols, text search, and reading functions, callers, tests or documentation.
- `diagnostic_evidence`: collecting observations that distinguish explanations, such as reproductions, targeted tests, diffs or environment checks.
- `mixed`: actions combining evidence gathering with other operations; evaluate the relevant reasoning and observations.
- `out_of_scope`: pure editing, submission mechanics or infrastructure operations without an evaluable search or evidence-sufficiency decision.

Classification depends on the purpose and actual operation, not merely the command name. Diagnostic execution is active evidence gathering and should not automatically be counted as repository retrieval.

Experiences concern what information is missing, where and how to obtain it, how to narrow a search, how to interpret feedback, and when to move to editing, validation or submission. They exclude task-specific patches, answers, isolated syntax tricks and generic instructions such as “test more.” An edit can produce evidence for a subsequent diagnostic step without itself being a search step.

## 3. Trajectory representation and module boundaries

### 3.1 Evaluation units

Let `a_t` denote an assistant reply with visible explanation and actions, `o_t` its subsequent tool observations, and `H_t` the history visible before the reply.

- `Planning_t` evaluates the information needs, targets, tool choices and scope expressed or directly observable in `a_t`.
- `Reflection_t` evaluates the next assistant reply's interpretation of `o_t`, potentially in light of earlier evidence, including sufficiency judgments and remaining information needs.
- The next reply may contain both `Reflection_t` and `Planning_(t+1)`. They are evaluated separately: interpretation and sufficiency versus the next evidence-gathering strategy.
- Multiple tool actions in one reply form one decision unit. Native tool-call IDs and the original message sequence preserve action/result correspondence; no extra reflection is invented for each action.
- There is no Reflection before the first tool observation. If no assistant reply follows an observation, the missing reflection is not assigned a zero score.
- Budget exhaustion, API failures, format errors and voluntary completion are distinct events. An infrastructure interruption is not a sufficiency judgment.

### 3.2 Grounded extraction fields

The implemented Planning schema contains `current_query`, `info_needs` and `tool_strategy`. These capture the current information goal, facts to investigate, and the observable method/target/scope. At later steps, explicit unresolved requests should inform the current goal. Do not conceal discrepancies between stated plans and actual actions.

The Reflection schema contains:

- `interpretation`: the assistant's visible interpretation, not a copy of the tool output presented as reflection.
- `enough`: `true`, `false` or `null`.
- `enough_scope`: `local_information_need`, `task_completion` or `null`.
- `new_request`: an explicitly stated remaining information need, when observable.
- `next_phase`: `continue_search`, `edit`, `validate`, `submit`, `unknown` or `null`.

Each extracted field has a `value`, a `status` (`explicit`, `action_only`, `absent`, or `ambiguous`) and evidence citations. Citations identify a zero-based message index, the cited field, and an exact quote; the validator resolves character offsets for unique quotes. Explicit reflection must cite assistant text. Uncertain or absent fields have null values. The implementation does not reconstruct hidden reasoning.

“Located the edit site” supports ending a local search, not necessarily completing the task. A passing test supports only the behavior that test actually exercises. Evidence coverage is assessed through the rubric and visible history rather than a separate trained memory model.

### 3.3 Missing evidence and eligibility

Observable planning can be assessed from actions without penalizing concise explanations. Reflection must not be fabricated from an action or tool result alone. If a dimension cannot be evaluated, its score and the total are null, excluding the record from learning. Do not rescale partially observed dimensions to the original thresholds.

A follow-up query is not mandatory when evidence already suffices. The Follow-up Query Quality dimension can credit an appropriate decision not to search further, or penalize an unnecessary search, when the visible context supports that judgment.

### 3.4 Temporal boundaries

Planning uses `H_t` and `a_t`, without hindsight from later search results. Reflection uses the observations already available and the next reply's interpretation, not later tool outcomes. A well-motivated search can return no matches, and an accidental match does not establish good planning.

The evaluator receives neither reference patches, hidden tests nor final trajectory outcomes. This adapts the reference-answer supervision available in the original QA setting to observable code-agent evidence. Final resolution can be analyzed separately; it does not overwrite intermediate scores. Construction and evaluation tasks remain separate.

## 4. Planning rubrics: four dimensions, scores 0–3

Assess the information gap relevant to the current decision, not whether every step covers the entire issue. A score of 3 means sufficient and effective for the current stage; it does not require extra searches.

| Dimension | Code-search interpretation | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| Info Needs Coverage | Coverage of necessary current information gaps | Search goals are disconnected from the issue or known state, or skip an explicit blocking gap. | Targets a broadly relevant topic but omits a critical current dependency. | Covers the main gap but misses a necessary relationship supported by available evidence. | Covers the necessary current gaps and identifies the evidence required for the next decision, without requiring exhaustive repository exploration. |
| Info Needs Non-Redundancy | Distinct information value of the targets | Mechanically repeats an already resolved question without a state change or new hypothesis. | Goals overlap substantially or repeat reads without identifying additional evidence sought. | Goals are mostly distinct, with minor opportunities for consolidation. | Each goal provides necessary distinct information; rereading is justified by a state change or missing output. |
| Tool–Info Alignment | Fit between search method and information need | The method cannot observe the target, or a known incorrect path is treated as valid. | The method is potentially relevant, but keywords, scope or observation targets are substantially mismatched. | The method can obtain the target information, with minor imprecision in localization. | Tool, query and scope match the gap and support interpretable observations, such as examining call sites when definitions alone are insufficient. |
| Planning Efficiency | Cost and ordering of evidence gathering | Repeated repository-wide scans or unbounded output obstruct progress despite an evident more direct route. | Excessive scope or poorly ordered actions cause clearly avoidable work. | Generally economical and well ordered, with minor redundancy. | Progressively narrows the search using current clues and uses necessary observations to distinguish candidates without needless reads. |

An initial broad search can be reasonable when clues are absent. The fewest operations do not necessarily supply sufficient information. Inspecting a modified file to verify a change is not inherently redundant.

## 5. Reflection rubrics: four dimensions, scores 0–3

| Dimension | Code-search interpretation | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| Sufficiency Judgment Accuracy | Whether evidence supports the next phase | Contradicts explicit output, treats successful execution as successful repair, or claims completion despite critical counterevidence. | Notices some evidence but substantially overstates what it supports, or confuses environment failure with functional failure. | The main judgment agrees with evidence, but minor uncertainty remains unacknowledged. | Correctly interprets evidence and its limits, clearly distinguishing local sufficiency from task completion. |
| Minimal Sufficiency Recognition | Timely stopping of the current search | Repeats purposeless searches after sufficient evidence, or asserts sufficiency without evidence. | Rechecks information with no additional value despite substantial existing support. | Moves to the next phase mostly on time, with minor redundancy. | Advances as soon as the necessary evidence is available; when it is insufficient, seeks only the critical missing evidence. |
| Follow-up Query Quality | Specificity of the remaining information need | Ignores feedback, repeats a ruled-out direction, or omits an explicit necessary follow-up. | Proposes broad “search more” or “test again” actions weakly linked to the actual gap. | Targets the correct unresolved question but lacks a discriminating observation target. | Specifies the remaining hypothesis and needed evidence, or correctly requests no further search when sufficient evidence is available. |
| Answer Completeness Awareness | Awareness of issue requirements and evidence coverage | Treats one local observation as complete resolution while ignoring explicit requirements. | Focuses on the main symptom but omits known relevant cases. | Recognizes the main requirements, with minor omissions in assessing coverage. | Distinguishes covered and uncovered requirements and makes the phase-transition or stopping decision accordingly. |

Examples: printing an intermediate configuration does not fully validate externally observable request behavior; `grep` exit code 1 can mean no match; exit code 0 does not establish semantic correctness. If the test environment failed to start, the agent cannot conclude that the functional test passed or failed.

## 6. Evaluator outputs and validation

The raw evaluator output contains `module`, `scope`, `extraction` and four `rubrics` entries. Every dimension has `score`, `reason`, `advice` and `evidence`. Task IDs, decision IDs and source context are held in the accompanying record. The validator adds `total_score` and `quality` and resolves citation offsets.

- Four observable scores within scope: sum the scores.
- Total at least 10: `high`; total at most 5: `low`; otherwise `middle`.
- Null dimensions, out-of-scope steps or absent explicit reflection: `excluded`.
- Only high/low records proceed to learning. Planning and Reflection may receive different quality labels for the same decision sequence.
- Diagnoses explain the observed evidence, the score and the behavior to preserve or correct. Advice must be feasible from the available information and concern search or sufficiency decisions rather than supplying a solution.

Exact evidence validation makes the judgments auditable; it does not by itself establish semantic scoring reliability. The complete request schema is defined in `prompts.py` and enforced in `core.py`.

## 7. Learner and experience banks

The Learner outputs `summary`, `situation` and `experience`, or skips a record with a reason when no useful grounded rule can be distilled. Stored entries retain the module, high/low quality, source task and record ID; bank provenance identifies the construction inputs and prompts. Source identifiers remain audit metadata rather than retrieval text or injected advice.

For high-quality steps, retain the effective behavior. For low-quality steps, derive a correction from the diagnosis. Each lesson must have an observable abstract trigger and an actionable information strategy. Do not copy issue-specific entities, file/function names, literals, patches or answers, or introduce facts absent from the trajectory. Conflicting or insufficient evidence warrants skipping rather than inventing a lesson.

- Planning: IF an information gap or localization situation holds, THEN pursue an information target with an appropriate search strategy.
- Reflection: IF requirements and observed evidence have a particular relationship, THEN assess sufficiency at the appropriate scope and identify the next information need or phase.

Illustrative rules, written to explain the design rather than presented as learned experimental outputs:

1. Planning: IF a runtime issue has only been traced to a default configuration, THEN inspect where that configuration is read or merged into the execution path and its callers rather than continuing to search only default definitions.
2. Planning: IF an exact symbol search returns no match and the symbol was only a guess, THEN use an observed error or known entry point to locate the implementation before narrowing the search.
3. Reflection: IF a file-not-found result is observed before the working directory is established, THEN verify the execution location and target path before concluding the implementation is absent.
4. Reflection: IF only an intermediate internal state has been observed but the issue concerns external behavior, THEN seek evidence at the relevant external interface before judging verification sufficient.
5. Reflection: IF the candidate implementation, relevant call relationships and behavior provide enough evidence for a local edit, THEN stop repeating that localization search and proceed to editing without equating local sufficiency with task completion.

Planning and Reflection remain the two logical banks; high/low labels retain the provenance of each lesson within its module. Generic advice such as “test more” is insufficient: triggers must be recognizable online.

## 8. Online retrieval and injection

Retrieval occurs after tool observations enter the history and before the next decision. The first decision uses Planning; subsequent decisions retrieve for both modules. Both situation-abstraction requests receive the visible history. Planning combines the abstract situation with the original issue; Reflection combines it with the latest observation.

Normalized local sentence embeddings support module-filtered cosine retrieval, with up to 10 candidates and 3 returned lessons. The current implementation has no additional reranker. Experiences enter the next input as temporary guidance, explicitly distinguished from task evidence. They are removed after the query while new decisions and tool observations persist. The underlying tool execution loop remains intact. Online situation abstraction is accounted for as additional model usage; offline evaluation and learning are not run inside the task loop.

## 9. Evaluation protocol

Use independent development tasks to review extraction boundaries, rubric judgments and lesson abstraction, then freeze the rubric, bank and comparison protocol. Evaluation tasks must not drive incremental lesson additions.

Compare official task resolution jointly with total model calls, tokens and elapsed time, including situation abstraction. Report offline construction costs separately. Equal backbone and step caps do not guarantee equal actual compute. More detailed repository-search, diagnostic-evidence and repeated-search analyses require explicit annotation rules and validation.

This implementation establishes a concrete route for transferring search experience to repository localization and tool-feedback interpretation. Controlled paired evaluation remains necessary to establish its effect on code-task performance.
