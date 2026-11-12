# Software validation and evaluation scope

The package includes 24 offline unit/integration tests covering trajectory boundaries, split isolation, citation and score validation, experience construction/resumption, injection behavior, and baseline execution/grading support. All 24 passed with the pinned mini-SWE-agent source available, with no skips. Tests use synthetic inputs and mocks; they do not make model calls or execute benchmark tasks.

Python syntax, JSON templates and command-line entry points are checked. YAML examples are supplied as configuration templates and require the runtime dependencies described in the README. File checksums are provided in SHA256SUMS.

The release preserves the original four scoring dimensions per module, 10/5 thresholds, IF–THEN learning and separate module retrieval. The code-domain rubric is supplied in English and is loaded directly into the evaluator prompt. Its translation is therefore also a prompt revision: new evaluations should use fresh output directories and record the current prompt provenance.

This supplement documents an implemented transfer design and its software checks. It does not include completed paired code-task results. Domain review and calibration on independent development tasks should precede a frozen evaluation of task success and search efficiency.
