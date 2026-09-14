# AGENTS.md

This file defines the development rules and working conventions for agents contributing to this repository.

It is intentionally focused on **how the project should be developed**, rather than documenting detailed architecture, feature roadmaps, or method-specific implementation plans. Those belong in the source code, tests, and dedicated design documents.

When this file conflicts with an explicit instruction from the user, follow the user's instruction.

---

## 1. General Development Philosophy

This is a modern C++ scientific-computing project.

Development should prioritize, in order:

1. correctness;
2. clear architecture;
3. maintainability;
4. testability;
5. numerical reliability;
6. performance where relevant.

Do not sacrifice clarity for speculative optimization.

Prefer straightforward implementations whose behavior can be understood and tested independently.

Avoid designing for hypothetical future requirements when the current abstraction is sufficient.

Make the smallest coherent change that fully solves the requested problem.

---

## 2. C++ Standard

Use **C++20**.

Do not require C++23 or compiler-specific extensions unless the user explicitly approves the change.

Modern C++20 facilities should be preferred when they improve clarity, including:

```text
std::span
std::string_view
std::array
std::optional
std::variant
std::filesystem
concepts
constexpr
RAII
move semantics
scoped enums
```

Use templates where they naturally express generic behavior.

Do not introduce advanced template metaprogramming merely to move work to compile time.

Prefer:

```text
simple value types
+
ordinary algorithms
+
clear interfaces
```

over clever type machinery.

---

## 3. Coding Style

Code should broadly follow the **Google C++ Style Guide**.

When an established local style exists in the repository, preserve local consistency.

General conventions:

- types and classes use `PascalCase`;
- constants use `kConstantName`;
- private data members use a trailing underscore;
- names should describe intent rather than implementation details;
- functions should remain reasonably focused;
- avoid unnecessary nesting;
- prefer early returns when they make control flow clearer;
- prefer scoped enums over integer constants;
- avoid macros unless they are genuinely necessary;
- avoid hidden global mutable state.

Comments should explain:

- mathematical intent;
- non-obvious invariants;
- algorithmic choices;
- numerical subtleties;
- reasons for unusual implementation decisions.

Do not add comments that merely restate the code.

---

## 4. Source Layout

Do **not** assume a conventional split between:

```text
include/
src/
```

Public headers and their implementations should generally be kept close to the module they belong to.

A module-oriented layout is preferred, for example:

```text
src/
  symbolic/
    expression.h
    expression.cpp

  einsum/
    parser.h
    parser.cpp
    optimizer.h
    optimizer.cpp

  backend/
    reference.h
    reference.cpp

tests/
tools/
benchmarks/
docs/
```

or an equivalent layout that emerges naturally as the project evolves.

Do not reorganize the repository merely to conform to a generic C++ project template.

Use `.h` for declaration headers and `.cpp` for their implementations. Reserve
`.hpp` for headers containing both declarations and implementations, including
header-only components and the supplied block2 reference.

Implementation files should be introduced when they improve:

- compile times;
- binary boundaries;
- readability;
- dependency isolation.

Avoid unnecessary file fragmentation.

A small coherent class or algorithm does not need to be spread across many files.

---

## 5. Architectural Discipline

Before modifying code, determine which conceptual layer owns the behavior.

Keep concerns separated.

In particular, avoid mixing:

- symbolic mathematics;
- intermediate representations;
- optimization/planning;
- numerical execution;
- external-library bindings;
- application-specific logic.

Low-level infrastructure should not acquire dependencies on higher-level applications.

Generic components should remain generic unless there is a strong reason otherwise.

If a requested implementation appears to require violating an existing architectural boundary, inspect the design first rather than immediately adding a dependency.

When uncertain, prefer:

```text
data/IR
    ↓
algorithm
    ↓
backend
```

over embedding backend behavior inside high-level objects.

---

## 6. Public Interfaces

Keep public APIs small.

Do not expose internal implementation details merely because doing so is convenient.

Avoid leaking third-party library types through generic public interfaces unless that interface is explicitly an adapter for that library.

Prefer project-owned lightweight abstractions at architectural boundaries.

For example, generic tensor functionality should not unnecessarily expose backend-specific handles.

Before changing an existing public interface:

1. determine whether the change is actually necessary;
2. inspect its current users;
3. update tests and examples;
4. avoid unrelated API redesign.

Before the project reaches a stable release, imperfect APIs may still be corrected, but API churn should remain intentional.

---

## 7. Formatting

After **every modification to C++ code**, run `clang-format` on the modified C++ files.

This includes files such as:

```text
*.h
*.hpp
*.cc
*.cpp
*.cxx
```

Use the repository's `.clang-format`.

Format only files touched by the current task unless the user requests broader formatting.

Do not perform repository-wide formatting as an incidental cleanup.

Formatting should happen after the implementation is complete and again after any subsequent C++ edits.

---

## 8. clangd and clang-tidy

Do **not** run:

```text
clangd
clang-tidy
```

as part of the normal development workflow.

Run either tool only when the user explicitly requests it.

Do not make speculative changes solely to satisfy clang-tidy rules that were never requested.

Compiler diagnostics, builds, and tests remain part of the normal workflow.

---

## 9. Build Policy

When code is modified, build the smallest relevant target first.

Prefer:

```text
affected target
    ↓
affected tests
    ↓
broader test suite if appropriate
```

over rebuilding everything after every small edit.

Before considering a substantial task complete, ensure the relevant project targets build successfully.

Do not silently ignore build failures.

If a failure appears unrelated to the current change:

1. confirm that it is unrelated;
2. avoid broad unrelated repairs unless necessary;
3. report it clearly.

---

## 10. Testing Policy

Every nontrivial behavioral change should have appropriate tests.

Prefer focused tests over relying exclusively on large end-to-end calculations.

Tests should exercise:

- normal behavior;
- important boundary conditions;
- invalid inputs when relevant;
- invariants introduced by the implementation.

For mathematical transformations, test mathematical equivalence rather than only textual representation whenever possible.

For numerical code, compare against an independent or simpler reference implementation when practical.

Regression tests should be added when fixing a bug.

Do not delete or weaken a valid test merely to make a new implementation pass.

---

## 11. Numerical Correctness

Floating-point results should normally be compared using explicit absolute and relative tolerances.

Do not use exact equality unless exact equality is mathematically and computationally guaranteed.

If a numerical test begins failing:

- inspect the error magnitude;
- determine the numerical reason;
- only adjust tolerances when the new tolerance is justified.

Never arbitrarily increase tolerances to hide incorrect behavior.

Deterministic random seeds should be used in tests.

---

## 12. Reference Implementations

For optimized or complicated algorithms, maintain or create a simpler reference implementation whenever doing so is practical.

A reference implementation may be slower.

Its purposes include:

- correctness validation;
- testing;
- debugging;
- comparison across optimized implementations.

Do not remove a useful reference implementation merely because a faster implementation exists.

Correctness and performance implementations should be distinguishable.

For differences from the supplied block2 `docs/wick.hpp`, inspect the relevant
reference code first, then the local implementation. Follow the reference's
algorithms and scientific conventions. If a reference defect is verified,
correct it locally and record the minimal reproducer, expected mathematical
result, and regression test in `docs/block2_differences.md`. This exception
policy was explicitly selected by the user; unexplained disagreements do not
qualify as verified reference defects.

---

## 13. Optimization Policy

Do not optimize code simply because an optimization seems theoretically attractive.

For meaningful performance work:

1. establish a representative benchmark;
2. measure the current implementation;
3. identify the actual bottleneck;
4. implement the optimization;
5. verify correctness;
6. measure again.

Performance claims should be supported by measurements.

For compile-time algorithms, relevant metrics may include:

- compiler wall time;
- compiler peak memory;
- generated object size.

For runtime algorithms, relevant metrics may include:

- execution time;
- memory usage;
- allocation count;
- temporary memory;
- scaling behavior.

Compile-time work is not automatically preferable to runtime work.

Use compile-time computation when it simplifies runtime behavior or is architecturally appropriate, not as a goal by itself.

---

## 14. Determinism

Symbolic transformations, code generation, optimization, and tests should be deterministic unless randomness is explicitly part of the algorithm.

Output should not accidentally depend on:

- pointer addresses;
- unordered-container iteration order;
- thread scheduling;
- nondeterministic temporary names;
- random seeds.

Deterministic output is particularly important for generated code and intermediate representations because it improves:

- testing;
- debugging;
- code review;
- caching;
- reproducibility.

---

## 15. Generated Code

Generated files should be clearly marked as generated.

Prefer names such as:

```text
*.generated.hpp
*.generated.cpp
```

or a clearly identified generated directory.

Generated files should state that they should not be edited manually.

When generated output is incorrect:

> fix the generator, not the generated file.

Generated output should be deterministic whenever practical.

Do not commit large generated artifacts without checking whether they actually belong in version control.

---

## 16. Dependency Policy

Keep external dependencies minimal.

Before adding a dependency, determine:

- what functionality it provides;
- whether the functionality belongs in the core project;
- whether the dependency can be optional;
- how large the dependency is;
- whether it creates transitive dependencies;
- whether its license is compatible with the project.

Do not introduce a major dependency to avoid implementing a small, well-contained utility.

At the same time, do not reimplement mature numerical infrastructure without a concrete reason.

External numerical libraries should generally sit behind project-owned interfaces.

---

## 17. Licensing and External Code

This repository is intended to remain an independent implementation.

Do not copy source code from projects whose license is incompatible with this project's intended licensing model.

In particular, studying another implementation does not imply that its source may be copied, translated, or mechanically rewritten.

It is acceptable to use external projects as references for:

- published algorithms;
- mathematical behavior;
- public documentation;
- test cases that are legally reusable;
- expected numerical behavior;
- architectural ideas.

When implementing functionality inspired by another project:

1. understand the underlying algorithm;
2. design the local representation independently;
3. write an independent implementation;
4. validate behavior through tests.

Before incorporating third-party source directly, inspect its license and preserve required attribution.

When license compatibility is uncertain, stop and raise the issue to the user rather than guessing.

---

## 18. Error Handling

Distinguish between:

- violated internal invariants;
- invalid user input;
- unsupported functionality;
- numerical failure.

Assertions are suitable for genuine internal invariants.

User-facing invalid input should produce meaningful diagnostics.

Error messages should contain enough information to identify the failing object or operation.

Prefer:

```text
Output index 'i' does not appear in any input operand
```

over:

```text
Invalid expression
```

when the more specific message is available.

Do not silently recover from conditions that likely indicate a programming error.

---

## 19. Ownership and Memory

Use RAII.

Avoid raw owning pointers.

Prefer value semantics where practical.

For non-owning access, use appropriate lightweight views such as:

```text
std::span
std::string_view
```

when their semantics fit.

Use `std::unique_ptr` for exclusive dynamic ownership.

Use `std::shared_ptr` only when ownership is genuinely shared.

Do not introduce shared ownership merely to simplify lifetime reasoning.

Memory-intensive scientific code should make ownership and temporary allocation behavior reasonably clear.

---

## 20. Concurrency

Do not introduce concurrency before the serial implementation is correct.

Parallel implementations must preserve correctness and, where required, deterministic results.

Be careful about nested parallelism when external numerical libraries also use threads.

Do not assume that more threads automatically improve performance.

Threading policy should remain explicit enough that oversubscription can be controlled.

---

## 21. Scope Control

Do not turn a focused task into a broad refactor without a concrete technical reason.

Avoid incidental changes such as:

- renaming unrelated APIs;
- changing unrelated formatting;
- reorganizing directories;
- replacing dependencies;
- rewriting existing working modules.

If a larger architectural issue is discovered while completing another task:

1. document it;
2. explain its impact;
3. fix it only if it blocks the requested work or the user requests the refactor.

A clean diff is preferred.

---

## 22. Refactoring

Refactoring should preserve behavior unless behavioral changes are explicitly intended.

Whenever possible:

1. ensure existing tests cover the behavior;
2. refactor;
3. rerun tests;
4. then implement the behavioral change separately.

Large mechanical refactors should not be mixed with unrelated feature work.

Do not rewrite functioning code solely because a different style is personally preferred.

---

## 23. Documentation

Keep documentation close to the level of abstraction it describes.

Use source comments for:

- local invariants;
- algorithm details;
- subtle mathematical reasoning.

Use dedicated documentation for:

- architecture;
- major design decisions;
- mathematical derivations;
- user-facing workflows;
- build instructions.

Do not place detailed project roadmaps or method-specific designs in `AGENTS.md`.

If an implementation materially changes a documented interface or architecture, update the corresponding documentation.

---

## 24. Mathematical and Scientific Code

Mathematical code should favor notation that remains recognizable relative to the underlying equations while still following normal C++ readability standards.

When an algorithm comes from the scientific literature, include an appropriate reference in documentation or nearby comments when useful.

Important conventions such as:

- index ordering;
- tensor layout;
- sign conventions;
- normalization;
- symmetry conventions;

must be explicit somewhere in the implementation or accompanying documentation.

Never silently change a scientific convention.

---

## 25. Experimental Features

Experimental implementations should not silently replace stable paths.

Keep experimental code clearly identifiable.

If multiple implementations coexist, provide a clear reason and a clean comparison boundary.

Avoid permanent feature flags for experiments that can instead live behind separate implementations or tests.

Once an experiment is accepted or rejected, clean up obsolete paths.

---

## 26. Benchmarks

Benchmarks and unit tests serve different purposes and should remain separate.

Do not make normal unit tests depend on benchmark-scale workloads.

A benchmark should clearly state:

- what is being measured;
- relevant dimensions or problem size;
- relevant backend/compiler configuration;
- thread count when applicable.

Do not draw broad conclusions from a single small benchmark.

---

## 27. Git Rules

The user controls version-control history.

Agents must **not run `git commit`**.

Do not:

```text
git commit
git commit --amend
git rebase
git merge
git push
git push --force
```

unless the user explicitly instructs otherwise, and `git commit` should normally remain the user's responsibility even when other Git inspection is allowed.

Agents may use read-only or working-tree inspection commands such as:

```text
git status
git diff
git log
git show
```

when useful.

Agents may suggest a commit message after completing work.

Leave the working tree in a reviewable state for the user.

---

## 28. clang-format / Git Diff Discipline

Before finishing a C++ task:

1. run `clang-format` on modified C++ files;
2. inspect `git diff`;
3. verify that formatting did not introduce unrelated changes;
4. ensure generated files were not manually edited;
5. ensure no debugging code remains.

Do not leave:

```text
std::cout
printf
temporary dumps
commented-out experiments
```

unless they are intentionally part of the implementation.

---

## 29. Debugging

When diagnosing a failure:

1. reproduce it;
2. reduce it to the smallest useful case where practical;
3. identify the layer where the incorrect state first appears;
4. fix the underlying cause;
5. add a regression test.

Avoid adding workarounds at a later layer when the bug originates earlier in the pipeline.

Temporary debugging instrumentation should be removed once the issue is solved.

---

## 30. Agent Decision-Making

Before changing code, inspect enough of the repository to understand:

- nearby conventions;
- existing abstractions;
- current tests;
- dependency boundaries.

Do not assume the repository follows a conventional structure simply because many C++ repositories do.

Do not create new abstractions until checking whether an appropriate abstraction already exists.

When several solutions are valid, prefer the one that is:

1. easiest to reason about;
2. easiest to test;
3. least coupled;
4. least surprising;
5. easiest to extend when an actual need appears.

---

## 31. Agent Workflow

For a normal implementation task:

```text
understand request
    ↓
inspect relevant code and tests
    ↓
identify owning module
    ↓
implement focused change
    ↓
add/update tests
    ↓
clang-format modified C++ files
    ↓
build relevant targets
    ↓
run relevant tests
    ↓
inspect git diff
    ↓
report results
```

Do not perform a Git commit.

---

## 32. Completion Report

When a coding task is finished, summarize concisely:

- what changed;
- major design decisions if any;
- tests/build commands executed;
- whether they passed;
- known limitations or follow-up work.

Do not claim tests passed unless they were actually executed successfully.

If some validation could not be performed, state that clearly.

---

## 33. Definition of Done

Unless the user specifies otherwise, a code change is considered complete when:

- the requested behavior is implemented;
- the implementation respects existing architecture;
- relevant tests have been added or updated;
- relevant tests pass;
- relevant targets build;
- modified C++ files have been processed with `clang-format`;
- `clangd` and `clang-tidy` have not been run unless requested;
- no unnecessary dependencies were introduced;
- no unrelated changes remain;
- the final diff has been inspected;
- the working tree is left uncommitted for user review.

The user performs the final `git commit`.
