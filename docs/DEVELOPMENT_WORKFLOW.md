# Development and verification workflow

This document records repository infrastructure policy. It does not define the finite-element formulation or its coding contract; those remain in `FORMULATION.tex` and `IMPLEMENTATION_SPEC.md`, respectively.

## Git identity and history

The repository-local default identity is the human maintainer, `Sourena Yadegari <srn.yad@gmail.com>`, so an ordinary manual `git commit` has the intended attribution.

An agent-created commit uses command-scoped Git configuration so that `Codex Implementation Agent <codex@local.invalid>` is both author and committer:

```bash
git -c user.name="Codex Implementation Agent" \
    -c user.email="codex@local.invalid" \
    commit ...
```

Commit identity reflects who performs the commit. Discussion depth is not encoded in authorship or trailers. Record important design rationale in the commit body or an appropriate documentation change. Do not add `Co-authored-by` or `Assisted-by` trailers by default.

The nine commits predating this policy are intentionally left unchanged while the repository remains unpublished and has no remote. Any future history rewrite requires explicit approval and must happen before publication if possible.

## Local-only configuration and secret hygiene

Tracked files must not contain credentials, private keys, tokens, passphrases, personal hostnames, IP addresses, usernames, absolute home-directory paths, or SSH destinations. These values belong in the user's SSH configuration, environment variables, or `.codex/project-local.md`, which is ignored by Git.

The local file may record:

- the absolute local repository and interpreter paths;
- a non-public SSH host alias;
- the remote workspace path;
- preferred remote concurrency limits.

Tracked automation must receive those values through explicit command-line options, environment variables, or the ignored local file. A committed example may describe field names but must use placeholders.

Before publishing or adding a public remote, scan the working tree and every Git ref for secrets and private machine identifiers. If a real secret has ever been committed, revoke or rotate it first and then sanitize history. Adding a pattern to `.gitignore` is not sufficient to remove historical content.

## Test-size policy

Codex runs short, bounded tests and verification directly. A useful initial threshold is approximately five minutes, adjusted when a command is known to be predictable and quiet.

For long, nonlinear, or open-ended calculations, Codex prepares a one-time script or exact command and the user launches it on the laptop or remote worker. The numerical process does not need an active model session. When it finishes, the user asks Codex to inspect the resulting summary and artifacts. Passive elapsed time consumes no model tokens, whereas polling and returning large logs do.

Keep full logs and large HDF5 artifacts on disk. Return only a concise status summary and relevant failure excerpts to the conversation. Independent cases may run concurrently, but each process must receive explicit BLAS/OpenMP thread limits to avoid oversubscribing the host.

Model switching remains a manual user action. Codex may recommend a less expensive model for routine inspection or a more capable model for difficult diagnosis, but no wrapper around the Codex application will be introduced at this stage.

## Deferred remote worker setup

The laptop remains the source of truth. The remote Ubuntu worker is an execution target, not a second bidirectionally synchronized development checkout.

Bootstrap and automation are intentionally deferred until the first long run that benefits from the worker. At that point:

1. Store the SSH destination behind an alias in the user's SSH configuration. Use an encrypted key through `ssh-agent`; do not copy the private key, passphrase, or SSH destination into the repository, and do not enable agent forwarding.
2. Record the alias and remote workspace only in `.codex/project-local.md`.
3. Reproduce the environment from a declarative environment specification and an exact Linux lockfile. Never copy a Conda environment directory between machines.
4. Use one-way laptop-to-worker synchronization. `rsync` may transfer a dirty development snapshot; committed acceptance runs should prefer an explicit Git revision or temporary ref.
5. Return only run summaries and selected artifacts to the laptop. Do not synchronize remote source changes back into the development tree.

The first remote script should create a unique run directory and record the command, Git revision, dirty-tree fingerprint, environment-lock hash, host and package information, elapsed time, exit status, concise summary, full log, and requested output artifacts.

## Public execution boundary

The personal remote worker may run trusted branches and manually launched jobs. It must not execute arbitrary code from public pull requests. Public CI should use hosted runners or disposable, isolated workers with narrowly scoped credentials.

Deeper Codex automation through an SDK or App Server is deferred. Reconsider it only if recurring unattended jobs require callbacks, retries, scheduling, automatic model routing, or coordination across multiple repositories or workers.
