# Test engineering workspace

The application opens **Test Engineering** directly. The former product-review
workflow has been removed. The original repository assessment is retained in
[architecture-assessment.md](architecture-assessment.md) as a historical reference.

## Start locally

```powershell
uv sync --locked --extra documents
Copy-Item .env.example .env  # only when you do not already have a .env
uv run streamlit run app.py --server.address 127.0.0.1
```

Keep existing `.env` credentials when migrating. The `documents` extra provides PDF,
DOCX and XLSX parsers. Text, Markdown, CSV and automation source ingestion work without
that extra. Python 3.14 is retained from the existing project.

The Hugging Face token and configured embedding endpoint must support
`google/embeddinggemma-300m`. Groq is the existing configured LLM provider. Use a
Groq model with structured output; image analysis additionally requires a model
supporting image input. The current INI model options are not a promise of vision support.

## Live workflow

1. Enter the Azure DevOps organization URL and PAT in the Streamlit connection form,
   then click Connect and validate access. The PAT input is masked and the form
   clears after submission. Azure URL/PAT environment values are not used for connecting.
   AZURE_DEVOPS_API_VERSION is an optional `.env` setting. Authentication and project reads are
   validated before project selection. Optional account discovery uses Microsoft's
   profile/account APIs; organization policy and PAT scope may deny that operation.
2. Select a project, team and sprint, then retrieve work items. Type names are editable
   for process templates and custom types. Search by ID/title and select a story.
   Current fields, relations, attachments and optionally discussion history are shown.
3. In Knowledge, sync project history, accessible Confluence Cloud pages, or local
   documentation/source uploads. Indexing is explicit; choosing a story never embeds
   the entire corpus. Repeat syncs detect versions/content changes and skip unchanged
   embeddings. Complete successful snapshots remove absent documents. Failed fetches
   leave prior indexed content intact and record failed status.
   Use Force re-index for deliberate rebuilds. Change ENGINEERING_EMBEDDING_VERSION
   when the embedding implementation changes, then synchronize every source before
   querying that project again.
4. Generate tests using current requirements and retrieved historical evidence.
   Inspect source metadata/content and relative ranking scores. FAISS distance is
   also included; the hybrid ranking score is not a probability or confidence measure.
5. Review/edit JSON fields, add/delete cases, or request regeneration with scenario
   instructions. Save creates an immutable revision; approval applies to that saved
   revision. Unknown sources, duplicate normalized titles, invalid step numbering,
   mismatched story IDs and uncovered AC references fail validation. Reference
   coverage is mechanical; humans still verify semantic coverage and assumptions.
6. Approve, load/select a plan and suite, or explicitly create a plan/static suite.
   Confirm displayed project/plan/suite and publish. A changed story revision blocks
   publication. Created IDs and statuses are displayed. Retry uses remote fingerprint
   tags and a local journal to recover from partial failures without recreating cases.
7. Upload screenshots and provide DOM/accessibility evidence and optional framework
   conventions. Generate an automation proposal, inspect/edit the files, and create
   a separate workspace. No generated application source is placed in the RAG core.
8. Approve the displayed code digest. Install fixed template dependencies, review
   the new package-lock digest, then approve again. Install Chromium, inspect the app,
   and use the inspection JSON/PNG as input to a refined proposal. Run TypeScript
   checking and Cucumber dry-run before tests. Execution requires another explicit
   action and passing validation for the same code digest. Reports include Cucumber
   JSON/HTML, screenshots and traces. Failure analysis proposes corrections only.
9. Framework evolution scans a repository for existing patterns and proposes complete
   code/tests/docs plus a unified diff. Approval creates an `ai-evolution/*` branch
   in a new worktree. Regression execution and commit/push/draft-PR creation have
   separate explicit controls. Main is never edited by that workflow.

The evolution runner currently supports Python unittest regressions and GitHub CLI
draft PRs. For another framework's test command or Azure Repos PRs, use the produced
diff/branch with that repository's normal review process.

## Offline demonstration and checks

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m examples.offline_demo --output .local/demo
uv run python -m compileall -q src tests examples
uv run python -m scripts.validate_automation_template
```

The demo uses mocked Azure/Confluence HTTP, deterministic embeddings and a deterministic
structured-output LLM substitute. It exercises the actual shared splitter, FAISS search,
SQLite sync/revisions, approval gates and publisher. `.local/demo/report.json` records
the result. Automation artifacts are deliberately incomplete review samples and are
not executed. This demo does not verify live service credentials, model quality,
application locators or an actual browser environment.

The template check installs the fixed Node dependencies with lifecycle scripts
disabled, checks the inspector's JavaScript syntax, compiles the trusted TypeScript
runner and dry-runs a fixed no-op Cucumber fixture. It does not launch a browser or
execute application scenarios. Use Node.js 22 or newer.

## Module map and extension points

| Module | Responsibility |
| --- | --- |
| `src/RAG/Rag.py` | Shared splitter and injectable embedding factory |
| `src/integrations/http.py` | Secret-contained HTTP with safe errors and no redirects |
| `src/integrations/azure_devops.py` | Discovery, WIQL, batches, comments, plans and suites |
| `src/integrations/confluence.py` | Confluence Cloud v2 pages/spaces and guarded pagination |
| `src/test_engineering/knowledge.py` | LangChain Document adapters, incremental sync, scoped hybrid retrieval |
| `src/test_engineering/models.py` | Structured test output schemas |
| `src/test_engineering/workflow.py` | LangChain generation, coverage checks, revisions and approval |
| `src/test_engineering/publishing.py` | Test steps XML, requirement links, suite membership and retry recovery |
| `src/automation/workspace.py` | Multimodal proposals, trusted runner template, digest approval and reports |
| `src/agents/framework_evolution.py` | Pattern comparison, diff review, approved worktrees/regressions/PRs |
| `src/ui/streamlitui/test_engineering.py` | End-to-end test engineering UI |

To add another knowledge connector, yield LangChain Documents with organization,
project, source type/document ID, title, version, timestamp and source link, then call
KnowledgeIndex.sync. Source grants must be supplied by deployment configuration.
Reuse `Rag.create_embeddings()` and `Rag.split_documents()`. Do not create a second
embedding/retrieval stack.

Historical source metadata explicitly labels evidence as historical information.
Current story requirements are passed separately to the engineering prompt. A new
business-rule verification workflow can introduce confirmed evidence; ingestion does
not silently certify historical requirements as current.

## Storage, security and deployment limits

This release targets a trusted local single-user deployment. Bind to loopback and
retain Streamlit's default XSRF protection. The Azure URL and PAT are supplied through
the session's Streamlit form. The PAT uses a password input and is passed only to
HTTP clients, never to workflow payloads or generation prompts. Confluence tokens
stay in HTTP clients/UI password state. Remote error bodies are discarded.
No PAT is written to `.env` or workflow persistence. Disconnect closes session clients
and clears engineering session state.
Reconnect after restart. Provider keys use environment or session inputs without exporting user keys
to process-wide environment variables.

Data is stored under ENGINEERING_DATA_DIR, namespaced by authenticated Azure identity
and organization/project. Deployment-controlled ENGINEERING_ALLOWED_SOURCES restricts
index access before vector search and restricts loading historical workflow context.
Confluence full sync is partitioned by site so one site's snapshot cannot delete
another site's content. Space-only sync is incremental; use an entire-site snapshot
to detect deletions across spaces.

Protect this directory with the operating system's user ACLs and encrypted storage.
SQLite contains business documents, drafts and retrieved excerpts. Application test
reports can contain screenshots and application data. The knowledge index persists
numeric vectors and documents rather than pickle files.

Generated code runs on the local machine after explicit review. It is not sandboxed.
The runner strips service/provider environment variables, but this does not isolate
the filesystem or network. Use a disposable controlled execution host without
production credentials for automation and evolution regression runs. Authentication
flows for the target application must be supplied through reviewed automation code;
the platform does not send browser login credentials to the LLM.

Before shared/hosted multi-user deployment, add an identity provider, server-owned
principal/source grants, per-document Confluence ACL refresh on reads, encrypted secret
vault, background job queue, isolated execution containers and storage encryption.
Live source permissions are refreshed by successful sync; cached documents can outlive
a per-page permission change between syncs. Project access is checked on every UI rerun.
This local implementation must not be presented as enterprise multi-user authorization.

Publishing is retry-safe within a single local store and reconciles remote fingerprint
tags. Concurrent independent deployments can still race because Azure work-item tags
are not a remote uniqueness constraint. Use a shared distributed publication lock
before scaling across instances. For exceptionally large corpora, replace per-query
FAISS reconstruction with a scoped persistent search service through this same adapter.

## API references

Connector request shapes follow Microsoft's [Suite Test Case Add](https://learn.microsoft.com/en-us/rest/api/azure/devops/testplan/suite-test-case/add?view=azure-devops-rest-7.1)
and Atlassian's [Confluence Cloud REST v2](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/).
Automation uses [Playwright TypeScript](https://playwright.dev/docs/test-typescript)
and [Cucumber JS](https://github.com/cucumber/cucumber-js/tree/main/docs).
