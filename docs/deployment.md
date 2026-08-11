# Vercel release guide

The repository uses two GitHub Actions workflows:

- **Quality** runs linting, tests, retrieval evaluations, and a package build on every push and pull
  request.
- **Release to Vercel** is manual. It repeats the quality checks, builds one immutable Vercel
  artifact, and deploys it to either preview or production.

Manual releases keep small development commits separate from deployment decisions. The production
GitHub environment can also require reviewer approval.

## One-time setup

1. Import the GitHub repository into Vercel and leave the project root as `.`.
2. Add the application variables from `.env.example` to both the Preview and Production
   environments in Vercel. Store real Azure keys only in Vercel; never commit `.env`.
3. Copy the project and team IDs from `.vercel/project.json` after linking the project locally, or
   from Vercel project settings.
4. Add these GitHub repository secrets:
   - `VERCEL_TOKEN`
   - `VERCEL_ORG_ID`
   - `VERCEL_PROJECT_ID`
5. Create GitHub environments named `preview` and `production`. Add a required reviewer to
   `production` if approval should be mandatory.

## Release

Open **Actions → Release to Vercel → Run workflow**, select the branch and choose:

- `preview` for a shareable test deployment;
- `production` only after the preview URL and health check pass.

The deployment URL is written to the workflow summary. Check `/health/services?deep=true` on that
URL before sharing it.

## Rollback

Vercel keeps previous immutable deployments. Roll back from the Vercel dashboard, or run:

```bash
vercel rollback
```

## Demo limitation

The answering flow is a good fit for Vercel's Python runtime. Chart-upload jobs currently use an
in-memory registry and local temporary files, so they are suitable for this take-home demo but not
for durable production ingestion: Vercel can recycle an instance or route polling to another one.
A real release should move job state to a durable store and processing to a queue or workflow while
keeping Azure AI Search as the shared knowledge base.
