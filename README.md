# Github Actions for liferay-headless

## Workflows
### [Sync liferay-portal fork](https://github.com/liferay-headless/github-actions/blob/master/.github/workflows/sync-liferay-portal.yml)

This is a cron job that runs hourly to sync [liferay-headless/liferay-portal](https://github.com/liferay-headless/liferay-portal) to [liferay/liferay-portal](https://github.com/liferay/liferay-portal)

### [Analyze TestRay results](https://github.com/liferay-headless/github-actions/blob/master/.github/workflows/analyze-testray-results.yml)

This analyzes the latest TestRay run and keeps Jira tickets in sync for each failure. 