import os
import time
import pandas as pd
import requests

JIRA_URL = os.environ.get("JIRA_URL")
JIRA_PROJECT = os.environ.get("JIRA_PROJECT")
JIRA_USER = os.environ.get("JIRA_USER")
JIRA_PASSWORD = os.environ.get("JIRA_PASSWORD")
JIRA_COOKIE = os.environ.get("JIRA_COOKIE")

if not JIRA_URL or not JIRA_PROJECT:
    raise EnvironmentError(
        "Missing required environment variables: JIRA_URL, JIRA_PROJECT\n"
        "Authentication (pick one):\n"
        "  Basic Auth: export JIRA_USER='user' JIRA_PASSWORD='pass'\n"
        "  Cookie:     export JIRA_COOKIE='JSESSIONID=...'"
    )

if not JIRA_USER and not JIRA_COOKIE:
    raise EnvironmentError(
        "Set either JIRA_USER + JIRA_PASSWORD (recommended) or JIRA_COOKIE"
    )

JQL_QUERY = (
    f'project = {JIRA_PROJECT} AND statusCategory = Done '
    'AND "Story Points" is not EMPTY AND timespent is not EMPTY '
    'ORDER BY createdDate DESC'
)

headers = {"Accept": "application/json", "Content-Type": "application/json"}
auth = None

if JIRA_USER:
    auth = (JIRA_USER, JIRA_PASSWORD or "")
    print(f"Auth: Basic ({JIRA_USER})")
else:
    headers["Cookie"] = JIRA_COOKIE
    headers["X-Atlassian-Token"] = "no-check"
    print("Auth: Cookie")


def fetch_all_jira_issues():
    start_at = 0
    max_results = 1000
    all_issues = []

    fields_to_download = [
        "issuekey",
        "summary",
        "customfield_10011",
        "timespent",
        "description",
        "created",
        "customfield_12703",
        "customfield_11949",
        "customfield_12800",
    ]

    while True:
        print(f"Fetching batch: startAt={start_at}...")

        url = f"{JIRA_URL}/rest/api/2/search"
        payload = {
            "jql": JQL_QUERY,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": fields_to_download,
        }

        response = requests.post(url, json=payload, headers=headers, auth=auth)

        if response.status_code != 200:
            print(f"API error! Status: {response.status_code}, Body: {response.text}")
            break

        data = response.json()
        issues = data.get("issues", [])

        if not issues:
            break

        all_issues.extend(issues)
        print(f"Downloaded: {len(all_issues)} / {data['total']}")

        if len(all_issues) >= 100000:
            break

        start_at += max_results
        time.sleep(1)

    parsed_data = []
    for issue in all_issues:
        fields = issue.get("fields", {})
        parsed_data.append({
            "key": issue.get("key"),
            "summary": fields.get("summary", ""),
            "description": fields.get("description", ""),
            "time_spent": fields.get("timespent"),
            "estimate": fields.get("customfield_10011"),
            "region": fields.get("customfield_11949"),
            "subsystem": fields.get("customfield_12703"),
            "commitments": fields.get("customfield_12800"),
        })

    df = pd.DataFrame(parsed_data)
    df.to_csv("data/seed.csv", index=False)
    print(f"Done! Data saved to seed.csv. Total rows: {len(df)}")


if __name__ == "__main__":
    fetch_all_jira_issues()
