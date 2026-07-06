# Homework: AI Orchestration with Kestra

## Question 1: Context Engineering

**Answer**: AI Copilot has access to current Kestra plugin documentation

## Question 2: RAG vs No RAG

### 1_chat_without_rag.yaml

```bash
INFO 2026-07-06T15:35:57.798812Z ❌ Response WITHOUT RAG (no retrieved context):
Kestra 1.1 introduced several exciting features that significantly enhanced its capabilities. Here are 5 major features with brief descriptions:

1.  **Event-Driven Flows (Triggers):** This was a major leap, allowing Kestra workflows to be triggered by external events rather than just schedules or manual execution. Users could define triggers based on file system changes (e.g., a new file appearing in an S3 bucket or a local directory), Kafka messages, or other incoming events. This transformed Kestra into a powerful tool for reactive and event-driven architectures.

2.  **Plugin Management UI:** Kestra 1.1 introduced a dedicated user interface for managing plugins. Previously, installing and updating plugins often involved manual configuration or file system manipulation. With the UI, users could easily browse available plugins, install them, update them, and manage their versions directly from the Kestra interface, significantly simplifying the extensibility of the platform.

3.  **Task UI Enhancements & Documentation:** The UI for individual tasks within flows received significant improvements. This included better visual representation of task properties, more intuitive input fields, and in-line documentation for tasks. This made it easier for users to understand what each task does, how to configure it, and provided quick access to relevant documentation without leaving the workflow editor.

4.  **Flow as a Task:** This feature allowed users to embed one Kestra flow as a task within another Kestra flow. This promoted modularity and reusability, enabling complex workflows to be broken down into smaller, manageable, and independently testable sub-flows. It's a powerful way to encapsulate logic and build sophisticated pipelines from reusable components.

5.  **Blueprint Generation & Sharing:** Kestra 1.1 introduced the concept of "Blueprints," which are essentially reusable flow templates. Users could now generate blueprints from existing flows and share them. This feature fostered collaboration and accelerated development by allowing teams to create, share, and reuse common workflow patterns, reducing boilerplate code and promoting best practices across projects.

🤔 Did you notice that this response seems to be:
- Incorrect?
- Vague/generic?
- Listing features that haven't been added in exactly this version but rather a long time ago?

👉 This is why context matters! Run `2_chat_with_rag.yaml` to see the accurate, context-grounded response.

```

### 2_chat_with_rag.yaml

```bash
INFO 2026-07-06T15:39:44.231386Z ✅ RAG Response (with retrieved context):
Kestra 1.1 introduced several major features. Here are at least 5 of them with brief descriptions:

1.  **New Filters**: The UI filters were completely redesigned to be more user-friendly and powerful, allowing for explicit filter options, single-click resets, saved filter combinations, and customizable table columns.
2.  **No-Code Dashboard Editor**: This feature allows users to create and edit dashboards using a no-code, multi-panel editor directly from the UI, similar to the existing no-code flow editor.
3.  **Multi-Agent AI Systems**: AI agents can now use other AI agents as tools, enabling the creation of sophisticated multi-agent orchestration workflows where a primary agent can delegate subtasks to specialized expert agents.
4.  **Fix with AI**: Kestra 1.1 provides AI-powered suggestions when tasks fail, helping users quickly diagnose and resolve issues by offering intelligent recommendations for fixes.
5.  **Human Task**: This Enterprise Edition feature introduces "human-in-the-loop" workflows. When an execution reaches a human task, it pauses until designated users or group members manually approve and resume it, perfect for validation steps.
6.  **Improved Air-Gapped Support**: The UI now fully adapts to run without internet connectivity, with blueprints fetching from the Kestra API, YouTube embeds hiding automatically, local font fallbacks, and internet-dependent features being hidden.
7.  **Dozens of New Plugins**: Kestra 1.1 added numerous new community-contributed plugins, expanding integrations across various categories such as Data & Database (e.g., Liquibase, dlt), SaaS & API (e.g., Airtable, Stripe, Odoo), Cloud & Infrastructure (e.g., Dataform, AWS CloudWatch), and AI Model Providers (e.g., Oracle Cloud Infrastructure GenAI, Cloudflare Workers AI).

🎉 Note that this response is detailed, accurate, and grounded in the actual release documentation. Compare this with the output from 1_chat_without_rag.yaml!

```

## Question 3: Token usage — short summary

### log_token_usage for short summary

```bash
INFO 2026-07-06T15:43:26.812306Z 📊 Token Usage Summary:

Multilingual Agent:
- Input tokens: 282
- Output tokens: 83
- Total tokens: 365

English Brevity Agent:
- Input tokens: 98
- Output tokens: 45
- Total tokens: 143

💡 Tip: Monitor token usage to understand costs and optimize prompts!

```

## Question 4: Token usage — long summary

### log_token_usage for long summary

```bash
INFO 2026-07-06T15:46:42.885280Z 📊 Token Usage Summary:

Multilingual Agent:
- Input tokens: 282
- Output tokens: 160
- Total tokens: 442

English Brevity Agent:
- Input tokens: 175
- Output tokens: 46
- Total tokens: 221

💡 Tip: Monitor token usage to understand costs and optimize prompts!


```

## Question 5: Modifying a flow

```bash
INFO 2026-07-06T15:49:44.493164Z 📊 Token Usage Summary:

Multilingual Agent:
- Input tokens: 282
- Output tokens: 160
- Total tokens: 442

English Brevity Agent:
- Input tokens: 175
- Output tokens: 88
- Total tokens: 263

💡 Tip: Monitor token usage to understand costs and optimize prompts!

```

## Question 6: Best Practices

**Answer**: Use traditional task-based workflows for predictability and auditability
