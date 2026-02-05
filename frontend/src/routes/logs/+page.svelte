<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { getLogs } from "../../lib/api/logs";
  import { connectRealtime } from "../../lib/api/realtime";
  import { loadFromStorage, saveToStorage } from "../../lib/storage";
  import type { BotEvent } from "../../lib/types";

  let logs: BotEvent[] = loadFromStorage("hb.logs", [] as BotEvent[]);
  let loading = true;
  let error = "";
  let socket: WebSocket | null = null;

  onMount(async () => {
    try {
      logs = await getLogs(200);
      saveToStorage("hb.logs", logs);
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to load logs";
    } finally {
      loading = false;
    }

    socket = connectRealtime((message) => {
      if (message.topic !== "chimerabot/logs" || !message.payload) {
        return;
      }
      try {
        const payload = JSON.parse(message.payload);
        logs = [
          {
            event_id: payload.event_id ?? crypto.randomUUID(),
            type: "log",
            severity: (payload.level ?? "info").toLowerCase(),
            message: payload.message ?? "Log",
            created_at: new Date().toISOString(),
            payload
          },
          ...logs
        ].slice(0, 200);
        saveToStorage("hb.logs", logs);
      } catch {
        // ignore malformed payloads
      }
    });
  });

  onDestroy(() => {
    if (socket) {
      socket.close();
    }
  });
</script>

<section class="page">
  <header>
    <h1>Live Logs</h1>
    <p>Streaming logs emitted by the ChimeraBot runtime.</p>
  </header>

  <div class="panel">
    {#if error}
      <div class="empty">{error}</div>
    {/if}
    {#if loading}
      <div class="empty">Loading logs...</div>
    {:else if logs.length === 0}
      <div class="empty">No logs yet.</div>
    {:else}
      <div class="log-list">
        {#each logs as log}
          <div class="log-row">
            <span class="log-level">{log.severity}</span>
            <span class="log-message">{log.message}</span>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  .page {
    display: grid;
    gap: 1.5rem;
  }

  header h1 {
    margin: 0 0 0.5rem 0;
    font-size: 2rem;
  }

  header p {
    margin: 0;
    color: #94a3b8;
  }

  .panel {
    padding: 1.5rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.25);
  }

  .log-list {
    display: grid;
    gap: 0.6rem;
    max-height: 420px;
    overflow: auto;
  }

  .log-row {
    display: grid;
    grid-template-columns: 80px 1fr;
    gap: 0.8rem;
    font-size: 0.85rem;
  }

  .log-level {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #38bdf8;
  }

  .log-message {
    color: #cbd5f5;
  }

  .empty {
    color: #94a3b8;
  }
</style>
