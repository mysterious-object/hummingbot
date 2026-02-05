<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { backendStatus } from "../lib/stores/status";
  import { getHealth, type HealthStatus } from "../lib/api/health";
  import { getChimeraBotHealth, importStrategy, startChimeraBot, stopChimeraBot } from "../lib/api/chimerabot";
  import { getStrategies } from "../lib/api/strategies";
  import { getLogs } from "../lib/api/logs";
  import { getEvents } from "../lib/api/events";
  import { connectRealtime } from "../lib/api/realtime";
  import { loadFromStorage, saveToStorage } from "../lib/storage";
  import type { Strategy, BotEvent } from "../lib/types";

  let health: HealthStatus | null = null;
  let hbHealth:
    | {
        connected: boolean;
        rpc_ready?: boolean;
        instance_id: string | null;
        error?: string | null;
        runtime?: {
          running: boolean;
          pid: number | null;
          started_at: number | null;
          exit_code: number | null;
          last_error: string | null;
        } | null;
      }
    | null = null;
  let strategies: Strategy[] = [];
  let logs: BotEvent[] = loadFromStorage("hb.logs", [] as BotEvent[]);
  let events: BotEvent[] = loadFromStorage("hb.events", [] as BotEvent[]);
  let selectedStrategy = loadFromStorage("hb.selectedStrategy", "");
  let actionMessage = "";
  let loading = true;
  let error = "";
  let wsConnected = false;
  let refreshTimer: number | null = null;
  let socket: WebSocket | null = null;

  const statusLabel = (value: boolean | undefined) => (value ? "Online" : "Offline");
  const hbStatusLabel = () => {
    if (hbHealth?.connected) return "Online";
    if (hbHealth?.runtime?.running) return "Running (Headless)";
    return "Offline";
  };

  const refresh = async () => {
    error = "";
    try {
      health = await getHealth();
      hbHealth = await getChimeraBotHealth();
      strategies = await getStrategies();
      logs = await getLogs(80);
      events = await getEvents();
      if (strategies.length > 0 && !selectedStrategy) {
        selectedStrategy = strategies[0].strategy_id;
      }
      saveToStorage("hb.logs", logs);
      saveToStorage("hb.events", events);
      saveToStorage("hb.selectedStrategy", selectedStrategy);
      backendStatus.set({
        state: health.mqtt_connected ? "connected" : health.mqtt_degraded ? "degraded" : "offline",
        message: health.mqtt_connected ? "Backend online" : "Backend degraded"
      });
    } catch (err) {
      error = err instanceof Error ? err.message : "Failed to load console data";
      backendStatus.set({ state: "offline", message: "Backend unreachable" });
    } finally {
      loading = false;
    }
  };

  onMount(async () => {
    await refresh();

    socket = connectRealtime((message) => {
      wsConnected = true;
      if (message.type !== "mqtt" || !message.topic) {
        return;
      }
      if (message.topic === "chimerabot/logs" && message.payload) {
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
          ].slice(0, 80);
          saveToStorage("hb.logs", logs);
        } catch {
          // ignore malformed payloads
        }
      }
      if (message.topic === "chimerabot/events" && message.payload) {
        try {
          const payload = JSON.parse(message.payload);
          events = [
            {
              event_id: payload.event_id ?? crypto.randomUUID(),
              type: payload.type ?? "event",
              severity: "info",
              message: payload.message ?? "Event",
              created_at: new Date().toISOString(),
              payload
            },
            ...events
          ].slice(0, 40);
          saveToStorage("hb.events", events);
        } catch {
          // ignore malformed payloads
        }
      }
    });

    socket.addEventListener("close", () => {
      wsConnected = false;
    });

    refreshTimer = window.setInterval(refresh, 8000);
  });

  onDestroy(() => {
    if (refreshTimer) {
      window.clearInterval(refreshTimer);
    }
    if (socket) {
      socket.close();
    }
  });

  const handleImport = async () => {
    if (!selectedStrategy) {
      actionMessage = "Select a strategy first.";
      return;
    }
    try {
      actionMessage = "Importing strategy...";
      await importStrategy(selectedStrategy);
      actionMessage = "Strategy imported.";
    } catch (err) {
      actionMessage = err instanceof Error ? err.message : "Failed to import strategy.";
    }
  };

  const handleStart = async () => {
    try {
      actionMessage = "Starting ChimeraBot...";
      const result = await startChimeraBot();
      if (result.runtime) {
        actionMessage = result.runtime.running
          ? "Headless runtime started."
          : "Failed to start headless runtime.";
      } else {
        actionMessage = "Start command sent.";
      }
    } catch (err) {
      actionMessage = err instanceof Error ? err.message : "Failed to start.";
    }
  };

  const handleStop = async () => {
    try {
      actionMessage = "Stopping ChimeraBot...";
      const result = await stopChimeraBot();
      if (result.runtime) {
        actionMessage = result.runtime.running
          ? "Failed to stop headless runtime."
          : "Headless runtime stopped.";
      } else {
        actionMessage = "Stop command sent.";
      }
    } catch (err) {
      actionMessage = err instanceof Error ? err.message : "Failed to stop.";
    }
  };
</script>

<section class="console">
  {#if error}
    <div class="banner error">
      <strong>Backend error</strong>
      <span>{error}</span>
    </div>
  {/if}

  <div class="status-grid">
    <div class="status-card">
      <div class="label">Backend</div>
      <div class="value">{health ? statusLabel(health.status === "ok") : "Offline"}</div>
      <div class="meta">DB: {health?.database ?? "unknown"}</div>
    </div>
    <div class="status-card">
      <div class="label">MQTT</div>
      <div class="value">{health ? statusLabel(health.mqtt_connected) : "Offline"}</div>
      <div class="meta">{health?.mqtt_error ?? (health?.mqtt_degraded ? "Degraded" : "Healthy")}</div>
    </div>
    <div class="status-card">
      <div class="label">ChimeraBot</div>
      <div class="value">{hbHealth ? hbStatusLabel() : "Offline"}</div>
      <div class="meta">
        {#if hbHealth?.connected}
          Instance: {hbHealth?.instance_id ?? "n/a"}
        {:else if hbHealth?.runtime?.running}
          Headless PID: {hbHealth.runtime.pid ?? "n/a"}
        {:else}
          {hbHealth?.runtime?.last_error ?? hbHealth?.error ?? "n/a"}
        {/if}
      </div>
    </div>
    <div class="status-card">
      <div class="label">Realtime</div>
      <div class="value">{wsConnected ? "Connected" : "Disconnected"}</div>
      <div class="meta">WebSocket stream</div>
    </div>
  </div>

  <div class="control-grid">
    <div class="panel">
      <h2>Control Panel</h2>
      <p>Import strategies, start/stop the runtime, and monitor responses.</p>
      <div class="actions">
        <button class="primary" on:click={handleImport} disabled={!selectedStrategy || loading}>
          Import Strategy
        </button>
        <button class="ghost" on:click={handleStart} disabled={loading}>
          Start Bot
        </button>
        <button class="danger" on:click={handleStop} disabled={loading}>
          Stop Bot
        </button>
      </div>
      {#if actionMessage}
        <div class="note">{actionMessage}</div>
      {/if}
    </div>

    <div class="panel">
      <h2>Strategies</h2>
      <p>Choose a config or script to import into ChimeraBot.</p>
      {#if loading}
        <div class="empty">Loading strategies...</div>
      {:else if strategies.length === 0}
        <div class="empty">No strategies found in conf/strategies or conf/scripts.</div>
      {:else}
        <div class="strategy-list">
          {#each strategies as strategy}
            <button
              class="strategy-item {selectedStrategy === strategy.strategy_id ? 'active' : ''}"
              on:click={() => {
                selectedStrategy = strategy.strategy_id;
                saveToStorage("hb.selectedStrategy", selectedStrategy);
              }}
            >
              <div>
                <div class="strategy-name">{strategy.name}</div>
                <div class="strategy-meta">{strategy.type}</div>
              </div>
              <span>Use</span>
            </button>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <div class="stream-grid">
    <div class="panel">
      <h2>Live Logs</h2>
      <p>Latest log output from the running instance.</p>
      {#if loading}
        <div class="empty">Loading logs...</div>
      {:else if logs.length === 0}
        <div class="empty">No logs yet.</div>
      {:else}
        <div class="stream">
          {#each logs as log}
            <div class="stream-row">
              <span class="tag">{log.severity}</span>
              <span class="text">{log.message}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <div class="panel">
      <h2>Events</h2>
      <p>Runtime events and notifications.</p>
      {#if loading}
        <div class="empty">Loading events...</div>
      {:else if events.length === 0}
        <div class="empty">No events yet.</div>
      {:else}
        <div class="stream">
          {#each events as event}
            <div class="stream-row">
              <span class="tag">{event.type}</span>
              <span class="text">{event.message}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</section>

<style>
  .console {
    display: grid;
    gap: 1.8rem;
  }

  .banner {
    padding: 0.9rem 1.2rem;
    border-radius: 14px;
    background: rgba(2, 6, 23, 0.7);
    border: 1px solid rgba(248, 113, 113, 0.5);
    color: #fecaca;
    display: flex;
    gap: 0.6rem;
  }

  .status-grid {
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  }

  .status-card {
    padding: 1rem 1.2rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.2);
  }

  .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
  }

  .value {
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0.4rem 0;
  }

  .meta {
    color: #94a3b8;
    font-size: 0.85rem;
  }

  .control-grid {
    display: grid;
    gap: 1.5rem;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }

  .panel {
    padding: 1.4rem;
    border-radius: 18px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.2);
    display: grid;
    gap: 0.8rem;
  }

  .panel h2 {
    margin: 0;
    font-size: 1.1rem;
  }

  .panel p {
    margin: 0;
    color: #94a3b8;
    font-size: 0.85rem;
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
  }

  button {
    border: none;
    border-radius: 999px;
    padding: 0.55rem 1rem;
    cursor: pointer;
    font-weight: 600;
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .primary {
    background: linear-gradient(120deg, #10b981, #38bdf8);
    color: #020617;
  }

  .ghost {
    background: rgba(148, 163, 184, 0.1);
    color: #e2e8f0;
    border: 1px solid rgba(148, 163, 184, 0.3);
  }

  .danger {
    background: rgba(248, 113, 113, 0.15);
    color: #fecaca;
    border: 1px solid rgba(248, 113, 113, 0.4);
  }

  .note {
    color: #38bdf8;
    font-size: 0.85rem;
  }

  .strategy-list {
    display: grid;
    gap: 0.6rem;
  }

  .strategy-item {
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #e2e8f0;
  }

  .strategy-item.active {
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35);
  }

  .strategy-name {
    font-weight: 600;
  }

  .strategy-meta {
    font-size: 0.75rem;
    color: #94a3b8;
  }

  .stream-grid {
    display: grid;
    gap: 1.5rem;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }

  .stream {
    display: grid;
    gap: 0.5rem;
    max-height: 320px;
    overflow: auto;
  }

  .stream-row {
    display: grid;
    grid-template-columns: 90px 1fr;
    gap: 0.6rem;
    font-size: 0.85rem;
  }

  .tag {
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #38bdf8;
    font-size: 0.7rem;
  }

  .text {
    color: #cbd5f5;
  }

  .empty {
    color: #94a3b8;
  }
</style>
