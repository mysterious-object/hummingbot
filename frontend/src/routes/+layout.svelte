<script lang="ts">
  import { onMount } from "svelte";
  import { backendStatus } from "../lib/stores/status";
  import { getHealth } from "../lib/api/health";
  import { page } from "$app/stores";
  import { goto } from "$app/navigation";

  const navItems = [
    { label: "Command", path: "/" },
    { label: "Agents", path: "/agents" },
    { label: "Strategies", path: "/strategies" },
    { label: "Portfolio", path: "/portfolio" },
    { label: "Trades", path: "/trades" },
    { label: "Logs", path: "/logs" },
    { label: "Risk", path: "/risk" },
    { label: "Settings", path: "/settings" }
  ];

  const navigate = (path: string) => {
    goto(path);
  };

  export let data: Record<string, unknown>;

  onMount(async () => {
    try {
      const health = await getHealth();
      backendStatus.set({
        state: health.mqtt_connected ? "connected" : health.mqtt_degraded ? "degraded" : "offline",
        message: health.mqtt_connected ? "Backend online" : "Backend degraded"
      });
    } catch {
      backendStatus.set({ state: "offline", message: "Backend unreachable" });
    }
  });
</script>

<svelte:head>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link
    href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap"
    rel="stylesheet"
  />
</svelte:head>

<div class="app-shell">
  <aside class="app-rail">
    <div class="logo">CB</div>
    <nav class="nav">
      {#each navItems as item}
        <button
          class="nav-item { $page.url.pathname === item.path ? 'active' : '' }"
          on:click={() => navigate(item.path)}
        >
          {item.label}
        </button>
      {/each}
    </nav>
    <div class="rail-footer">v0.1</div>
  </aside>

  <div class="app-main">
    <header class="app-header">
      <div class="brand">
        <span class="brand-title">ChimeraBot</span>
        <span class="brand-sub">AI-Enhanced Trading Platform</span>
      </div>
      <div class="status">
        <span class="status-dot {$backendStatus.state}"></span>
        <span class="status-text">{ $backendStatus.message }</span>
      </div>
    </header>

    <slot />
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    font-family: "Space Grotesk", sans-serif;
    color: #e2e8f0;
    background: radial-gradient(circle at top left, #111827, #020617 45%, #020617 100%);
  }

  :global(*) {
    box-sizing: border-box;
  }

  .app-shell {
    min-height: 100vh;
    display: grid;
    grid-template-columns: 80px 1fr;
  }

  .app-rail {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.95), rgba(2, 6, 23, 0.95));
    border-right: 1px solid rgba(148, 163, 184, 0.2);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem 0;
    gap: 2rem;
  }

  .logo {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, #10b981, #38bdf8);
    color: #020617;
    font-weight: 700;
    font-family: "Fraunces", serif;
  }

  .nav {
    display: grid;
    gap: 0.75rem;
  }

  .nav-item {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    cursor: pointer;
  }

  .nav-item.active {
    color: #e2e8f0;
  }

  .rail-footer {
    margin-top: auto;
    font-size: 0.7rem;
    color: #64748b;
  }

  .app-main {
    display: flex;
    flex-direction: column;
    padding: 2rem 2.5rem 3rem;
    gap: 2rem;
  }

  .app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .brand {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .brand-title {
    font-size: 1.4rem;
    font-weight: 600;
  }

  .brand-sub {
    font-size: 0.85rem;
    color: #94a3b8;
  }

  .status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: #cbd5f5;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #64748b;
    box-shadow: 0 0 0 rgba(100, 116, 139, 0.6);
  }

  .status-dot.connected {
    background: #10b981;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
  }

  .status-dot.degraded {
    background: #fbbf24;
    box-shadow: 0 0 12px rgba(251, 191, 36, 0.6);
  }

  .status-dot.offline {
    background: #f87171;
    box-shadow: 0 0 12px rgba(248, 113, 113, 0.6);
  }

  @media (max-width: 900px) {
    .app-shell {
      grid-template-columns: 1fr;
    }

    .app-rail {
      flex-direction: row;
      justify-content: space-between;
      padding: 1rem 1.5rem;
    }

    .nav {
      grid-auto-flow: column;
      grid-template-columns: repeat(6, auto);
    }

    .nav-item {
      writing-mode: horizontal-tb;
      transform: none;
      font-size: 0.7rem;
    }
  }
</style>
