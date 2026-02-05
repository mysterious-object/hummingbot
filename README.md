![ChimeraBot](https://github.com/user-attachments/assets/3213d7f8-414b-4df8-8c1b-a0cd142a82d8)

----
[![License](https://img.shields.io/badge/License-Apache%202.0-informational.svg)](https://github.com/chimerabot/chimerabot/blob/master/LICENSE)
[![Twitter](https://img.shields.io/twitter/url?url=https://twitter.com/_chimerabot?style=social&label=_chimerabot)](https://twitter.com/_chimerabot)
[![Youtube](https://img.shields.io/youtube/channel/subscribers/UCxzzdEnDRbylLMWmaMjywOA)](https://www.youtube.com/@chimerabot)
[![Discord](https://img.shields.io/discord/530578568154054663?logo=discord&logoColor=white&style=flat-square)](https://discord.gg/chimerabot)

ChimeraBot is an open-source framework that helps you design and deploy automated trading strategies, or **bots**, that can run on many centralized or decentralized exchanges. Over the past year, ChimeraBot users have generated over $34 billion in trading volume across 140+ unique trading venues.

The ChimeraBot codebase is free and publicly available under the Apache 2.0 open-source license. Our mission is to **democratize high-frequency trading** by creating a global community of algorithmic traders and developers that share knowledge and contribute to the codebase.

## Quick Links

* [Website and Docs](https://chimerabot.org): Official ChimeraBot website and documentation
* [Installation](https://chimerabot.org/installation/docker/): Install ChimeraBot on various platforms
* [Discord](https://discord.gg/chimerabot): The main gathering spot for the global ChimeraBot community
* [YouTube](https://www.youtube.com/c/chimerabot): Videos that teach you how to get the most out of ChimeraBot
* [Twitter](https://twitter.com/_chimerabot): Get the latest announcements about ChimeraBot
* [Reported Volumes](https://p.datadoghq.com/sb/a96a744f5-a15479d77992ccba0d23aecfd4c87a52): Reported trading volumes across all ChimeraBot instances
* [Newsletter](https://chimerabot.substack.com): Get our newsletter whenever we ship a new release

## Getting Started

The easiest way to get started with ChimeraBot is using Docker:

* To install the Telegram Bot [Condor](https://github.com/chimerabot/condor), follow the instructions in the [ChimeraBot Docs](https://chimerabot.org/condor/installation/) site.

* To install the CLI-based ChimeraBot client, follow the instructions below.

Alternatively, if you are building new connectors/strategies or adding custom code, see the [Install from Source](https://chimerabot.org/client/installation/#source-installation) section in the documentation.

### Install ChimeraBot with Docker

Install [Docker Compose website](https://docs.docker.com/compose/install/).

Clone the repo and use the provided `docker-compose.yml` file:

```bash
# Clone the repository
git clone https://github.com/chimerabot/chimerabot.git
cd chimerabot

# Run Setup & Deploy
make setup
make deploy

# Attach to the running instance
docker attach chimerabot
```

### Install ChimeraBot + Gateway DEX Middleware

Gateway provides standardized connectors for interacting with automatic market maker (AMM) decentralized exchanges (DEXs) across different blockchain networks.

To run ChimeraBot with Gateway, clone the repo and answer `y` when prompted after running `make setup`

```yaml
# Clone the repository
git clone https://github.com/chimerabot/chimerabot.git
cd chimerabot
```
```bash
make setup

# Answer `y` when prompted
Include Gateway? [y/N]
```

Then run:
```bash
make deploy

# Attach to the running instance
docker attach chimerabot
```

By default, Gateway will start in development mode with unencrypted HTTP endpoints. To run in production model with encrypted HTTPS, use the `DEV=false` flag and run `gateway generate-certs` in ChimeraBot to generate the certificates needed. See [Development vs Production Modes](http://chimerabot.org/gateway/installation/#development-vs-production-modes) for more information.

---

For comprehensive installation instructions and troubleshooting, visit our [Installation](https://chimerabot.org/installation/) documentation.

## Getting Help

If you encounter issues or have questions, here's how you can get assistance:

* Consult our [FAQ](https://chimerabot.org/faq/), [Troubleshooting Guide](https://chimerabot.org/troubleshooting/), or [Glossary](https://chimerabot.org/glossary/)
* To report bugs or suggest features, submit a [Github issue](https://github.com/chimerabot/chimerabot/issues)
* Join our [Discord community](https://discord.gg/chimerabot) and ask questions in the #support channel

We pledge that we will not use the information/data you provide us for trading purposes nor share them with third parties.

## Exchange Connectors

ChimeraBot connectors standardize REST and WebSocket API interfaces to different types of exchanges, enabling you to build sophisticated trading strategies that can be deployed across many exchanges with minimal changes.

### Connector Types

We classify exchange connectors into three main categories:

* **CLOB CEX**: Centralized exchanges with central limit order books that take custody of your funds. Connect via API keys.
  - **Spot**: Trading spot markets
  - **Perpetual**: Trading perpetual futures markets

* **CLOB DEX**: Decentralized exchanges with on-chain central limit order books. Non-custodial, connect via wallet keys.
  - **Spot**: Trading spot markets on-chain
  - **Perpetual**: Trading perpetual futures on-chain

* **AMM DEX**: Decentralized exchanges using Automated Market Maker protocols. Non-custodial, connect via Gateway middleware.
  - **Router**: DEX aggregators that find optimal swap routes
  - **AMM**: Traditional constant product (x*y=k) pools
  - **CLMM**: Concentrated Liquidity Market Maker pools with custom price ranges

### Exchange Sponsors

We are grateful for the following exchanges that support the development and maintenance of ChimeraBot via broker partnerships and sponsorships.

| Exchange | Type | Sub-Type(s) | Connector ID(s) | Discount |
|------|------|------|-------|----------|
| [Binance](https://chimerabot.org/exchanges/binance/) | CLOB CEX | Spot, Perpetual | `binance`, `binance_perpetual` | [![Sign up for Binance using ChimeraBot's referral link for a 10% discount!](https://img.shields.io/static/v1?label=Fee&message=%2d10%25&color=orange)](https://accounts.binance.com/register?ref=CBWO4LU6) |
| [BitMart](https://chimerabot.org/exchanges/bitmart/) | CLOB CEX | Spot, Perpetual | `bitmart`, `bitmart_perpetual` | [![Sign up for BitMart using ChimeraBot's referral link!](https://img.shields.io/static/v1?label=Sponsor&message=Link&color=orange)](https://www.bitmart.com/invite/ChimeraBot/en) |
| [Bitget](https://chimerabot.org/exchanges/bitget/) | CLOB CEX | Spot, Perpetual | `bitget`, `bitget_perpetual` | [![Sign up for Bitget using ChimeraBot's referral link!](https://img.shields.io/static/v1?label=Sponsor&message=Link&color=orange)](https://www.bitget.com/expressly?channelCode=v9cb&vipCode=26rr&languageType=0) |
| [Derive](https://chimerabot.org/exchanges/derive/) | CLOB DEX | Spot, Perpetual | `derive`, `derive_perpetual` | [![Sign up for Derive using ChimeraBot's referral link!](https://img.shields.io/static/v1?label=Sponsor&message=Link&color=orange)](https://www.derive.xyz/invite/7SA0V) |
| [dYdX](https://chimerabot.org/exchanges/dydx/) | CLOB DEX | Perpetual | `dydx_v4_perpetual` | - |
| [Gate.io](https://chimerabot.org/exchanges/gate-io/) | CLOB CEX | Spot, Perpetual | `gate_io`, `gate_io_perpetual` | [![Sign up for Gate.io using ChimeraBot's referral link for a 20% discount!](https://img.shields.io/static/v1?label=Fee&message=%2d20%25&color=orange)](https://www.gate.io/referral/invite/HBOTGATE_0_103) |
| [HTX (Huobi)](https://chimerabot.org/exchanges/htx/) | CLOB CEX | Spot | `htx` | [![Sign up for HTX using ChimeraBot's referral link for a 20% discount!](https://img.shields.io/static/v1?label=Fee&message=%2d20%25&color=orange)](https://www.htx.com.pk/invite/en-us/1h?invite_code=re4w9223) |
| [Hyperliquid](https://chimerabot.org/exchanges/hyperliquid/) | CLOB DEX | Spot, Perpetual | `hyperliquid`, `hyperliquid_perpetual` | - |
| [KuCoin](https://chimerabot.org/exchanges/kucoin/) | CLOB CEX | Spot, Perpetual | `kucoin`, `kucoin_perpetual` | [![Sign up for Kucoin using ChimeraBot's referral link for a 20% discount!](https://img.shields.io/static/v1?label=Fee&message=%2d20%25&color=orange)](https://www.kucoin.com/r/af/chimerabot) |
| [OKX](https://chimerabot.org/exchanges/okx/) | CLOB CEX | Spot, Perpetual | `okx`, `okx_perpetual` | [![Sign up for OKX using ChimeraBot's referral link for a 20% discount!](https://img.shields.io/static/v1?label=Fee&message=%2d20%25&color=orange)](https://www.okx.com/join/1931920269) |
| [XRP Ledger](https://chimerabot.org/exchanges/xrpl/) | CLOB DEX | Spot | `xrpl` | - |

### Other Exchange Connectors

Currently, the master branch of ChimeraBot also includes the following exchange connectors, which are maintained and updated through the ChimeraBot Foundation governance process. See [Governance](https://chimerabot.org/governance/) for more information.

| Exchange | Type | Sub-Type(s) | Connector ID(s) | Discount |
|------|------|------|-------|----------|
| [0x Protocol](https://chimerabot.org/exchanges/gateway/0x/) | AMM DEX | Router | `0x` | - |
| [AscendEx](https://chimerabot.org/exchanges/ascendex/) | CLOB CEX | Spot | `ascend_ex` | - |
| [Balancer](https://chimerabot.org/exchanges/gateway/balancer/) | AMM DEX | AMM | `balancer` | - |
| [BingX](https://chimerabot.org/exchanges/bing_x/) | CLOB CEX | Spot | `bing_x` | - |
| [Bitrue](https://chimerabot.org/exchanges/bitrue/) | CLOB CEX | Spot | `bitrue` | - |
| [Bitstamp](https://chimerabot.org/exchanges/bitstamp/) | CLOB CEX | Spot | `bitstamp` | - |
| [BTC Markets](https://chimerabot.org/exchanges/btc-markets/) | CLOB CEX | Spot | `btc_markets` | - |
| [Bybit](https://chimerabot.org/exchanges/bybit/) | CLOB CEX | Spot, Perpetual | `bybit`, `bybit_perpetual` | - |
| [Coinbase](https://chimerabot.org/exchanges/coinbase/) | CLOB CEX | Spot | `coinbase_advanced_trade` | - |
| [Cube](https://chimerabot.org/exchanges/cube/) | CLOB CEX | Spot | `cube` | - |
| [Curve](https://chimerabot.org/exchanges/gateway/curve/) | AMM DEX | AMM | `curve` | - |
| [Dexalot](https://chimerabot.org/exchanges/dexalot/) | CLOB DEX | Spot | `dexalot` | - |
| [Injective Helix](https://chimerabot.org/exchanges/injective/) | CLOB DEX | Spot, Perpetual | `injective_v2`, `injective_v2_perpetual` | - |
| [Jupiter](https://chimerabot.org/exchanges/gateway/jupiter/) | AMM DEX | Router | `jupiter` | - |
| [Kraken](https://chimerabot.org/exchanges/kraken/) | CLOB CEX | Spot | `kraken` | - |
| [Meteora](https://chimerabot.org/exchanges/gateway/meteora/) | AMM DEX | CLMM | `meteora` | - |
| [MEXC](https://chimerabot.org/exchanges/mexc/) | CLOB CEX | Spot | `mexc` | - |
| [PancakeSwap](https://chimerabot.org/exchanges/gateway/pancakeswap/) | AMM DEX | AMM | `pancakeswap` | - |
| [QuickSwap](https://chimerabot.org/exchanges/gateway/quickswap/) | AMM DEX | AMM | `quickswap` | - |
| [Raydium](https://chimerabot.org/exchanges/gateway/raydium/) | AMM DEX | AMM, CLMM | `raydium` | - |
| [SushiSwap](https://chimerabot.org/exchanges/gateway/sushiswap/) | AMM DEX | AMM | `sushiswap` | - |
| [Trader Joe](https://chimerabot.org/exchanges/gateway/traderjoe/) | AMM DEX | AMM | `traderjoe` | - |
| [Uniswap](https://chimerabot.org/exchanges/gateway/uniswap/) | AMM DEX | Router, AMM, CLMM | `uniswap` | - |
| [Vertex](https://chimerabot.org/exchanges/vertex/) | CLOB DEX | Spot | `vertex` | - |

## Other ChimeraBot Repos

* [Condor](https://github.com/chimerabot/condor): Telegram Interface for ChimeraBot
* [ChimeraBot API](https://github.com/chimerabot/chimerabot-api): The central hub for running ChimeraBot trading bots
* [ChimeraBot MCP](https://github.com/chimerabot/mcp): Enables AI assistants like Claude and Gemini to interact with ChimeraBot for automated cryptocurrency trading across multiple exchanges.
* [Quants Lab](https://github.com/chimerabot/quants-lab): Jupyter notebooks that enable you to fetch data and perform research using ChimeraBot
* [Gateway](https://github.com/chimerabot/gateway): Typescript based API client for DEX connectors
* [ChimeraBot Site](https://github.com/chimerabot/chimerabot-site): Official documentation for ChimeraBot - we welcome contributions here too!

## Contributions

The ChimeraBot architecture features modular components that can be maintained and extended by individual community members.

We welcome contributions from the community! Please review these [guidelines](./CONTRIBUTING.md) before submitting a pull request.

To have your exchange connector or other pull request merged into the codebase, please submit a New Connector Proposal or Pull Request Proposal, following these [guidelines](https://chimerabot.org/about/proposals/). Note that you will need the required governance tokens in your Ethereum wallet to submit a proposal.

## Legal

* **License**: ChimeraBot is open source and licensed under [Apache 2.0](./LICENSE).
* **Data collection**: See [Reporting](https://chimerabot.org/reporting/) for information on anonymous data collection and reporting in ChimeraBot.
