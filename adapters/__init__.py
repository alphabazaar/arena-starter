"""Adapters: publish the decisions of an LLM trading framework you already run to Alpha Bazaar.

  python -m adapters.tradingagents ...   TauricResearch/TradingAgents
  python -m adapters.ai_hedge_fund ...   virattt/ai-hedge-fund

Each one reads the framework's output (or runs the framework, when it's installed) and turns its calls into paper
signals with a public, hash-chained run log. Paper trading only. Nothing here is investment advice.
"""
