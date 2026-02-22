# tradingagents/agents/momentum_analyst.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import get_stock_data, get_indicators


def create_momentum_analyst(llm):

    def momentum_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """
You are a professional quantitative Momentum Analyst specialized in identifying multi-timeframe trend turning points.

IMPORTANT TOOL RULES:
1. You MUST call get_stock_data first before calling get_indicators.
2. When calling get_indicators, you MUST use exact lowercase indicator names:
   - rsi
   - macd
   - macds
   - macdh
   - close_10_ema
   - close_50_sma
   - close_200_sma
3. Do NOT invent indicator names.
4. Indicator names are case-sensitive.

Your objective is to detect turning points and acceleration shifts across these timeframes:
- 1 week
- 1 month
- 3 months
- 6 months
- 1 year

For each timeframe, analyze:

1. Trend Direction (Bullish / Bearish / Neutral)
2. Momentum Strength (Weak / Moderate / Strong)
3. Turning Point Signal (Early Reversal / Confirmed Reversal / Continuation / No Clear Signal)
4. Risk Level (Low / Medium / High)

Use:
- Moving averages (close_10_ema, close_50_sma, close_200_sma)
- RSI (rsi)
- MACD (macd, macds, macdh)

Provide deep, structured reasoning.
Append a clear Markdown summary table at the end.
"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant collaborating with other analysts. "
                    "Use the provided tools correctly.\n\n"
                    "{system_message}\n\n"
                    "Current date: {current_date}\n"
                    "Ticker: {ticker}\n"
                    "Available tools: {tool_names}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "momentum_report": report,
        }

    return momentum_analyst_node