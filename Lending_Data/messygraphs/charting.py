"""
Module for making charts,
The aim is to make this as flexible as possible
"""
import pandas as pd

# import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# For network graphs
import networkx as nx

### This may need to be hardcoded?
categories = {
    "0x Protocol Token": "DEX",
    "Aave Token": "Lending",
    "Ampleforth": "Rebase",
    "Balancer": "DEX",
    "Balancer Pool Token": "DEX",
    "Basic Attention Token": "Misc",
    "Binance USD": "Stablecoin",
    "ChainLink Token": "Infrastructure",
    "Convex Token": "Metagovernance",
    "Curve DAO Token": "Governance",
    "Dai Stablecoin": "Stablecoin",
    "Decentraland MANA": "Metaverse",
    "DefiPulse Index": "Index",
    "Enjin Coin": "Infrastructure",
    "Ethereum Name Service": "Infrastructure",
    "Fei USD": "Stablecoin",
    "Frax": "Stablecoin",
    "Gelato Uniswap DAI/USDC LP": "LPT",
    "Gelato Uniswap USDC/USDT LP": "LPT",
    "Gemini dollar": "Stablecoin",
    "Kyber Network Crystal": "DEX",
    "Liquid staked Ether 2.0": "Staking",
    "Maker": "Lending",
    "Paxos Standard": "Stablecoin",
    #'Rai Reflex Index': None,
    "Republic Token": "Infrastructure",
    "SushiBar": "DEX",
    "Synth sUSD": "Stablecoin",
    "Synthetix Network Token": "Derivatives",
    "Tether USD": "Stablecoin",
    "TrueUSD": "Stablecoin",
    "USD Coin": "Stablecoin",
    "UST (Wormhole)": "Stablecoin",
    "Uniswap": "DEX",
    "Uniswap V2": "DEX",
    "Wrapped BTC": "L1",
    "Wrapped Ether": "L1",
    "renFIL": "Infrastructure",
    "yearn.finance": "Yield Aggregator",
}


def plot_market_snapshots_treemap(market_snapshots: pd.DataFrame):

    start_date = str(
        pd.to_datetime(
            market_snapshots["timestamp"].astype(float).min(), unit="s"
        ).date()
    )
    end_date = str(
        pd.to_datetime(
            market_snapshots["timestamp"].astype(float).max(), unit="s"
        ).date()
    )
    #### Getting values for plotting
    starting_balance = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenBalance"]
        .first()
        .to_frame("balance.starting")
    )

    ending_balance = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenBalance"]
        .last()
        .to_frame("balance.ending")
    )
    ending_price = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenPriceUSD"]
        .last()
        .to_frame("price.ending.usd")
    )

    balances_df = pd.concat([starting_balance, ending_balance, ending_price], axis=1)
    balances_df["balance.ending.usd"] = (
        balances_df["balance.ending"] * balances_df["price.ending.usd"]
    )
    balances_df["balance.change.pct"] = (
        (balances_df["balance.ending"] - balances_df["balance.starting"])
        / balances_df["balance.starting"]
        * 100
    )

    balances_df["category"] = [
        categories.get(idx, "Other") for idx in balances_df.index
    ]

    balances_df["name"] = balances_df.index

    # Do this while QA is Work in Progress
    balances_df = balances_df[balances_df["balance.ending.usd"] > 0]

    # make a log scale for pct changes
    # balances_df['balance.change.pct.log'] = balances_df['balance.change.pct'].apply(lambda x: math.log(x))

    fig = px.treemap(
        balances_df,
        path=[px.Constant("Aave-V2 (Mainnet)"), "category", "name"],
        names="name",
        values="balance.ending.usd",
        color="balance.change.pct",
        hover_data=["balance.ending.usd", "balance.change.pct", "balance.ending"],
        hover_name="name",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,  # np.average(df['lifeExp'], weights=df['last_tvl']))
        labels=["name", "balance.change.pct"],
    )
    fig.data[0].textinfo = "label+text+value+percent entry"

    fig.update_layout(
        title=f"Aave Liquidity {start_date} - {end_date}",
        margin=dict(t=50, l=25, r=25, b=25),
    )
    return fig


def plot_flows(
    events_df: pd.DataFrame, prices_df: pd.DataFrame, name: str, extra: str = ""
):
    # For title
    first_day = str(events_df["datetime"].min().date())
    last_day = str(events_df["datetime"].max().date())

    # Adjust for price
    tt = events_df.groupby(["assetSymbol", "action"])["amount"].sum().unstack()
    for col in tt.keys():
        tt[col] = prices_df["price.ending.usd"] * tt[col]

    # dealing with times where event is 0
    for event in ["deposit", "repay", "borrow", "withdraw"]:
        if event not in tt.keys():
            tt[event] = 0

    # Gotta drop this because price is super whack
    dropme = [
        "UNI-V2",
        "G-UNI",
        "GUSD",
    ]
    tt.drop(labels=dropme, axis=0, inplace=True, errors="ignore")

    # Set some key indexes
    deposit_idx = len(tt.index)
    repay_idx = deposit_idx + 1
    pool_idx = repay_idx + 1
    borrow_idx = pool_idx + 1
    withdraw_idx = borrow_idx + 1

    # [Assets] + [pool groups] + [Assets]
    labels = (
        list(tt.index)
        + ["Deposit", "Repay", "LendingPool", "Borrow", "Withdraw"]
        + list(tt.index)
    )

    # [asset0, asset0, asset1, asset1, ...] -> [deposit, repay, deposit, repay]
    incoming_sources = list(range(len(tt.index))) * 2
    incoming_sources.sort()
    incoming_targets = [deposit_idx, repay_idx] * len(tt.index)
    values = tt[["deposit", "repay"]].astype(float).values
    incoming_values = [x for xs in values for x in xs]

    # [deposit, repay] -> [pool, pool]
    pool_in_sources = [deposit_idx, repay_idx]
    pool_in_targets = [pool_idx, pool_idx]
    pool_in_values = [
        tt["deposit"].astype(float).sum(),
        tt["repay"].astype(float).sum(),
    ]

    # [pool, pool] -> [borrow, withdraw]
    pool_out_sources = [pool_idx, pool_idx]
    pool_out_targets = [borrow_idx, withdraw_idx]
    pool_out_values = [
        tt["borrow"].astype(float).sum(),
        tt["withdraw"].astype(float).sum(),
    ]

    # [borrow, ..., withdraw, ...] -> [asset0, asset1, ..., asset0, asset1, ...]
    outgoing_sources = ([borrow_idx] * len(tt.index)) + ([withdraw_idx] * len(tt.index))
    outgoing_targets = list(range(len(tt.index) + 5, len(labels))) * 2
    outgoing_values = (
        tt["borrow"].astype(float).tolist() + tt["withdraw"].astype(float).tolist()
    )

    sources = incoming_sources + pool_in_sources + pool_out_sources + outgoing_sources
    targets = incoming_targets + pool_in_targets + pool_out_targets + outgoing_targets
    values = incoming_values + pool_in_values + pool_out_values + outgoing_values

    # Plot
    link = dict(source=sources, target=targets, value=values)
    node = dict(label=labels, pad=50, thickness=5)

    data = go.Sankey(link=link, node=node)

    fig = go.Figure(data)

    fig.update_layout(
        title_text=f"{name} Lending Pool Total Flows {first_day} to {last_day} (USD) "
        + extra,
        font_size=10,
    )

    return fig


def plotly_lines(df: pd.DataFrame, fig: go.Figure = None) -> go.Figure:
    """
    Doing a line plot, index should be datetime, eveything else is fine
    """

    new_fig = fig if fig else go.Figure()
    for col in df.columns:
        new_fig.add_trace(go.Scatter(x=df.index, y=df[col].astype(float), name=col))
    return new_fig


def plotly_box_plot(
    df: pd.DataFrame,
    name: str,
    upper: bool = True,
    lower: bool = True,
    mean: bool = True,
    ylabel: str = None,
) -> go.Figure:
    # Reference: https://plotly.com/python/box-plots/
    fig = go.Figure()

    dates = df.index
    fig.add_trace(go.Box(y=[], x=dates, name=f"{name} BoxPlot"))

    mean_list = df[f"{name}.mean"] if mean else None
    median = df[f"{name}.median"]

    lowerfence = df[f"{name}.min"] if lower else None
    upperfence = df[f"{name}.max"] if upper else None

    std = df[f"{name}.std"]

    q1 = df[f"{name}.q1"]
    q3 = df[f"{name}.q3"]

    # y_title = ylabel

    fig.update_layout(
        title=name,
        xaxis_title="date",
        yaxis_title=ylabel,
    )

    fig.update_traces(
        q1=q1,
        q3=q3,
        median=median,
        lowerfence=lowerfence,
        upperfence=upperfence,
        mean=mean_list,
        sd=std,
        # notchspan=[ 0.2, 0.4, 0.6 ]
    )
    return fig


def clean_plotly_fig(fig: go.Figure) -> go.Figure:
    """
    Remove x-axis & y-axis lines
    Remove figure background
    """
    fig.update_layout(
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    # Bold move to go full RH here
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


#**** For historical aave-v2
def plot_amarket_snapshots_treemap(market_snapshots: pd.DataFrame):

    start_date = str(
        pd.to_datetime(
            market_snapshots["timestamp"].astype(float).min(), unit="s"
        ).date()
    )
    end_date = str(
        pd.to_datetime(
            market_snapshots["timestamp"].astype(float).max(), unit="s"
        ).date()
    )
    #### Getting values for plotting
    starting_balance = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenBalances"]
        .first()
        .to_frame("balance.starting")
    )

    ending_balance = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenBalances"]
        .last()
        .to_frame("balance.ending")
    )
    ending_price = (
        market_snapshots.sort_values("timestamp", ascending=True)
        .groupby("market")["inputTokenPricesUSD"]
        .last()
        .to_frame("price.ending.usd")
    )

    balances_df = pd.concat([starting_balance, ending_balance, ending_price], axis=1)
    balances_df["balance.ending.usd"] = (
        balances_df["balance.ending"] * balances_df["price.ending.usd"]
    )
    balances_df["balance.change.pct"] = (
        (balances_df["balance.ending"] - balances_df["balance.starting"])
        / balances_df["balance.starting"]
        * 100
    )

    balances_df["category"] = [
        categories.get(idx, "Other") for idx in balances_df.index
    ]

    balances_df["name"] = balances_df.index

    # Do this while QA is Work in Progress
    balances_df = balances_df[balances_df["balance.ending.usd"] > 0]

    # make a log scale for pct changes
    # balances_df['balance.change.pct.log'] = balances_df['balance.change.pct'].apply(lambda x: math.log(x))

    fig = px.treemap(
        balances_df,
        path=[px.Constant("Aave-V2 (Mainnet)"), "category", "name"],
        names="name",
        values="balance.ending.usd",
        color="balance.change.pct",
        hover_data=["balance.ending.usd", "balance.change.pct", "balance.ending"],
        hover_name="name",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,  # np.average(df['lifeExp'], weights=df['last_tvl']))
        labels=["name", "balance.change.pct"],
    )
    fig.data[0].textinfo = "label+text+value+percent entry"

    fig.update_layout(
        title=f"Aave Liquidity {start_date} - {end_date}",
        margin=dict(t=50, l=25, r=25, b=25),
    )
    return fig


def plot_aflows(
    events_df: pd.DataFrame, prices_df: pd.DataFrame, name: str, extra: str = ""
):
    # For title
    first_day = str(events_df["datetime"].min().date())
    last_day = str(events_df["datetime"].max().date())

    # Adjust for price
    tt = events_df.groupby(["asset.symbol", "action"])["amount"].sum().unstack()
    for col in tt.keys():
        tt[col] = prices_df["price.ending.usd"] * tt[col]

    # dealing with times where event is 0
    for event in ["deposit", "repay", "borrow", "withdraw"]:
        if event not in tt.keys():
            tt[event] = 0

    # Gotta drop this because price is super whack
    dropme = [
        "UNI-V2",
        "G-UNI",
        "GUSD",
    ]
    tt.drop(labels=dropme, axis=0, inplace=True, errors="ignore")

    # Set some key indexes
    deposit_idx = len(tt.index)
    repay_idx = deposit_idx + 1
    pool_idx = repay_idx + 1
    borrow_idx = pool_idx + 1
    withdraw_idx = borrow_idx + 1

    # [Assets] + [pool groups] + [Assets]
    labels = (
        list(tt.index)
        + ["Deposit", "Repay", "LendingPool", "Borrow", "Withdraw"]
        + list(tt.index)
    )

    # [asset0, asset0, asset1, asset1, ...] -> [deposit, repay, deposit, repay]
    incoming_sources = list(range(len(tt.index))) * 2
    incoming_sources.sort()
    incoming_targets = [deposit_idx, repay_idx] * len(tt.index)
    values = tt[["deposit", "repay"]].astype(float).values
    incoming_values = [x for xs in values for x in xs]

    # [deposit, repay] -> [pool, pool]
    pool_in_sources = [deposit_idx, repay_idx]
    pool_in_targets = [pool_idx, pool_idx]
    pool_in_values = [
        tt["deposit"].astype(float).sum(),
        tt["repay"].astype(float).sum(),
    ]

    # [pool, pool] -> [borrow, withdraw]
    pool_out_sources = [pool_idx, pool_idx]
    pool_out_targets = [borrow_idx, withdraw_idx]
    pool_out_values = [
        tt["borrow"].astype(float).sum(),
        tt["withdraw"].astype(float).sum(),
    ]

    # [borrow, ..., withdraw, ...] -> [asset0, asset1, ..., asset0, asset1, ...]
    outgoing_sources = ([borrow_idx] * len(tt.index)) + ([withdraw_idx] * len(tt.index))
    outgoing_targets = list(range(len(tt.index) + 5, len(labels))) * 2
    outgoing_values = (
        tt["borrow"].astype(float).tolist() + tt["withdraw"].astype(float).tolist()
    )

    sources = incoming_sources + pool_in_sources + pool_out_sources + outgoing_sources
    targets = incoming_targets + pool_in_targets + pool_out_targets + outgoing_targets
    values = incoming_values + pool_in_values + pool_out_values + outgoing_values

    # Plot
    link = dict(source=sources, target=targets, value=values)
    node = dict(label=labels, pad=50, thickness=5)

    data = go.Sankey(link=link, node=node)

    fig = go.Figure(data)

    fig.update_layout(
        title_text=f"{name} Lending Pool Total Flows {first_day} to {last_day} (USD) "
        + extra,
        font_size=10,
    )

    return fig


