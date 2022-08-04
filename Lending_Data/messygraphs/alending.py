from messygraphs.subgraph import Subgraph

# Networking
import asyncio
from aiohttp import ClientSession
import requests

# Datasci
import pandas as pd
import numpy as np

# Plotting
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

# other
import datetime
import time

from string import Template
from typing import List, Tuple, Dict

# purely for local testing
import copy


class aLending(Subgraph):
    def __init__(self, url: str):
        Subgraph.__init__(self, url)

        # Snapshots
        self.market_snapshots = pd.DataFrame()
        self.financial_snapshots = pd.DataFrame()

    ###############
    # Snapshots
    async def get_market_snapshots(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:

        if not self.market_snapshots.empty and not refresh:
            return self.market_snapshots

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
            query($first: Int, $start: Int, $end: Int) {
              data: marketDailySnapshots(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                protocol {
                  name
                  network
                }
                market {
                  name
                  inputTokens {
                    id
                    name
                    symbol
                    decimals
                    underlyingAsset
                  }
                  outputToken {
                    id
                    name
                    symbol
                    decimals
                    underlyingAsset
                  }
                  rewardTokens {
                    id
                    name
                    symbol
                    decimals
                  }
                  reserveFactor
                  isActive
                }
                totalValueLockedUSD
                totalVolumeUSD
                totalDepositUSD
                totalBorrowUSD
                inputTokenBalances
                inputTokenPricesUSD
                outputTokenSupply
                outputTokenPriceUSD
                rewardTokenEmissionsAmount
                rewardTokenEmissionsUSD
                blockNumber
                timestamp
                depositRate
                stableBorrowRate
                variableBorrowRate
              }
            }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }
        df = await self.loop_timestamp_query(query, session)

        # unpacking
        df["network"] = df["protocol"].apply(
            lambda x: x["network"]
        )  # NOTE: must go before protocol
        df["protocol"] = df["protocol"].apply(lambda x: x["name"])

        df["inputTokens"] = df["market"].apply(lambda x: x["inputTokens"])
        df["rewardTokens"] = df["market"].apply(lambda x: x["rewardTokens"])
        df["outputToken"] = df["market"].apply(lambda x: x["outputToken"])
        df["reserveFactor"] = df["market"].apply(lambda x: x["reserveFactor"])
        df["isActive"] = df["market"].apply(lambda x: x["isActive"])
        df["market"] = df["market"].apply(lambda x: x["name"])

        # NOTE: this warning shouldn't matter as we depricate this?
        # WARNING: Hardcoding to expect one input token only, This will bite me later
        df["inputTokenBalances"] = df["inputTokenBalances"].apply(lambda x: x[0])
        df["inputTokenPricesUSD"] = df["inputTokenPricesUSD"].apply(lambda x: x[0])
        inputTokens_df = pd.DataFrame(df["inputTokens"].apply(lambda x: x[0]).tolist())
        inputTokens_df.columns = [
            f"inputTokens.{col}" for col in inputTokens_df.columns
        ]
        inputTokens_df.index = df.index  # WARNING: This may do me dirty

        # WARNING: more bad practice of adjusting by decimal for single asset
        # df['inputTokenBalances'] = df['inputTokenBalances'] / (10 ** df['inputTokens.decimals'].astype(int))

        outputToken_df = pd.DataFrame(df["outputToken"].tolist())
        outputToken_df.columns = [
            f"outputToken.{col}" for col in outputToken_df.columns
        ]
        outputToken_df.index = df.index  # WARNING: This may do me dirty

        df = pd.concat([df, inputTokens_df, outputToken_df], axis=1)

        # Fixing typing
        df["totalValueLockedUSD"] = df["totalValueLockedUSD"].astype(float)
        df["totalVolumeUSD"] = df["totalVolumeUSD"].astype(float)
        df["totalDepositUSD"] = df["totalDepositUSD"].astype(float)
        df["totalBorrowUSD"] = df["totalBorrowUSD"].astype(float)
        df["inputTokenBalances"] = df["inputTokenBalances"].astype(float)
        df["inputTokenPricesUSD"] = df["inputTokenPricesUSD"].astype(float)
        df["outputTokenSupply"] = df["outputTokenSupply"].astype(float)
        df["outputTokenPriceUSD"] = df["outputTokenPriceUSD"].astype(float)
        # df['rewardTokenEmissionsUSD'] = df['rewardTokenEmissionsUSD'].astype(float)
        df["blockNumber"] = df["blockNumber"].astype(float)
        df["depositRate"] = df["depositRate"].astype(float)
        df["stableBorrowRate"] = df["stableBorrowRate"].astype(float)
        df["variableBorrowRate"] = df["variableBorrowRate"].astype(float)
        df["reserveFactor"] = df["reserveFactor"].astype(float)
        df["inputTokens.decimals"] = df["inputTokens.decimals"].astype(float)
        df["outputToken.decimals"] = df["outputToken.decimals"].astype(float)

        dropme = ["inputTokens", "outputToken"]
        df.drop(labels=dropme, axis=1, inplace=True, errors="ignore")

        # Sort by timestamp
        df.sort_values("timestamp", ascending=True, inplace=True)

        # Pulling out date
        print(df.keys())
        df["date"] = df["timestamp"].apply(lambda x: pd.to_datetime(x, unit="s").date())

        # Wild reindexing, this will break if a 'market' pops up twice in one day
        # df.index = pd.MultiIndex.from_tuples(list(zip(df.index, df['market'])))
        # df = df.unstack().swaplevel(axis=1).sort_index(axis=1)
        self.market_snapshots = df
        return df

    async def get_financial_snapshots(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        if not self.financial_snapshots.empty and not refresh:
            return self.financial_snapshots

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
            query($first: Int, $start: Int, $end: Int) {
              data: financialsDailySnapshots (first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                totalValueLockedUSD
                protocol {
                  name
                  network
                }
                totalVolumeUSD
                totalDepositUSD
                totalBorrowUSD
                supplySideRevenueUSD
                protocolSideRevenueUSD
                totalRevenueUSD
                blockNumber
                timestamp
              }
            }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }
        df = await self.loop_timestamp_query(query, session)

        df["network"] = df["protocol"].apply(
            lambda x: x["network"]
        )  # NOTE: must go before protocol
        df["protocol"] = df["protocol"].apply(lambda x: x.get("name"))

        # Pulling out date
        df["date"] = df["timestamp"].apply(lambda x: pd.to_datetime(x, unit="s").date())

        # Transform some data into floats
        df["totalRevenueUSD"] = df["totalRevenueUSD"].astype(float)
        df["protocolSideRevenueUSD"] = df["protocolSideRevenueUSD"].astype(float)
        df["supplySideRevenueUSD"] = df["supplySideRevenueUSD"].astype(float)
        self.financial_snapshots = df
        return df

    async def get_snapshots(
        self,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        async with ClientSession() as session:
            await asyncio.gather(
                *[
                    self.get_financial_snapshots(session),
                    self.get_usage_snapshots(session),
                    self.get_market_snapshots(session),
                ]
            )
        return

    ###############
    # Event utils
    def clean_events_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        tmp_df = df

        markets = pd.DataFrame(tmp_df["market"].tolist())
        markets.columns = [f"market.{col}" for col in markets.columns]

        assets = pd.DataFrame(tmp_df["asset"].tolist())
        assets.columns = [f"asset.{col}" for col in assets.columns]

        tmp_df = pd.concat([tmp_df, markets, assets], axis=1)
        # scale token amount by decimals

        tmp_df["amount"] = tmp_df["amount"].astype(float) / (
            10 ** tmp_df["asset.decimals"]
        )
        tmp_df["network"] = tmp_df["protocol"].apply(
            lambda x: x["network"]
        )  # NOTE: must go before protocol
        tmp_df["protocol"] = tmp_df["protocol"].apply(lambda x: x["name"])

        # type casting
        tmp_df["datetime"] = pd.to_datetime(tmp_df["timestamp"], unit="s")
        tmp_df["amount"] = tmp_df["amount"].astype(float)
        tmp_df["amountUSD"] = tmp_df["amountUSD"].astype(float)
        tmp_df["date"] = tmp_df["datetime"].apply(lambda x: x.date())
        if "profitUSD" in tmp_df.keys():  # Special case for liquidations
            tmp_df["profitUSD"] = tmp_df["profitUSD"].astype(float)

        # Decimal adjusting
        tmp_df["amount"] = tmp_df["amount"] / (
            10 ** tmp_df["asset.decimals"].astype(int)
        )

        dropme = [
            "market",
            "asset",
            "asset.decimals",
        ]
        tmp_df.drop(labels=dropme, inplace=True, axis=1, errors="ignore")
        tmp_df.drop_duplicates(inplace=True)

        return tmp_df

    ###############
    # Events
    async def get_deposits(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
    ) -> pd.DataFrame:

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
        query($first: Int, $start: Int, $end: Int) {
            data: deposits(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                timestamp

                addr: to
                user: from

                market {
                  name
                  maximumLTV
                  liquidationThreshold
                  liquidationPenalty
                }

                protocol{
                  name
                  network
                }

                asset {
                  name
                  symbol
                  decimals
                }

                amount
                amountUSD
            }
        }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }

        df = await self.loop_timestamp_query(query, session)
        df = self.clean_events_df(df)
        df["action"] = "deposit"
        return df

    async def get_withdraws(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
    ) -> pd.DataFrame:

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
        query($first: Int, $start: Int, $end: Int) {
            data: withdraws(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                timestamp

                user: to
                addr: from

                market {
                  name
                  maximumLTV
                  liquidationThreshold
                  liquidationPenalty
                }

                protocol{
                  name
                  network
                }

                asset {
                  name
                  symbol
                  decimals
                }

                amount
                amountUSD
            }
        }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }

        df = await self.loop_timestamp_query(query, session)
        df = self.clean_events_df(df)
        df["action"] = "withdraw"
        return df

    async def get_borrows(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
    ) -> pd.DataFrame:

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
        query($first: Int, $start: Int, $end: Int) {
            data: borrows(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                timestamp

                user: to
                addr: from

                market {
                  name
                  maximumLTV
                  liquidationThreshold
                  liquidationPenalty
                }

                protocol {
                  name
                  network
                }

                asset {
                  name
                  symbol
                  decimals
                }

                amount
                amountUSD
            }
        }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }

        df = await self.loop_timestamp_query(query, session)
        df = self.clean_events_df(df)
        df["action"] = "borrow"
        return df

    async def get_repays(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
    ) -> pd.DataFrame:

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
        query($first: Int, $start: Int, $end: Int) {
            data: repays(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                timestamp

                addr: to
                user: from

                market {
                  name
                  maximumLTV
                  liquidationThreshold
                  liquidationPenalty
                }

                protocol {
                  name
                  network
                }

                asset {
                  name
                  symbol
                  decimals
                }

                amount
                amountUSD
            }
        }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }

        df = await self.loop_timestamp_query(query, session)
        df = self.clean_events_df(df)
        df["action"] = "repay"
        return df

    async def get_liquidates(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
    ) -> pd.DataFrame:

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
        query($first: Int, $start: Int, $end: Int) {
            data: liquidates(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                timestamp

                addr: to
                user: from

                market {
                  name
                  maximumLTV
                  liquidationThreshold
                  liquidationPenalty
                }

                protocol {
                  name
                  network
                }

                asset {
                  name
                  symbol
                  decimals
                }

                amount
                amountUSD
                profitUSD
            }
        }
        """
        query = {
            "query": q,
            "variables": {
                "first": first,
                "start": start_timestamp,
                "end": stop_timestamp,
            },
        }

        df = await self.loop_timestamp_query(query, session)
        df = self.clean_events_df(df)
        df["action"] = "liquidate"
        return df

    async def get_events(
        self,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        # Checking for cache
        if not self.events.empty and not self.users.empty and not refresh:
            return self.users, self.events

        async with ClientSession() as session:
            results = await asyncio.gather(
                *[
                    self.get_deposits(session, start_timestamp, end_timestamp, first),
                    self.get_withdraws(session, start_timestamp, end_timestamp, first),
                    self.get_borrows(session, start_timestamp, end_timestamp, first),
                    self.get_repays(session, start_timestamp, end_timestamp, first),
                    self.get_liquidates(session, start_timestamp, end_timestamp, first),
                ]
            )
        events_all = pd.concat(results).reset_index(drop=True).sort_values("timestamp")

        # Label users
        users, label_dict = self.label_users(events_all)
        for label in label_dict:
            events_all[label] = events_all["user"].apply(lambda x: label_dict[label][x])

        self.events = events_all
        self.users = users
        return users, events_all

    ###############
    # NOTE: maybe this should be a helper or on it's own? don't wanna bake my flow too hard into this
    # user labels
    def label_users(self, events_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        user_actions = (
            events_df.groupby(["user", "action"])["amountUSD"].sum().unstack()
        )
        user_actions["volume"] = (
            user_actions[["deposit", "borrow"]].fillna(0).sum(axis=1)
        )
        user_actions.columns = [f"{col}.usd" for col in user_actions.columns]

        # Has this address ever borrowed
        user_actions["borrower"] = user_actions["borrow.usd"].apply(
            lambda x: True if x > 0 else False
        )

        # Note: This is people who liquidate others, not those who get liquidated
        # TODO: We need to see when people get liquidated
        user_actions["liquidator"] = user_actions["liquidate.usd"].apply(
            lambda x: True if x > 0 else False
        )

        # Has this address ever withdrawed
        user_actions["saver"] = (
            user_actions["withdraw.usd"]
            .fillna(0)
            .apply(lambda x: True if x == 0 else False)
        )

        # Whale is volume over 1mil
        user_actions["whale"] = user_actions["volume.usd"].apply(
            lambda x: True if x > 1000000 else False
        )

        # Get top ten users
        top_ten_vol = user_actions["volume.usd"].sort_values(ascending=False)[:10][-1]
        user_actions["top.ten"] = user_actions["volume.usd"].apply(
            lambda x: True if x >= top_ten_vol else False
        )

        # Get this first times people did things
        user_firsts = events_df.groupby(["user", "action"])["datetime"].min().unstack()
        user_firsts.columns = [f"{col}.first.dt" for col in user_firsts.columns]
        user_firsts["first.dt"] = user_firsts.min(
            axis=1
        )  # must do this to get first time using protocol

        # NOTE: had to disable for sreamlit, will re-enable later
        user_firsts["time-to-borrow"] = (
            user_firsts["borrow.first.dt"] - user_firsts["deposit.first.dt"]
        )

        # Get how big users first actions were
        user_first_amounts = (
            events_df.groupby(["user", "action"])["amount"].first().unstack()
        )
        user_first_amounts.columns = [
            f"{col}.first.usd" for col in user_first_amounts.columns
        ]

        # Brind it all together
        users = pd.concat([user_actions, user_firsts, user_first_amounts], axis=1)

        ### Adding labels to events
        labels = ["whale", "saver", "borrower", "liquidator", "top.ten"]  # Add top ten
        label_dict = users[labels].to_dict()

        return users, label_dict
