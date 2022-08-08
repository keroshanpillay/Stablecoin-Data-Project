# Networking
from aiohttp import ClientSession
import requests

# Datasci
import pandas as pd

# Plotting
# other
import datetime
import time

from typing import Dict


class Subgraph:
    def __init__(self, url: str):

        self.url = url

        # Basically a cache
        self.events = pd.DataFrame()
        self.users = pd.DataFrame()
        self.users_raw = pd.DataFrame()  # for get users raw

        # Snapshots
        self.usage_snapshots = pd.DataFrame()

    # NOTE: this should be a general function
    # TODO: figure out async
    def get_users_raw(self, first: int = 1000, refresh: bool = False) -> pd.DataFrame:
        # Checking for cache
        if not self.users_raw.empty and not refresh:
            return self.users_raw

        q = """
        query($first: Int, $id: ID!) {
          accounts(first: $first, orderBy: id, orderDirection: asc, where: {id_gt: $id}) {
            id
          }
        }
        """

        # Starting Variables
        addr = "0x0000000000000000000000000000000000000000"

        query = {
            "query": q,
            "variables": {
                "first": first,
                "id": addr,
            },
        }
        response = requests.post(self.url, json=query).json()
        df = pd.DataFrame(response["data"]["accounts"])

        while len(response["data"]["accounts"]) == first:
            addr = df["id"].max()
            print(addr)

            query["variables"]["id"] = addr  # iterate to next address
            response = requests.post(self.url, json=query).json()
            tmp_df = pd.DataFrame(response["data"]["accounts"])

            df = pd.concat([df, tmp_df]).reset_index(drop=True)

        # Store & return
        self.users_raw = df
        return df

    # Snapshots
    async def loop_timestamp_query(self, query: Dict, session) -> pd.DataFrame:

        response = await session.post(self.url, json=query)
        response = await response.json()

        # Error catching
        if "data" not in response:
            print(f"warning, no data: {response} {query}")
            return pd.DataFrame()

        # Case for empty response
        df = pd.DataFrame(response["data"]["data"])
        if df.empty:
            print(f"warning, empty response {query}")
            return df

        # Loop until done
        while len(response["data"]["data"]) == query["variables"]["first"]:

            # move timestamp forward
            timestamp = int(df["timestamp"].max())
            query["variables"]["start"] = timestamp
            print(timestamp, pd.to_datetime(timestamp, unit="s"))  # progress bar

            # Get response
            response = await session.post(self.url, json=query)
            response = await response.json()

            # handle data
            tmp_df = pd.DataFrame(response["data"]["data"])
            df = pd.concat([df, tmp_df]).reset_index(drop=True)

        # Clean & return
        # TODO: fix this
        # TODO: or maybe drop duplicate IDs bc there should be no repeat there
        # df = df.drop_duplicates() # Should be handled outside of this function ?? -- gonna cause issues
        return df

    async def get_usage_snapshots(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:
        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
            query($first: Int, $start: Int, $end: Int) {
              data: usageMetricsDailySnapshots (first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                protocol {
                  name
                }
                activeUsers
                totalUniqueUsers
                dailyTransactionCount
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

        # Pulling out date
        df["date"] = df["timestamp"].apply(lambda x: pd.to_datetime(x, unit="s").date())

        df["protocol"] = df["protocol"].apply(lambda x: x.get("name"))
        df["firstTimeUsers"] = df["totalUniqueUsers"].diff()
        self.usage_snapshots = df
        return df
