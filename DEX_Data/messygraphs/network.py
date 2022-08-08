import requests
import pandas as pd
from aiohttp import ClientSession
import asyncio

import datetime

from messygraphs.subgraph import Subgraph


class Network(Subgraph):
    def __init__(self, url: str, nakamoto: float = 0.33):
        Subgraph.__init__(self, url)
        self.nakamoto = nakamoto

        # Snapshots
        # NOTE: maybe all snaps should be abstracted away
        self.daily_snapshots = pd.DataFrame()

        # Handmade snapshots
        self.author_snapshots = pd.DataFrame()
        self.author_stats = pd.DataFrame()

    ###############
    # Snapshots
    async def get_daily_snapshots(
        self,
        session,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ) -> pd.DataFrame:

        if not self.daily_snapshots.empty and not refresh:
            return self.daily_snapshots

        # input handling
        stop_timestamp = (
            end_timestamp if end_timestamp else int(datetime.datetime.now().timestamp())
        )

        q = """
            query($first: Int, $start: Int, $end: Int) {
              data: dailySnapshots(first: $first, orderDirection: asc, orderBy: timestamp, where: {timestamp_gte: $start, timestamp_lt: $end}) {
                id
                network {
                  id
                }
                cumulativeUniqueAuthors
                dailyActiveAuthors
                blockHeight
                timestamp
                dailyBlocks
                cumulativeDifficulty
                dailyCumulativeGasUsed
                dailyCumulativeGasLimit
                dailyBlockUtilization
                dailyMeanGasUsed
                dailyMeanGasLimit
                gasPrice
                dailyBurntFees
                cumulativeBurntFees
                dailyMeanRewards
                totalSupply
                dailyMeanBlockInterval
                cumulativeSize
                dailyCumulativeSize
                dailyMeanBlockSize
                dailyTransactionCount
                firstSupply
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
        df["network"] = df["network"].apply(lambda x: x.get("id"))

        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["date"] = df["datetime"].apply(lambda x: x.date())

        # Casting
        df["dailyMeanGasUsed"] = df["dailyMeanGasUsed"].astype(float)
        df["dailyMeanGasLimit"] = df["dailyMeanGasLimit"].astype(float)

        self.daily_snapshots = df
        return df

    async def get_snapshots(
        self,
        start_timestamp: int = 0,
        end_timestamp: int = None,
        first: int = 1000,
        refresh: bool = False,
    ):
        async with ClientSession() as session:
            await asyncio.gather(
                *[
                    self.get_daily_snapshots(session),
                ]
            )
        return

    #### Authors
    async def get_authors(
        self, session, sem, height: int, first: int = 1000, refresh: bool = False
    ) -> pd.DataFrame:
        q = """
            query($first: Int, $skip: Int, $height: Int) {
              data: authors(first: $first, skip: $skip, orderDirection: asc, orderBy: id, block: {number: $height}) {
                id
                cumulativeDifficulty
                cumulativeBlocksCreated
              }
            }
        """
        skip = 0
        query = {
            "query": q,
            "variables": {
                "first": first,
                "skip": skip,
                "height": height,
            },
        }
        await sem.acquire()
        print(len(asyncio.all_tasks()), "0x0")
        await asyncio.sleep(1)

        response = await session.post(self.url, json=query)
        response = await response.json()
        sem.release()

        df = pd.DataFrame(response["data"]["data"])
        while len(response["data"]["data"]) == first:
            skip += 1000

            await sem.acquire()
            print(len(asyncio.all_tasks()), df["id"].max())
            await asyncio.sleep(1)

            query["variables"]["skip"] = skip  # iterate to next address
            response = await session.post(self.url, json=query)
            response = await response.json()
            sem.release()

            response = requests.post(self.url, json=query).json()
            tmp_df = pd.DataFrame(response["data"]["data"])

            df = pd.concat([df, tmp_df]).reset_index(drop=True)

        # Casting
        df["cumulativeDifficulty"] = df["cumulativeDifficulty"].astype(float)
        df["cumulativeBlocksCreated"] = df["cumulativeBlocksCreated"].astype(float)

        # metadata, NOTE: maybe store this as an attribute?
        df["height"] = height
        return df

    async def get_author_snapshots(self, refresh: bool = False) -> pd.DataFrame:
        if not self.author_snapshots.empty and not refresh:
            return self.author_snapshots

        await self.get_snapshots()

        daily_snapshots = self.daily_snapshots
        daily_snapshots["datetime"] = pd.to_datetime(
            daily_snapshots["timestamp"], unit="s"
        )

        ### Getting start of months, day = 1
        dates = daily_snapshots["datetime"].apply(lambda x: x.date())
        days = dates.apply(lambda x: x.day)

        # This should be a list of blocks where they happened the first of the month
        # NOTE, may have to ignore the first one because 0's can be weird
        blocks = daily_snapshots[days == 1]["blockHeight"].tolist()

        async with ClientSession() as session:
            sem = asyncio.Semaphore(value=10)
            dfs = await asyncio.gather(
                *[self.get_authors(session, sem, height) for height in blocks]
            )

        df = pd.concat(dfs).reset_index(drop=True)

        block_timestamp_dict = (
            daily_snapshots[["blockHeight", "datetime"]]
            .set_index("blockHeight")
            .to_dict()
        )
        df["datetime"] = df["height"].apply(
            lambda x: block_timestamp_dict["datetime"][x]
        )

        df = df.set_index(["height", "datetime", "id"])  # set index as (height, id)
        df = df.unstack()  # Move index id to columns
        df = df.swaplevel(axis=1).sort_index(axis=1)  # swap & sort columns

        self.author_snapshots = df
        return df

    async def get_author_stats(self, refresh: bool = False) -> pd.DataFrame:
        if not self.author_stats.empty and not refresh:
            return self.author_stats

        df = await self.get_author_snapshots(refresh=refresh)

        df = df.diff()  # get diff over time
        df = df.xs("cumulativeBlocksCreated", axis=1, level=1)  # grab blocks created
        df = df.iloc[1:]  # drop the first one

        tmp_list = []
        for idx, row in df.iterrows():
            date = idx[1].date()

            # TODO, drop 0s from row
            row = row[row > 0]

            author_count = len(row)

            # get stats about block mining
            blocks_mined_total = row.sum()
            blocks_mined_median = row.median()
            blocks_mined_mean = row.mean()

            blocks_mined_max = row.max()
            blocks_mined_min = row.min()

            blocks_mined_std = row.std()

            blocks_mined_q3 = row.quantile(0.75)
            blocks_mined_q1 = row.quantile(0.25)

            # Getting realized nakamoto
            authored = row.sort_values(ascending=False).to_frame(name="authored")
            authored["authored.cumulative"] = authored["authored"].cumsum()
            authored["pct"] = authored["authored"] / blocks_mined_total
            authored["pct.cumulative"] = authored["pct"].cumsum()

            realized_nakamoto = (
                len(authored[authored["pct.cumulative"] < self.nakamoto]) + 1
            )

            # NOTE: consider normalizing this to pct so the charts are simplier
            tmp_dict = {
                "date": date,
                "author.count": author_count,
                "nakamoto.realized": realized_nakamoto,
                "blocks.authored.total": blocks_mined_total,
                "blocks.authored.mean": blocks_mined_mean,
                "blocks.authored.median": blocks_mined_median,
                "blocks.authored.max": blocks_mined_max,
                "blocks.authored.min": blocks_mined_min,
                "blocks.authored.std": blocks_mined_std,
                "blocks.authored.q3": blocks_mined_q3,
                "blocks.authored.q1": blocks_mined_q1,
            }
            tmp_list.append(tmp_dict)

        stats = pd.DataFrame(tmp_list)
        self.author_stats = stats
        return stats


# nw = network('https://api.thegraph.com/subgraphs/name/dmelotik/network-mainnet')
