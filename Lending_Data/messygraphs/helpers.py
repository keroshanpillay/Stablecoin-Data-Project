"""
Supporting helpers for Messari Subgraphs
"""

import pandas as pd
import requests


def get_deployments() -> pd.DataFrame:
    url = "https://subgraphs.messari.io/deployments.json"
    response = requests.get(url).json()

    df = pd.DataFrame(response)

    df2 = df.stack().to_frame()
    types = [item[1] for item in df2.index]

    deployments = pd.DataFrame(df2[0].tolist())
    deployments.index = df.index

    # set schema in index
    deployments["schema"] = types
    deployments = deployments.set_index([deployments.index, "schema"])

    # bring up schema to second level of axis=1 & sort
    deployments = deployments.unstack().swaplevel(axis=1).sort_index(axis=1)
    return deployments


def get_network_subgraphs() -> pd.DataFrame:

    # TODO: make this programatic
    network_subgraphs = [
        {"name": "arbitrum-one", "slug": "arbitrum", "nakamoto.ratio": 0},
        {"name": "aurora", "slug": "aurora-near", "nakamoto.ratio": 0},
        {"name": "avalanche", "slug": "avalanche", "nakamoto.ratio": 0},
        {"name": "boba", "slug": "boba-network", "nakamoto.ratio": 0},
        {"name": "bsc", "slug": "binance-coin", "nakamoto.ratio": 0},
        {"name": "celo", "slug": "celo", "nakamoto.ratio": 0},
        {"name": "clover", "slug": "clover-finance", "nakamoto.ratio": 0},
        {"name": "cronos", "slug": "cronos", "nakamoto.ratio": 0},
        {"name": "fantom", "slug": "fantom", "nakamoto.ratio": 0},
        {"name": "fuse", "slug": "fuse", "nakamoto.ratio": 0},
        {"name": "harmony", "slug": "harmony", "nakamoto.ratio": 0},
        {"name": "mainnet", "slug": "ethereum", "nakamoto.ratio": 0},
        {"name": "matic", "slug": "polygon", "nakamoto.ratio": 0},
        {"name": "moonbeam", "slug": "moonbeam", "nakamoto.ratio": 0},
        {"name": "moonriver", "slug": "moonriver", "nakamoto.ratio": 0},
        {"name": "optimism", "slug": "optimism", "nakamoto.ratio": 0},
        {"name": "xdai", "slug": "xdai", "nakamoto.ratio": 0},
        {"name": "arweave-mainnet", "slug": "arweave", "nakamoto.ratio": 0},
        {"name": "cosmos", "slug": "cosmos", "nakamoto.ratio": 0},
        {"name": "juno", "slug": "juno-network", "nakamoto.ratio": 0},
        {"name": "osmosis", "slug": "osmosis", "nakamoto.ratio": 0},
        {"name": "near", "slug": "near-protocol", "nakamoto.ratio": 0},
    ]

    network_subgraphs = pd.DataFrame(network_subgraphs)
    network_subgraphs["url"] = network_subgraphs["name"].apply(
        lambda x: f"https://api.thegraph.com/subgraphs/name/dmelotik/network-{x}"
    )
    return network_subgraphs


def date_filter_df(
    df: pd.DataFrame, start, end, col_name: str = "date"
) -> pd.DataFrame:
    """Helper for dt filtering by date column"""
    tmp_df = df[df[col_name] >= start]
    tmp_df = tmp_df[tmp_df[col_name] <= end]
    return tmp_df
