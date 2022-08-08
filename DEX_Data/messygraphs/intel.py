import pandas as pd
import requests

import datetime
import time

import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Intel:
    def __init__(self, slug, bearer_key):
        self.slug = slug
        self.headers = {
            "authorization": f"Bearer {bearer_key}",
        }

        # Memory
        self.events = pd.DataFrame()

        # Constants
        self.db_url = "https://graphql.messari.io/query"

    def sauced(self) -> bool:
        """
        Use ethereum as a quick authenticator
        """
        payload = {
            "query": """
                query events ($slug: [String!]) {
                    events (first: 10, where: {assetSlugs_in: $slug}) {
                        totalCount
                    }
                }""",
            "variables": {
                "slug": [self.slug],  # TODO: we can make this a list
            },
        }

        try:
            response = requests.post(
                self.db_url, json=payload, headers=self.headers
            ).json()
        except:
            return False

        if "errors" in response.keys():
            return False
        return True

    def get_events(self, refresh=False) -> pd.DataFrame:
        """
        Return asset events
        """
        if not self.events.empty and not refresh:
            return self.events

        print(f"getting events {self.slug}")

        payload = {
            "query": """
                query events ($slug: [String!]) {
                    events (first: 1000, where: {assetSlugs_in: $slug}) {
                        totalCount
                        edges {
                            node {
                                eventName
                                importance
                                details
                                createDate
                                updateDate
                                eventDate
                                assets {
                                  slug
                                }
                            }
                        }
                    }
                }""",
            "variables": {
                "slug": [self.slug],  # TODO: we can make this a list
            },
        }

        # payload = {'query':q}
        response = requests.post(self.db_url, json=payload, headers=self.headers).json()
        tmp_df = pd.DataFrame(response["data"]["events"]["edges"])

        if tmp_df.empty:
            print(f"response empty {response}")
            self.events = tmp_df
            return tmp_df
        intel_events = pd.DataFrame(tmp_df["node"].tolist())

        colors = {
            "Low": "green",
            "Medium": "yellow",
            "High": "orange",
            "Very High": "red",
        }
        levels = {
            "Low": 1,
            "Medium": 2,
            "High": 3,
            "Very High": 4,
        }

        # Assign a color & level to each importance
        intel_events["color"] = intel_events["importance"].apply(lambda x: colors[x])
        intel_events["level"] = intel_events["importance"].apply(lambda x: levels[x])

        # unpack the event date
        # NOTE: lots of dates to chose from
        # createDate is in theory when the market knows that this is happening so it should be priced in?
        intel_events["createDate"] = intel_events["createDate"].apply(
            lambda x: pd.to_datetime(x, unit="s").date()
        )
        # updateDate is the 'latest' info about the event?
        ###intel_events['date'] = intel_events['updateDate'].apply(lambda x: pd.to_datetime(x, unit='s').date())
        # eventDate is when it actually happens?
        ###intel_events['date'] = intel_events['eventDate'].apply(lambda x: pd.to_datetime(x, unit='s').date())

        self.events = intel_events
        return intel_events

    def plotly_events(self, level: int = 0, fig: go.Figure = None, start_date = None, end_date = None) -> go.Figure:
        # NOTE: need to think about seconday y axis, or not actually?

        # Get events & filter
        events = self.get_events()
        if start_date:
            events = events[events['createDate'] >= start_date]
        if end_date:
            events = events[events['createDate'] <= end_date]
        sub_df = events[events["level"] >= level]

        # find figure & get max values
        if fig:
            new_fig = fig
            maxy = int(
                pd.DataFrame(pd.DataFrame(new_fig._data)["y"].tolist()).max().max()
            )
        else:
            new_fig = go.Figure()
            maxy = 4
        # new_fig = fig if fig else go.Figure()

        # Hacking to get biggest y value for scale

        # Add filtered events to figure & return
        #print(sub_df)
        for idx, row in sub_df.iterrows():
            date = row["createDate"] # NOTE: this could be other dates
            color = row["color"]
            name = row["eventName"]
            level = row["level"]

            #print(name)

            # Getting dates & vertical heights
            x = [date] * 6
            steps = maxy // 5
            y = [0, 1, 2, 3, 4, 5]
            y = [item * steps for item in y]

            new_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    hoverinfo="text",
                    marker_color=color,
                    text=name,
                ),
            )

        return new_fig

    # plotting intel events
    def matplotlib_events(self, level: int = 0, show_legend: bool = False):
        """
        right now only works for pure matplotlib charts.
        TODO: seaborn, plotly etc...
        If you run this is should recognize any ambient plt instance?
        """
        intel_events = self.get_events()
        if intel_events.empty:
            return

        # Filter by level
        sub_df = intel_events[intel_events["level"] >= level]

        for idx, row in sub_df.iterrows():
            date = row["date"]
            color = row["color"]
            name = row["eventName"]
            plt.axvline(x=date, color=color, label=name)

            # Do this to show names, TODO: fix up
            if show_legend:
                plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left")


### NOTE: ancient code for plotting, keeping for reference

# Add figure title
# fig.update_layout(
#    title_text="Price with Intel Events",
#    showlegend=False,
#    hovermode="x",
# )

# Set x-axis title
# fig.update_xaxes(title_text="Date")

# Set y-axes titles
# fig.update_yaxes(title_text="<b>Price</b>", secondary_y=False)
# fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

# fig.show()


# intel_events = get_intel_events(messari_asset_slug)

# ax = price.plot(
#    figsize=(15,10),
#    title=f'{messari_asset_slug} Price with Intel Events',
#    ylabel='Price ($USD)',
#    xlabel='Date'
# )

# plot_intel_events(
#    intel_events,
#    level=3,
#    show_legend=True,
# )

# Add traces
# fig = go.Figure()
# fig.add_trace(
#    go.Scatter(x=price.index, y=price['ethereum'], name='ethereum'),
# )

# mi = messari_intel('aave', messari_db_headers)
# mi.get_events()
# mi.plotly_events(level=4, fig=fig)
