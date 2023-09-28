import random
import time
from tqdm import tqdm
import fire
import json
import shutil
from pathlib import Path
import urllib.request

import ankipandas as akp
import numpy as np
import pandas as pd

from utils.logger import red, whi, yel

class Anki_PrioriTag:
    def __init__(
            self,
            profile_name="Default",
            k=3,
            topdecks="",
            toptags="",
            rescale_values_before=True,
            *args,
            **kwargs,
            ):
        """
        Parameters
        ----------
        profile_name: str, default 'Default'
            name of the anki profile
        k: int, default 3
            number of decks to create
        topdecks: str, default ""
            example : if you set topdecks to 'pro::companyA::subjectA' then
            only cards that are in this deck (or any of its subdeck) will
            be taken into account
            To use multiple topdecks, separate them by '//'. It will be
            additive and not exclusive.
        toptags: str, default ""
            same logic as topdecks. Any cards that does not contain this tag
            will be ignored.
            To use multiple toptags, separate them by '//'. It will be
            additive and not exclusive.
        rescale_values_before: str, default True
            if True, the values of the deck will be scaled before being
            aggregated by tag. Note: scaling is just a minmax scaling
        args / kwargs: any unexpected args or kwargs will raise an exception
        """
        if args:
            raise Exception(f"Unexpected args: '{args}'")
        if kwargs:
            raise Exception(f"Unexpected kwargs: '{kwargs}'")
        self.profile_name = profile_name
        if "//" in topdecks:
            self.topdecks = [d.strip() for d in topdecks.split("//")]
        else:
            self.topdecks = [topdecks.strip()]
        if "//" in toptags:
            self.toptags = [t.strip() for t in toptags.split("//")]
        else:
            self.toptags = [toptags.strip()]
        self.rescale_values_before = rescale_values_before
        assert isinstance(k, int), f"arg k must be int, not {k}"
        self.k = k

        self._do_sync()

        # get the deck_config for each deck
        self.decks = [
                d for d in self._call_anki(action="deckNames")
                ]
        to_remove = []
        for i, d in enumerate(self.decks):
            excl = True
            for topd in self.topdecks:
                if d.startswith(topd):
                    excl = False
                    break
            if excl:
                to_remove.append(d)
        self.decks = [d for d in self.decks if d not in to_remove]
        assert self.decks, "Empty list of decks!"
        self.decks_config = {
                d: self._call_anki(action="getDeckConfig", deck=d)
                for d in tqdm(self.decks, desc="Loading deck config")
                }
        # exclude filtered decks
        self.decks_config = {
                d: conf
                for d, conf in self.decks_config.items()
                if conf["dyn"] == 0
                }

        self._load_db()

        df = self._compute_scoring()

        urgent_tags = self._compute_aggregate_scoring(df)

        picked = random.sample(urgent_tags, k=self.k)

        self._create_filtered_deck(picked)

        self._do_sync()

        raise SystemExit("Done")

    def _do_sync(self):
        sync_output = self._call_anki(action="sync")
        assert sync_output is None or sync_output == "None", (
            f"Error during sync?: '{sync_output}'")
        time.sleep(1)  # wait for sync to finish, just in case
        whi("Done syncing!")

    def _load_db(self):
        """
        find the location of the anki collection, then copy it to a cache folder
        to make 1. make sure anki pandas will not affect the db and 2. open
        even if the db is locked.
        """
        whi(f"Loading db from profile {self.profile_name}")
        cache_dir = Path(".cache")
        cache_dir.mkdir(exist_ok=True)
        [file.unlink() for file in cache_dir.rglob("*")]  # empty cache
        original_db = akp.find_db(user=self.profile_name)
        name = f"{self.profile_name}_{int(time.time())}".replace(" ", "_")
        temp_db = shutil.copy(
            original_db,
            f"./.cache/{name.replace('/', '_')}"
            )
        col = akp.Collection(path=temp_db)

        # keep only unsuspended cards from the right deck
        df = col.cards.merge_notes()

        # fix a weird character in the deckname
        df["cdeck"] = df["cdeck"].apply(
            lambda x: x.replace("\x1f", "::"))

        # keep only cards children of the topdecks
        def deck_checker(deckname):
            for d in self.topdecks:
                if deckname.startswith(d):
                    return True
            return False
        df["in_topdecks"] = df["cdeck"].apply(lambda x: deck_checker(x))
        df = df[df["in_topdecks"]]

        # remove suspended
        df = df[df["cqueue"] != "suspended"]

        # exclude if no tag are children of the toptags
        self.tag_list = set()
        exclude = []
        for i in tqdm(df.index, desc="filtering cards by top tags"):
            keep = False
            for t in df.loc[i, "ntags"]:
                for one_toptag in self.toptags:
                    if t.startswith(one_toptag):
                        if "anki" not in t.lower():  # exclude tags like AnkiSummary, ankiconnect etc
                            if t not in ["marked", "leech"]:
                                self.tag_list.add(t)
                        keep = True
                        break
            if not keep:
                exclude.append(i)
        df.drop(index=exclude, inplace=True)

        self.df = df.copy()

    def _compute_scoring(self):
        whi("Computing scoring of cards")
        df = self.df

        # setting interval to correct value for learning and relearnings:
        df["adjusted_steps"] = False

        for deck, deck_config in tqdm(self.decks_config.items(), desc="processing each deck"):

            steps_L = sorted([x / 1440 for x in deck_config["new"]["delays"]])
            steps_RL = sorted([x / 1440 for x in deck_config["lapse"]["delays"]])

            for i in df.index:
                if df.loc[i, "cdeck"] != deck:
                    continue
                if df.loc[i, "adjusted_steps"]:
                    continue

                # if a card was reset, don't take into account the previous intervals
                if df.loc[i, "cqueue"] == "new":
                    df.loc[i, "codue"] = 0
                    df.loc[i, "cdue"] = 0

                if df.loc[i, "ctype"] == "learning":  # learning
                    step_L_index = int(str(df.loc[i, "cleft"])[-3:])-1
                    try:
                        df.at[i, "civl"] = steps_L[step_L_index]
                    except Exception as e:
                        whi(f"Invalid learning step, card was recently moved from another deck? cid: {i}; '{e}'")
                        df.at[i, "civl"] = steps_L[0]

                    assert df.at[i,
                                 "civl"] >= 0, (
                                         f"negative interval for card {i}")

                elif df.loc[i, "ctype"] == "relearning":  # relearning
                    step_RL_index = int(str(df.loc[i, "cleft"])[-3:])-1
                    try:
                        df.at[i, "civl"] = steps_RL[step_RL_index]
                    except Exception as e:
                        whi(f"Invalid relearning step, card was recently moved from another deck? cid: {i}; '{e}'")
                        df.at[i, "civl"] = steps_RL[0]
                    assert df.at[i,
                                 "civl"] >= 0, (
                                         f"negative interval for card {i}")

                # TODO: check that it's okay to repeat this?
                # assert df.loc[i, "adjusted_steps"] is False, (
                #     f"Card {i} was already adjusted!")
                df.at[i, "adjusted_steps"] = True

                if df.loc[i, "civl"] < 0:  # negative values are in seconds
                    yel(f"Changing interval: cid: {i}, ivl: "
                        f"{df.loc[i, 'interval']} => "
                        f"{df.loc[i, 'interval']/(-86400)}")
                    df.at[i, "civl"] /= -86400


        # make sure no cards were missed when iterating over the decks
        non_adjusted_ids = []
        for i in df.index:
            if not df.loc[i, "adjusted_steps"]:
                non_adjusted_ids.append(i)
        if non_adjusted_ids:
            raise Exception("Non adjusted cards!")

        # by lowest interval
        ivl = df['civl'].to_numpy().reshape(-1, 1).squeeze()
        # make start at 0
        assert (ivl >= 0).all(), "Negative intervals"
        ivl -= ivl.min()
        if ivl.max() != 0:
            ivl = ivl / ivl.max()
        df["scaled_civl"] = ivl

        # relative overdueness
        # note: the code for relative overdueness is not exactly the same as
        # in anki, as I was not able to fully replicate it.
        # Here's a link to one of the original implementation :
        # https://github.com/ankitects/anki/blob/afff4fc437f523a742f617c6c4ad973a4d477c15/rslib/src/storage/card/filtered.rs

        # first, get the offset for due cards values that are timestamp
        anki_col_time = int(self._call_anki(
            action="getCollectionCreationTime"))
        time_offset = int((time.time() - anki_col_time) / 86400)

        df["ref_due"] = np.nan
        for i in df.index:
            df.at[i, "ref_due"] = df.loc[i, "codue"]
            if df.loc[i, "ref_due"] == 0:
                df.at[i, "ref_due"] = df.at[i, "cdue"]
            if df.loc[i, "ref_due"] >= 100_000:  # timestamp and not days
                df.at[i, "ref_due"] -= anki_col_time
                df.at[i, "ref_due"] /= 86400
            assert df.at[i,
                         "ref_due"] >= 0, f"negative interval for card {i}"
        df["overdue"] = df["ref_due"] - time_offset

        # then, correct overdue values to make sure they are negative
        correction = max(df["overdue"].values.max(), 0) + 0.01

        # my implementation of relative overdueness:
        # (intervals are positive, overdue are negative for due cards
        # hence ro is positive)
        # low ro means urgent, high ro means not urgent
        assert (df["civl"].values >= 0).all(), "Negative interval"
        assert correction >= 0, "Negative correction factor"
        assert (-df["overdue"].values + correction > 0).all(), "Positive overdue - correction"
        ro = (df["civl"].values + correction) / (-df["overdue"] + correction)
        assert (ro >= 0).all(), "wrong values of relative overdueness"
        assert ro.max() < np.inf, "Infinity is part of relative overdueness"

        # clipping extreme values, above 1 is useless anyway
        #ro = np.clip(ro, 0, 10)

        # reduce the increase of ro as a very high ro is not important
        while ro.max() > 1.5:
            #whi("(Smoothing relative overdueness)")
            ro[ro > 1] = 1 + np.log(ro[ro > 1])

        # minmax scaling of ro
        ro -= ro.min()
        if ro.max() != 0:
            ro /= ro.max()
        ro += 0.001

        df["relative_overdueness"] = ro

        # weighted mean of lowest interval and relative overdueness (LIRO)
        weights = [1, 4]
        df["LIRO"] = (weights[0] * ro + weights[1] * ivl) / sum(weights)

        # optionnaly apply minmax to the value before aggregating the tags
        if self.rescale_values_before:
            for col in [
                    "clapses",
                    "creps",
                    "LIRO",
                    "relative_overdueness",
                    ]:
                df[col] -= df[col].min()
                if df[col].max() != 0:
                    df[col] /= df[col].max()

        # avoid dividing by 0
        df["creps"] += 1

        return df

    def _compute_aggregate_scoring(self, df):
        tag_summary = pd.DataFrame(index=list(self.tag_list))

        # sort by lapses
        skipped = []
        for t in tqdm(self.tag_list, desc="scoring each tag"):
            grouped = df.groupby(df["ntags"].apply(lambda x: t in x))
            if True not in grouped.groups.keys():
                skipped.append(t)
                continue
            else:
                dfgt = grouped.get_group(True)
            assert not dfgt.empty, f"the dataframe could not be grouped by tag '{t}'"


            tag_summary.loc[t, "mean_lapse"] = dfgt["clapses"].mean()
            tag_summary.loc[t, "median_lapse"] = dfgt["clapses"].median()

            tag_summary.loc[t, "lapse_over_review"] = dfgt["clapses"].sum() / dfgt["creps"].sum()

            tag_summary.loc[t, "mean_lapse_over_review"] = (dfgt["clapses"] / dfgt["creps"]).mean()
            tag_summary.loc[t, "median_lapse_over_review"] = (dfgt["clapses"] / dfgt["creps"]).median()

            tag_summary.loc[t, "mean_review"] = dfgt["creps"].mean()
            tag_summary.loc[t, "median_review"] = dfgt["creps"].median()
            tag_summary.loc[t, "sum_review"] = dfgt["creps"].sum()

            tag_summary.loc[t, "mean_interval"] = dfgt["scaled_civl"].mean()
            tag_summary.loc[t, "median_interval"] = dfgt["scaled_civl"].median()

            tag_summary.loc[t, "mean_LIRO"] = dfgt["LIRO"].mean()
            tag_summary.loc[t, "median_LIRO"] = dfgt["LIRO"].median()

            tag_summary.loc[t, "mean_relative_overdueness"] = dfgt["relative_overdueness"].mean()
            tag_summary.loc[t, "median_relative_overdueness"] = dfgt["relative_overdueness"].median()

            tag_summary.loc[t, "mean_new"] = (dfgt["cqueue"].values == "new").mean()
            tag_summary.loc[t, "mean_learning"] = (dfgt["ctype"].values == "learning").mean()
            tag_summary.loc[t, "mean_relearning"] = (dfgt["ctype"].values == "relearning").mean()

        if skipped:
            whi("Skipped tags:" + '\n    * '.join(t for t in sorted(skipped)))
            whi("\n\n")

        # a smaller value indicate an urgent tag, high value indicate
        # an up to date tag. So let's reverse some columns
        col_to_reverse = [
                "mean_lapse",
                "median_lapse",
                "lapse_over_review",
                "mean_lapse_over_review",
                "median_lapse_over_review",
                "mean_new",
                "mean_learning",
                "mean_relearning",
                ]
        for col in col_to_reverse:
            tag_summary[col] *= -1

        # minmax scaling of value
        for col in tag_summary.columns:
            tag_summary[col] -= tag_summary[col].min()
            if tag_summary[col].max() != 0:
                tag_summary[col] /= tag_summary[col].max()

        # getting the index of each score
        for col in sorted(tag_summary.columns.tolist()):
            sorted_col = "s_" + col
            tag_summary[sorted_col] = tag_summary[col].argsort()
            worst = tag_summary[col].idxmin()
            whi(f"{col:<40} - {worst}")

        sorted_columns = [c for c in tag_summary.columns.tolist() if c.startswith("s_")]
        tag_summary["sum_sorted_index"] = tag_summary[sorted_columns].sum(axis=1)

        threshold = max(len(self.tag_list) * 0.05, 5)
        most_urgent_rows = tag_summary["sum_sorted_index"].nsmallest(int(threshold))
        most_urgent_tags = most_urgent_rows.index.tolist()

        whi("The most urgent tags are:" + "\n    * ".join([t for t in most_urgent_tags]))

        return most_urgent_tags

    def _create_filtered_deck(self, tags):

        queries = ["tag:" + tag for tag in tags]

        # remove any common parent
        if len([t for t in tags if "::" not in t]) == 0:
            for i in range(tags[0].count("::")):
                parent = tags[0].split("::")[0]
                if len([t for t in tags if t.startswith(parent)]) > 1:
                    tags = [
                            t.replace(parent + "::", "", 1)
                            if t.startswith(parent + "::")
                            else t
                            for t in tags
                            ]

        for query, tag in zip(queries, tags):
            for t in sorted(self.toptags, key=len, reverse=True):
                if tag.startswith(t):
                    tag = tag.replace(t, "")
                    while tag.startswith(":"):
                        tag = tag[1:]
                    break
            tag = tag.replace("::", ":")
            self._call_anki(
                    action="createFilteredDeck",
                    newDeckName=f"AnnA PrioTag - " + tag,
                    searchQuery=query,
                    gatherCount=1000,
                    reschedule=True,
                    sortOrder=8,  # relative overdueness
                    createEmpty=True,
                    )
            whi(f"Created filtered deck for tag '{tag}'")
        return

    def _call_anki(self, action, **params):
        """ bridge between local python libraries and AnnA Companion addon
        (a fork from anki-connect) """
        def request_wrapper(action, **params):
            return {'action': action, 'params': params, 'version': 6}

        requestJson = json.dumps(request_wrapper(action, **params)
                                 ).encode('utf-8')

        try:
            response = json.load(urllib.request.urlopen(
                urllib.request.Request(
                    'http://localhost:8775',
                    requestJson)))
        except (ConnectionRefusedError, urllib.error.URLError) as e:
            red(f"{str(e)}: is Anki open and 'AnnA Companion addon' "
                 "enabled? Firewall issue?")
            raise Exception(f"{str(e)}: is Anki open and 'AnnA Companion "
                            "addon' enabled? Firewall issue?")

        if len(response) != 2:
            red('response has an unexpected number of fields')
            raise Exception('response has an unexpected number of fields')
        if 'error' not in response:
            red('response is missing required error field')
            raise Exception('response is missing required error field')
        if 'result' not in response:
            red('response is missing required result field')
            raise Exception('response is missing required result field')
        if response['error'] is not None:
            red(response['error'])
            raise Exception(response['error'])
        return response['result']



if __name__ == "__main__":
    intance = fire.Fire(Anki_PrioriTag)
