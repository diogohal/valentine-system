"""Mock implementation of Goodness-of-Fit matcher.

This class is intentionally a stub to enable the frontend to select and submit
Goodness-of-Fit as a matching algorithm.

It does not implement any real matching logic yet.
"""

from typing import List, Dict, Union
import uuid
import numpy as np

from ..base_matcher import BaseMatcher
from ...data_sources.base_db import BaseDB
from ...data_sources.base_table import BaseTable
from .gof_methods import ks_test, ad_test, chisq_test, g_test


class GoodnessOfFit(BaseMatcher):
    """A mock matcher for Goodness-of-Fit.

    Parameters
    ----------
    top_ranking: int
        Number of top matches to return (stubbed, not used yet).
    continuous_threshold: int
        Threshold for continuous data (stubbed, not used yet).
    p_value_threshold: float
        P-value threshold (stubbed, not used yet).
    """

    def __init__(self, top_ranking: int = 10, continuous_threshold: int = 127, p_value_threshold: float = 0.95):
        # basic parameters
        self.top_ranking = int(top_ranking)
        self.continuous_threshold = int(continuous_threshold)
        self.p_value_threshold = float(p_value_threshold)
        # identifiers used by other matchers for logging/tracking
        self.uuid: str = str(uuid.uuid4())
        self.target_name: str = ""
        self.source_guid = None
        self.target_guid = None

    def get_matches(self, source_input: Union[BaseDB, BaseTable], target_input: Union[BaseDB, BaseTable]) -> List[Dict]:
        """Process incoming tables and compute placeholder statistics.

        The implementation follows the pattern used in
        :class:`CorrelationClustering`: gather all tables from both inputs, iterate
        over every pair (skipping identical objects), convert columns to numpy
        arrays and delegate the actual column-wise computation to
        ``attr_cart_product``. Results from that helper are concatenated and
        returned as a flat list. This ensures the front‑end can receive data
        derived from the input even though the tests themselves are still
        simplistic.
        """
        # record identifiers for compatibility
        self.target_name = target_input.name
        self.source_guid = source_input.db_belongs_uid if isinstance(source_input, BaseTable) else source_input.unique_identifier
        self.target_guid = target_input.db_belongs_uid if isinstance(target_input, BaseTable) else target_input.unique_identifier

        all_tables: List[Union[BaseTable, BaseDB]] = (
            list(source_input.get_tables().values()) +
            list(target_input.get_tables().values())
        )

        t1 = all_tables[0]
        t2 = all_tables[1]
        
        print("-------------------------------------------------------------------")
        print(f"Processing {len(all_tables)} tables for Goodness-of-Fit matcher...")
        print("-------------------------------------------------------------------")

        results: List[Dict] = []

        print(f"Comparing tables: {t1.name} vs {t2.name}...")
        base_cols = [col.name for col in t1.get_columns()]
        new_cols = [col.name for col in t2.get_columns()]
        try:
            base_data = np.vstack([col.data for col in t1.get_columns()]).T
            new_data = np.vstack([col.data for col in t2.get_columns()]).T
        except Exception:
            print(f"Error converting columns to numpy arrays. Skipping matching.")
            return results
        interim = self.attr_cart_product(
            base_cols, base_data, new_cols, new_data,
            t1.name, t2.name,
            delimiter=self.continuous_threshold
        )
        results.extend(interim)
        
        filtered_results = self.getOnlyHighestPValueBetweenTests(results)
        filtered_results = self.truncateResultsForEachColumn(filtered_results)
        print(f"------------------- Completed processing. Total matches found: {len(filtered_results)} ------------------")
        print(f"Results:")
        for result in filtered_results:
            print(f"Col1: {result[2]}, Col2: {result[3]}, Test: {result[4]}, Statistic: {result[5]:.4f}, P-value: {result[6]:.4f}")
        
        print("-------------------------------------------------------------------")

        return self._to_valentine_matches(filtered_results, t1, t2)

    def _to_valentine_matches(self, results, t1, t2):
        """Convert internal result rows to Valentine match dicts."""
        valentine_matches = []
        for result in results:
            col1 = result[2]
            col2 = result[3]
            pvalue = float(result[6])

            valentine_matches.append({
                "source": {
                    "db_guid": self.source_guid,
                    "tbl_nm": t1.name,
                    "tbl_guid": t1.unique_identifier,
                    "clm_nm": col1,
                    "clm_guid": t1.get_guid_column_lookup.get(col1)
                },
                "target": {
                    "db_guid": self.target_guid,
                    "tbl_nm": t2.name,
                    "tbl_guid": t2.unique_identifier,
                    "clm_nm": col2,
                    "clm_guid": t2.get_guid_column_lookup.get(col2)
                },
                "sim": pvalue
            })

        print(f"valentine_matches: {valentine_matches}")
        return valentine_matches
    
    def attr_cart_product(self, base_cols, base_data, new_cols, new_data, dist1, dist2, delimiter=127, hist_bin=10):
        results = []
        for i, col1 in enumerate(base_cols):
            for j, col2 in enumerate(new_cols):
                print(f"Comparing {col1} with {col2}...")
                nuniq1 = len(set(base_data[:, i]))
                nuniq2 = len(set(new_data[:, j]))

                if nuniq1 < 2:
                    break
                elif nuniq2 < 2:
                    continue

                data1 = base_data[:, i]
                data2 = new_data[:, j]

                # Avoiding dtype string columns
                try:
                    data1 = data1.astype(float)
                    data2 = data2.astype(float)
                except ValueError:
                    print(f"Skipping comparison due to non-numeric data.")
                    continue      

                if nuniq1 <= delimiter and nuniq2 <= delimiter:
                    uniq1 = set(data1)
                    uniq2 = set(data2)

                    if len(uniq1.intersection(uniq2)) < 2:
                        continue
                    res = chisq_test(data1, data2)
                    results.append([dist1, dist2, col1, col2, 'CHISQ', res.statistic, res.pvalue])
                    res = g_test(data1, data2)
                    results.append([dist1, dist2, col1, col2, 'G', res.statistic, res.pvalue])

                elif nuniq1 > delimiter and nuniq2 > delimiter:
                    res = ks_test(data1, data2, hist_bin)
                    results.append([dist1, dist2, col1, col2, 'KS', res.statistic, res.pvalue])
                    res = ad_test(data1, data2, hist_bin)
                    results.append([dist1, dist2, col1, col2, 'AD', res.statistic, res.pvalue])
        return results
    
    def getOnlyHighestPValueBetweenTests(self, results):
        # Group results by col1 and col2
        grouped = {}
        for result in results:
            key = (result[2], result[3])  # (col1, col2)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # For each group, keep only the test with the highest p-value
        filtered_results = []
        for key, group in grouped.items():
            best_result = max(group, key=lambda x: x[6])  # x[6] is the p-value
            if best_result[6] >= self.p_value_threshold:
                filtered_results.append(best_result) # Only keep if p-value meets threshold

        return filtered_results
    
    def truncateResultsForEachColumn(self, results):
        # Group results by col1
        grouped = {}
        for result in results:
            key = result[2]  # col1
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # For each group, keep only the top N results based on p-value
        truncated_results = []
        for key, group in grouped.items():
            sorted_group = sorted(group, key=lambda x: x[6], reverse=True)  # Sort by p-value
            truncated_results.extend(sorted_group[:self.top_ranking])  # Keep top N

        return truncated_results