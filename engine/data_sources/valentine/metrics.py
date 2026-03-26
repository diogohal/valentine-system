import math

from .golden_standard import GoldenStandardLoader
from .utils import one_to_one_matches


def _extract_source_target_column_names(matches: dict) -> tuple[set[str], set[str]]:
    """Extract source and target column names from valentine matches keys."""
    source_cols_with_matches = set()
    target_cols_with_matches = set()

    for key in matches.keys():
        src, tgt = key
        source_cols_with_matches.add(src[1])
        target_cols_with_matches.add(tgt[1])

    return source_cols_with_matches, target_cols_with_matches


def get_tp_fn(matches: dict, golden_standard: GoldenStandardLoader, n: int = None):
    """
    Calculate the true positive  and false negative numbers of the given matches

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    n : int, optional
        The percentage number that we want to consider from the ranked list (matches)
        e.g. (90) for 90% of the matches

    Returns
    -------
    (int, int)
        True positive and false negative counts
    """
    tp = 0
    fn = 0

    all_matches = list(map(lambda m: frozenset(m), list(matches.keys())))

    if n is not None:
        all_matches = all_matches[:n]

    for expected_match in golden_standard.expected_matches:
        if expected_match in all_matches:
            tp = tp + 1
        else:
            fn = fn + 1
    return tp, fn


def get_fp(matches: dict, golden_standard: GoldenStandardLoader, n: int = None):
    """
    Calculate the false positive number of the given matches

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    n : int, optional
        The percentage number that we want to consider from the ranked list (matches)
        e.g. (90) for 90% of the matches

    Returns
    -------
    int
        False positive
    """
    fp = 0

    all_matches = list(map(lambda m: frozenset(m), list(matches.keys())))

    if n is not None:
        all_matches = all_matches[:n]

    for possible_match in all_matches:
        if possible_match not in golden_standard.expected_matches:
            fp = fp + 1
    return fp


def recall(matches: dict, golden_standard: GoldenStandardLoader, one_to_one=False):
    """
    Function that calculates the recall of the matches against the golden standard. If one_to_one is set to true, it
    also performs an 1-1 match filer. Meaning that each column will match only with another one.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    one_to_one : bool, optional
        If to perform the 1-1 match filter

    Returns
    -------
    float
        The recall
    """
    if one_to_one:
        matches = one_to_one_matches(matches)
    tp, fn = get_tp_fn(matches, golden_standard)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def persistent_acc(matches: dict,
                   golden_standard: GoldenStandardLoader,
                   source_columns: list[str],
                   target_columns: list[str]):
    """Persistent accuracy for non-GoodnessOfFit algorithms using regular matches."""
    source_set = set(source_columns)
    target_set = set(target_columns)
    persistent_cols = source_set.intersection(target_set)

    print(f'--------------------------------------------------')
    print(f"Persistent columns (non-gof): {persistent_cols}")

    total_count = len(persistent_cols)
    if total_count == 0:
        return -1

    correct_count = 0
    for col in persistent_cols:
        expected_key = (('source', col), ('target', col))
        if expected_key in matches:
            correct_count += 1

    result = correct_count / total_count
    print(result)
    print(f'--------------------------------------------------')
    return result


def new_acc(matches: dict,
            golden_standard: GoldenStandardLoader,
            source_columns: list[str],
            target_columns: list[str]):
    """New-column accuracy for non-GoodnessOfFit algorithms using regular matches."""
    source_set = set(source_columns)
    target_set = set(target_columns)
    new_cols = target_set - source_set

    print(f'--------------------------------------------------')
    print(f"New columns (non-gof): {new_cols}")

    total_count = len(new_cols)
    if total_count == 0:
        return -1

    _, target_cols_with_matches = _extract_source_target_column_names(matches)

    correct_count = 0
    for col in new_cols:
        if col not in target_cols_with_matches:
            correct_count += 1

    result = correct_count / total_count
    print(result)
    print(f'--------------------------------------------------')
    return result


def missing_acc(matches: dict,
                golden_standard: GoldenStandardLoader,
                source_columns: list[str],
                target_columns: list[str]):
    """Missing-column accuracy for non-GoodnessOfFit algorithms using regular matches."""
    source_set = set(source_columns)
    target_set = set(target_columns)
    missing_cols = source_set - target_set

    print(f'--------------------------------------------------')
    print(f"Missing columns (non-gof): {missing_cols}")

    total_count = len(missing_cols)
    if total_count == 0:
        return -1

    source_cols_with_matches, _ = _extract_source_target_column_names(matches)

    correct_count = 0
    for col in missing_cols:
        if col not in source_cols_with_matches:
            correct_count += 1

    result = correct_count / total_count
    print(result)
    print(f'--------------------------------------------------')
    return result


def persistent_acc_gof(normalized_matches: dict,
                       golden_standard: GoldenStandardLoader,
                       top: int = 10,
                       sig_thresh: float = 0.95,
                       column_types: dict[str, dict[str, str]] = None):
    """
    Calculate persistent accuracy for GoodnessOfFit normalized matches.

    Parameters
    ----------
    normalized_matches : dict
        Normalized matches dict: {col1: {col2: {'KS': pval, 'AD': pval, ...}}}
    golden_standard : GoldenStandardLoader
        The golden standard loader
    top : int
        Number of top matches to consider
    sig_thresh : float
        Significance threshold for p-values
    column_types : dict[str, dict[str, str]]
        Dictionary with source and target column types

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    if column_types is not None:
        source_set = set(column_types['source'].keys())
        target_set = set(column_types['target'].keys())
    else:
        source_set = set(normalized_matches.keys())
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # Persistent columns: present in both source and target
    persistent_cols = source_set.intersection(target_set)

    continuous_count = 0
    discrete_count = 0
    total_count = len(persistent_cols)
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_persistent_acc = 0

    print(f'--------------------------------------------------')
    print(f"Persistent columns: {persistent_cols}")

    for col1 in persistent_cols:
        matches_dict = normalized_matches.get(col1, {})
        has_correct_match_any_test = False

        if len(matches_dict) == 0:
            continue

        tests_performed = set()
        for tests in matches_dict.values():
            tests_performed.update(tests.keys())

        has_cont = 'KS' in tests_performed or 'AD' in tests_performed
        has_disc = 'CHISQ' in tests_performed or 'G' in tests_performed
        if has_cont:
            continuous_count += 1
        if has_disc:
            discrete_count += 1

        # Keep only comparisons above threshold and then top values by max p-value among tests.
        filtered_candidates = []
        for col2, tests in matches_dict.items():
            valid_tests = {t: p for t, p in tests.items() if p >= sig_thresh}
            if len(valid_tests) == 0:
                continue
            filtered_candidates.append((col2, valid_tests, max(valid_tests.values())))

        filtered_candidates.sort(key=lambda x: x[2], reverse=True)
        top_candidates = filtered_candidates[:int(top)]

        # Correct persistent pair is col1 -> col1
        for col2, tests, _ in top_candidates:
            if col2 == col1:
                if 'KS' in tests:
                    ks_acc += 1
                if 'AD' in tests:
                    ad_acc += 1
                if 'CHISQ' in tests:
                    chi_acc += 1
                if 'G' in tests:
                    g_acc += 1
                if len(tests) > 0:
                    has_correct_match_any_test = True
                break

        if has_correct_match_any_test:
            total_persistent_acc += 1

    if total_count > 0:
        total_persistent_acc = total_persistent_acc / total_count
    else:
        total_persistent_acc = -1

    if continuous_count > 0:
        ks_acc = ks_acc / continuous_count
        ad_acc = ad_acc / continuous_count
    else:
        ks_acc = -1
        ad_acc = -1

    if discrete_count > 0:
        chi_acc = chi_acc / discrete_count
        g_acc = g_acc / discrete_count
    else:
        chi_acc = -1
        g_acc = -1

    result = {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_persistent_acc}
    print(result)
    print(f'--------------------------------------------------')
    return result


def new_acc_gof(normalized_matches: dict,
                golden_standard: GoldenStandardLoader,
                sig_thresh: float = 0.95,
                column_types: dict[str, dict[str, str]] = None):
    """
    Calculate new accuracy for GoodnessOfFit normalized matches.

    Parameters
    ----------
    normalized_matches : dict
        Normalized matches dict: {col1: {col2: {'KS': pval, 'AD': pval, ...}}}
    golden_standard : GoldenStandardLoader
        The golden standard loader
    sig_thresh : float
        Significance threshold for p-values
    column_types : dict[str, dict[str, str]]
        Dictionary with source and target column types

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    if column_types is not None:
        source_set = set(column_types['source'].keys())
        target_set = set(column_types['target'].keys())
    else:
        source_set = set(normalized_matches.keys())
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # New columns: only in target
    new_cols = target_set - source_set

    continuous_count = 0
    discrete_count = 0
    total_count = len(new_cols)
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_new_acc = 0

    print(f'--------------------------------------------------')
    print(f"New columns: {new_cols}")

    for col in new_cols:
        if column_types is not None:
            col_type = column_types['target'].get(col)
        else:
            col_type = None

        if col_type == 'numerica':
            continuous_count += 1
            relevant_tests = ['KS', 'AD']
        elif col_type == 'categorica':
            discrete_count += 1
            relevant_tests = ['CHISQ', 'G']

        per_test_has_sig = {test_name: False for test_name in relevant_tests}
        has_any_match_for_col = False

        # Iterate over ALL source columns to find any significant match with this new target column
        for src_matches in normalized_matches.values():
            if col not in src_matches:
                continue
            has_any_match_for_col = True
            tests = src_matches[col]
            for test_name in relevant_tests:
                p_value = tests.get(test_name)
                if p_value is not None and p_value >= sig_thresh:
                    per_test_has_sig[test_name] = True

        if 'KS' in relevant_tests and not per_test_has_sig['KS']:
            ks_acc += 1
        if 'AD' in relevant_tests and not per_test_has_sig['AD']:
            ad_acc += 1
        if 'CHISQ' in relevant_tests and not per_test_has_sig['CHISQ']:
            chi_acc += 1
        if 'G' in relevant_tests and not per_test_has_sig['G']:
            g_acc += 1

        if not has_any_match_for_col:
            total_new_acc += 1

    if total_count > 0:
        total_new_acc = total_new_acc / total_count
    else:
        total_new_acc = -1

    if continuous_count > 0:
        ks_acc = ks_acc / continuous_count
        ad_acc = ad_acc / continuous_count
    else:
        ks_acc = -1
        ad_acc = -1

    if discrete_count > 0:
        chi_acc = chi_acc / discrete_count
        g_acc = g_acc / discrete_count
    else:
        chi_acc = -1
        g_acc = -1

    result = {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_new_acc}
    print(result)
    print(f'--------------------------------------------------')
    return result


def missing_acc_gof(normalized_matches: dict,
                    golden_standard: GoldenStandardLoader,
                    sig_thresh: float = 0.95,
                    column_types: dict[str, dict[str, str]] = None):
    """
    Calculate missing accuracy for GoodnessOfFit normalized matches.

    Parameters
    ----------
    normalized_matches : dict
        Normalized matches dict: {col1: {col2: {'KS': pval, 'AD': pval, ...}}}
    golden_standard : GoldenStandardLoader
        The golden standard loader
    sig_thresh : float
        Significance threshold for p-values
    column_types : dict[str, dict[str, str]]
        Dictionary with source and target column types

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    if column_types is not None:
        source_set = set(column_types['source'].keys())
        target_set = set(column_types['target'].keys())
    else:
        source_set = set(normalized_matches.keys())
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # Missing columns: only in source
    missing_cols = source_set - target_set

    print(f'--------------------------------------------------')
    print(f"Missing columns: {missing_cols}")

    continuous_count = 0
    discrete_count = 0
    total_count = len(missing_cols)
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_missing_acc = 0
    
    for col in missing_cols:
        matches_dict = normalized_matches.get(col, {})

        if column_types is not None:
            col_type = column_types['source'].get(col)
        else:
            col_type = None

        if col_type == 'numerica':
            continuous_count += 1
            relevant_tests = ['KS', 'AD']
        elif col_type == 'categorica':
            discrete_count += 1
            relevant_tests = ['CHISQ', 'G']

        per_test_has_sig = {test_name: False for test_name in relevant_tests}

        for target_col, tests in matches_dict.items():
            for test_name in relevant_tests:
                p_value = tests.get(test_name)
                if p_value is not None and p_value >= sig_thresh:
                    per_test_has_sig[test_name] = True

        if 'KS' in relevant_tests and not per_test_has_sig['KS']:
            ks_acc += 1
        if 'AD' in relevant_tests and not per_test_has_sig['AD']:
            ad_acc += 1
        if 'CHISQ' in relevant_tests and not per_test_has_sig['CHISQ']:
            chi_acc += 1
        if 'G' in relevant_tests and not per_test_has_sig['G']:
            g_acc += 1

        if all(not per_test_has_sig[test_name] for test_name in relevant_tests):
            total_missing_acc += 1

    if total_count > 0:
        total_missing_acc = total_missing_acc / total_count
    else:
        total_missing_acc = -1

    if continuous_count > 0:
        ks_acc = ks_acc / continuous_count
        ad_acc = ad_acc / continuous_count
    else:
        ks_acc = -1
        ad_acc = -1

    if discrete_count > 0:
        chi_acc = chi_acc / discrete_count
        g_acc = g_acc / discrete_count
    else:
        chi_acc = -1
        g_acc = -1

    result = {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_missing_acc}
    print(result)
    print(f'--------------------------------------------------')
    return result


def precision(matches: dict, golden_standard: GoldenStandardLoader, one_to_one=False):
    """
    Function that calculates the precision of the matches against the golden standard. If one_to_one is set to true, it
    also performs an 1-1 match filer. Meaning that each column will match only with another one.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    one_to_one : bool, optional
        If to perform the 1-1 match filter

    Returns
    -------
    float
        The precision
    """
    if one_to_one:
        matches = one_to_one_matches(matches)
    tp, fn = get_tp_fn(matches, golden_standard)
    fp = get_fp(matches, golden_standard)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def f1_score(matches: dict, golden_standard: GoldenStandardLoader, one_to_one=False):
    """
    Function that calculates the F1 score of the matches against the golden standard. If one_to_one is set to true, it
    also performs an 1-1 match filer. Meaning that each column will match only with another one.

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    one_to_one : bool, optional
        If to perform the 1-1 match filter

    Returns
    -------
    float
        The f1_score
    """
    pr = precision(matches, golden_standard, one_to_one)
    re = recall(matches, golden_standard, one_to_one)
    if pr + re == 0:
        return 0
    return 2 * ((pr * re) / (pr + re))


def precision_at_n_percent(matches: dict, golden_standard: GoldenStandardLoader, n: int):
    """
    Function that calculates the precision at n %
    e.g. if n is 10 then only the first 10% of the matches will be considered for the precision calculation

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard
    n : int
        The integer percentage number

    Returns
    -------
    float
        The precision at n %
    """
    number_to_keep = int(math.ceil((n / 100) * len(matches.keys())))
    tp, fn = get_tp_fn(matches, golden_standard, number_to_keep)
    fp = get_fp(matches, golden_standard, number_to_keep)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall_at_sizeof_ground_truth(matches: dict, golden_standard: GoldenStandardLoader):
    """
    Function that calculates the recall at the size of the ground truth.
    e.g. if the size of ground truth size is 10 then only the first 10 matches will be considered for
    the recall calculation

    Parameters
    ----------
    matches : dict
        Ranked list of matches from the match with higher similarity to lower
    golden_standard : GoldenStandardLoader
        An object that contains all the information about the golden standard

    Returns
    -------
    float
        The recall at the size of ground truth
    """
    tp, fn = get_tp_fn(matches, golden_standard, golden_standard.size)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def create_spurious_result_dict(match: frozenset, matches: dict, result_type: str):
    t = tuple(match)
    inv_t = (t[1], t[0])
    if t in matches:
        similarity = matches[t]
    elif inv_t in matches:
        similarity = matches[inv_t]
    else:
        similarity = 0
    left_table: str = list(match)[0][0]
    if left_table.endswith('source'):
        clm_1 = list(match)[0][1]
        clm_2 = list(match)[1][1]
    else:
        clm_1 = list(match)[1][1]
        clm_2 = list(match)[0][1]
    return {'Column 1': clm_1, 'Column 2': clm_2, 'Similarity': similarity, 'Type': result_type}


def get_spurious_results_at_sizeof_ground_truth(matches: dict, golden_standard: GoldenStandardLoader):
    matches_at_sizeof_ground_truth = list(map(lambda m: frozenset(m), list(matches.keys())))[:golden_standard.size]
    spurious_results = []
    for expected_match in golden_standard.expected_matches:
        if expected_match not in matches_at_sizeof_ground_truth:
            spurious_results.append(create_spurious_result_dict(expected_match, matches, 'False Negative'))
    for detected_match in matches_at_sizeof_ground_truth:
        if detected_match not in golden_standard.expected_matches:
            spurious_results.append(create_spurious_result_dict(detected_match, matches, 'False Positive'))
    return spurious_results
