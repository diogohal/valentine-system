import math

from .golden_standard import GoldenStandardLoader
from .utils import one_to_one_matches


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


def persistent_acc(normalized_matches: dict,
                   golden_standard: GoldenStandardLoader,
                   top: int = 10,
                   sig_thresh: float = 0.95,
                   source_columns: list[str] = None,
                   target_columns: list[str] = None):
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

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    source_set = set(source_columns) if source_columns is not None else set(normalized_matches.keys())
    if target_columns is not None:
        target_set = set(target_columns)
    else:
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # Persistent columns: present in both source and target
    persistent_cols = source_set.intersection(target_set)

    continuous_count = 0
    discrete_count = 0
    total_count = 0
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_persistent_acc = 0

    for col1 in persistent_cols:
        matches_dict = normalized_matches.get(col1, {})
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
                break

    total_count = continuous_count + discrete_count
    if total_count > 0:
        total_persistent_acc = (ks_acc + ad_acc + chi_acc + g_acc) / total_count
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

    return {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_persistent_acc}


def new_acc(normalized_matches: dict,
            golden_standard: GoldenStandardLoader,
            sig_thresh: float = 0.95,
            source_columns: list[str] = None,
            target_columns: list[str] = None):
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

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    source_set = set(source_columns) if source_columns is not None else set(normalized_matches.keys())
    if target_columns is not None:
        target_set = set(target_columns)
    else:
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # New columns: only in target
    new_cols = target_set - source_set

    print(f'--------------------------------------------------')
    print(f"New columns: {new_cols}")
    print(f'Normalized matches items: {normalized_matches.items()}')

    continuous_count = 0
    discrete_count = 0
    total_count = 0
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_new_acc = 0

    for col in new_cols:
        # Gather all comparisons where this target column appears
        tests_performed = set()
        per_test_has_sig = {}

        for _, matches_dict in normalized_matches.items():
            if col not in matches_dict:
                continue
            tests = matches_dict[col]
            for test_name, p_value in tests.items():
                tests_performed.add(test_name)
                if test_name not in per_test_has_sig:
                    per_test_has_sig[test_name] = False
                if p_value >= sig_thresh:
                    per_test_has_sig[test_name] = True

        has_cont = 'KS' in tests_performed or 'AD' in tests_performed
        has_disc = 'CHISQ' in tests_performed or 'G' in tests_performed
        if has_cont:
            continuous_count += 1
        if has_disc:
            discrete_count += 1

        # Accuracy for new columns: correct when no significant equivalent exists
        if 'KS' in tests_performed and not per_test_has_sig.get('KS', False):
            ks_acc += 1
        if 'AD' in tests_performed and not per_test_has_sig.get('AD', False):
            ad_acc += 1
        if 'CHISQ' in tests_performed and not per_test_has_sig.get('CHISQ', False):
            chi_acc += 1
        if 'G' in tests_performed and not per_test_has_sig.get('G', False):
            g_acc += 1

    total_count = continuous_count + discrete_count
    if total_count > 0:
        total_new_acc = (ks_acc + ad_acc + chi_acc + g_acc) / total_count
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

    return {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_new_acc}


def missing_acc(normalized_matches: dict,
                golden_standard: GoldenStandardLoader,
                sig_thresh: float = 0.95,
                source_columns: list[str] = None,
                target_columns: list[str] = None):
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

    Returns
    -------
    dict
        {'ks': accuracy, 'ad': accuracy, 'chi': -1, 'g': -1}
    """
    source_set = set(source_columns) if source_columns is not None else set(normalized_matches.keys())
    if target_columns is not None:
        target_set = set(target_columns)
    else:
        target_set = set()
        for matches_dict in normalized_matches.values():
            target_set.update(matches_dict.keys())

    # Missing columns: only in source
    missing_cols = source_set - target_set

    print(f'--------------------------------------------------')
    print(f"Missing columns: {missing_cols}")
    print(f'--------------------------------------------------')

    continuous_count = 0
    discrete_count = 0
    total_count = 0
    ks_acc = 0
    ad_acc = 0
    chi_acc = 0
    g_acc = 0
    total_missing_acc = 0

    for col in missing_cols:
        matches_dict = normalized_matches.get(col, {})

        tests_performed = set()
        per_test_has_sig = {}

        for _, tests in matches_dict.items():
            for test_name, p_value in tests.items():
                tests_performed.add(test_name)
                if test_name not in per_test_has_sig:
                    per_test_has_sig[test_name] = False
                if p_value >= sig_thresh:
                    per_test_has_sig[test_name] = True

        has_cont = 'KS' in tests_performed or 'AD' in tests_performed
        has_disc = 'CHISQ' in tests_performed or 'G' in tests_performed
        if has_cont:
            continuous_count += 1
        if has_disc:
            discrete_count += 1

        # Accuracy for missing columns: correct when no significant equivalent exists
        if 'KS' in tests_performed and not per_test_has_sig.get('KS', False):
            ks_acc += 1
        if 'AD' in tests_performed and not per_test_has_sig.get('AD', False):
            ad_acc += 1
        if 'CHISQ' in tests_performed and not per_test_has_sig.get('CHISQ', False):
            chi_acc += 1
        if 'G' in tests_performed and not per_test_has_sig.get('G', False):
            g_acc += 1

    total_count = continuous_count + discrete_count
    if total_count > 0:
        total_missing_acc = (ks_acc + ad_acc + chi_acc + g_acc) / total_count
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

    return {'ks': ks_acc, 'ad': ad_acc, 'chi': chi_acc, 'g': g_acc, 'total_acc': total_missing_acc}


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
