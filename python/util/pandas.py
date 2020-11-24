import pandas as pd


def are_dataframe_indices_compatible(df_a: pd.DataFrame,
                                     df_b: pd.DataFrame,
                                     indices_must_be_disjunct: bool = True) -> bool:
    """
    We test two dataframes for compatibility based on index and column dtypes. Note: there is
    pd.testing.assert_frame_equal(), but this also compares the contents.
    :param df_a:
    :param df_b:
    :param indices_must_be_disjunct: Don't care about indices being disjunct. You probably want to use this whenever you
                                     would use ignore_index in pd.concat.
    :return:
    """
    # index should be of same type
    is_multiindex = lambda idx: isinstance(idx, pd.MultiIndex)
    if is_multiindex(df_a.index) != is_multiindex(df_b.index):
        return False

    # index dtypes should be the same
    get_multiindex_dtypes = lambda idx: [idx.get_level_values(i).dtype for i in range(len(idx.levels))]
    if is_multiindex(df_a.index):
        df_a_column_dtypes = get_multiindex_dtypes(df_a.index)
        df_b_column_dtypes = get_multiindex_dtypes(df_b.index)
        if df_a_column_dtypes != df_b_column_dtypes:
            return False
    else:
        if df_a.index.dtype != df_b.index.dtype:
            return False

    # index names should match
    if df_a.index.names != df_b.index.names:
        return False

    # indexes should be disjunct in some cases
    if indices_must_be_disjunct and not df_a.index.intersection(df_b.index).empty:
        return False

    # column index should be of same type
    if is_multiindex(df_a.columns) != is_multiindex(df_b.columns):
        return False

    # index dtypes should be the same
    if is_multiindex(df_a.columns):
        df_a_column_dtypes = get_multiindex_dtypes(df_a.columns)
        df_b_column_dtypes = get_multiindex_dtypes(df_b.columns)
        if df_a_column_dtypes != df_b_column_dtypes:
            return False
    else:
        if df_a.columns.dtype != df_b.columns.dtype:
            return False

    # column names must match
    if df_a.columns.names != df_b.columns.names:
        return False

    return True