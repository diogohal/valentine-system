
def acc_format(continuous_count, discrete_count, ks_acc, ad_acc, chi_acc, g_acc):
    if continuous_count > 0:
        print("{}\t[KS: {:.3f} | AD: {:.3f}],\t".format(continuous_count, ks_acc, ad_acc), end='')
    else:
        print("0\t[KS: ----- | AD: -----],\t", end='')
    if discrete_count > 0:
        print("{}\t[CHI: {:.3f} | G: {:.3f}]".format(discrete_count, chi_acc, g_acc))
    else:
        print("0\t[CHI: ----- | G: -----]")


def persistent_acc(changes_df, year_df, i, top, sig_thresh):
    # Get persistent names
    filtered_df = changes_df[(changes_df[str(i)] == True) & (changes_df[str(i+1)] == True)]

    continuous_count = discrete_count = 0
    ks_acc = ad_acc = chi_acc = g_acc = 0
    no_comp = 0

    for col in filtered_df['columns']:
        # Get persistent column comparation result
        col_df = year_df[(year_df['attr1'] == col)]

        # Check if comparison is performed and count if cont. or disc.
        if len(col_df) == 0:
            #print(col)
            no_comp += 1
        else:
            if 'KS' in col_df['test'].values or 'AD' in col_df['test'].values:
                continuous_count += 1
            if 'CHISQ' in col_df['test'].values or 'G' in col_df['test'].values:
                discrete_count += 1

            # Filter top values
            col_df = col_df[col_df['p-value'] >= sig_thresh]
            col_df = col_df.sort_values('p-value', ascending=False).head(int(top))

            # Accuracy for each test
            if col in col_df['attr2'].values:
                test_df = col_df[col_df['attr2'] == col]
                # Account which test compare it right
                if 'KS' in test_df['test'].values:
                    ks_acc += 1
                if 'AD' in test_df['test'].values:
                    ad_acc += 1
                if 'CHISQ' in test_df['test'].values:
                    chi_acc += 1
                if 'G' in test_df['test'].values:
                    g_acc += 1

    # -1 for no comparison found
    if continuous_count > 0:
        ks_acc = ks_acc/continuous_count
        ad_acc = ad_acc/continuous_count
    else: ks_acc = -1; ad_acc = -1
    if discrete_count > 0:
        chi_acc = chi_acc/discrete_count
        g_acc = g_acc/discrete_count
    else: chi_acc = -1; g_acc = -1

    #print(f"Comparisons not performed: {no_comp}")
    print(f"Persistent\t({len(filtered_df['columns'])}):\t", end='')
    acc_format(continuous_count, discrete_count, ks_acc, ad_acc, chi_acc, g_acc)
    return ks_acc, ad_acc, chi_acc, g_acc


def new_acc(changes_df, year_df, i, sig_thresh):
    # Get new column names
   filtered_df = changes_df[(changes_df[str(i)] == False) & (changes_df[str(i+1)] == True)]

   continuous_count = discrete_count = 0
   ks_acc = ad_acc = chi_acc = g_acc = 0
   no_comp = 0

   for col in filtered_df['columns']:
       # Get new column comparation result
       col_df = year_df[(year_df['attr2'] == col)]

       # Check if comparison is performed and count if cont. or disc.
       if len(col_df) == 0:
           #print(col)
           no_comp += 1
       else:
           if 'KS' in col_df['test'].values or 'AD' in col_df['test'].values:
               continuous_count += 1
           if 'CHISQ' in col_df['test'].values or 'G' in col_df['test'].values:
               discrete_count += 1

           # +1 if comparison is performed and all is lower the sig_thresh
           # i.e. there is no equivalent in previous year
           tests = set(col_df['test'].values)
           col_df = col_df[col_df['p-value'] >= sig_thresh]
           for test in tests:
               #print(col_df[col_df['test'] == test].to_string())
               if len(col_df[col_df['test'] == test]) == 0:
                  if test == 'KS':
                       ks_acc += 1
                  elif test == 'AD':
                      ad_acc += 1
                  elif test == 'CHISQ':
                      chi_acc += 1
                  elif test == 'G':
                      g_acc += 1

   # -1 for no comparison found
   if continuous_count > 0:
       ks_acc = ks_acc/continuous_count
       ad_acc = ad_acc/continuous_count
   else: ks_acc = -1; ad_acc = -1
   if discrete_count > 0:
       chi_acc = chi_acc/discrete_count
       g_acc = g_acc/discrete_count
   else: chi_acc = -1; g_acc = -1

   #print(f"Comparisons not performed: {no_comp}")
   print(f"New\t\t({len(filtered_df['columns'])}):\t", end='')
   acc_format(continuous_count, discrete_count, ks_acc, ad_acc, chi_acc, g_acc)
   return ks_acc, ad_acc, chi_acc, g_acc

def missing_acc(changes_df, year_df, i, sig_thresh):
    # Get missing column names
   filtered_df = changes_df[(changes_df[str(i)] == True) & (changes_df[str(i+1)] == False)]

   continuous_count = discrete_count = 0
   ks_acc = ad_acc = chi_acc = g_acc = 0
   no_comp = 0

   for col in filtered_df['columns']:
       # Get missing column comparation result
       col_df = year_df[year_df['attr1'] == col]

       # Check if comparison is performed and count if cont. or disc.
       if len(col_df) == 0:
           #print(col)
           no_comp += 1
       else:
           if 'KS' in col_df['test'].values or 'AD' in col_df['test'].values:
               continuous_count += 1
           if 'CHISQ' in col_df['test'].values or 'G' in col_df['test'].values:
               discrete_count += 1

           # +1 if comparison is performed and all is lower the sig_thresh
           # i.e. there is no equivalent in next year
           tests = set(col_df['test'].values)
           col_df = col_df[col_df['p-value'] >= sig_thresh]
           for test in tests:
               if len(col_df[col_df['test'] == test]) == 0:
                  if test == 'KS':
                       ks_acc += 1
                  elif test == 'AD':
                      ad_acc += 1
                  elif test == 'CHISQ':
                      chi_acc += 1
                  elif test == 'G':
                      g_acc += 1

   # -1 for no comparison found
   if continuous_count > 0:
       ks_acc = ks_acc/continuous_count
       ad_acc = ad_acc/continuous_count
   else: ks_acc = -1; ad_acc = -1
   if discrete_count > 0:
       chi_acc = chi_acc/discrete_count
       g_acc = g_acc/discrete_count
   else: chi_acc = -1; g_acc = -1

   #print(f"Comparisons not performed: {no_comp}")
   print(f"Missing\t\t({len(filtered_df['columns'])}):\t", end='')
   acc_format(continuous_count, discrete_count, ks_acc, ad_acc, chi_acc, g_acc)
   return ks_acc, ad_acc, chi_acc, g_acc

