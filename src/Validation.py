import numpy as np
import re

def table_to_phenodict(df):
    """
    Deduce table to a dictionary of phenotypes (initial format)
    
    Important to note: Ignores phenotypes that have no location
        (These are inferred parent features)
    
    Input:
        df = imported pandas dataframe, that is labeled
    Output:
        new_d = dictionary with phenotypes (a.k.a. d_pheno) and
            associated parameters, like location/ negation etc..
            
    """
    new_d = {}
    for k, g in df.iterrows():
        ix = g['index']
        if np.isnan(ix) == False:
            if ix in new_d.keys():
                new_d[ix].append(g.to_dict())
            else :
                new_d[ix] = [g.to_dict()]
    return new_d

def get_lbl(df_tracker, i):
    return df_tracker[((df_tracker['HPO']==i['hp_id']) & (df_tracker['start']==i['start']) & (df_tracker['end']==i['end']))]['LBL'].iloc[0]

def annotate_text_validation(title, parsed_doc, df_tracker, pheno):
    """
    Annotate text to get more insight in the validation process
    
    Input:
        title = title of case study
        parsed_doc = segmented content of the case study
        df_tracker = reporting the validation results
        pheno = phenotyper extraction tool
    """
    start_str = '<span style="color:red">'
    end_str = '</span>'
    
    # ENTITY LINKING

    # remove semicolon if table
    df_tracker.fillna('') 
    df_tracker = df_tracker[~(df_tracker['index']=='')]
    df_tracker = df_tracker[df_tracker['section']!='Unassigned'] # easy fix
    l_sect = ['Main text', 'Captions']
    mask = [ 'Table File' in key for key in list(df_tracker['section'].unique())]
    l_sect.extend(np.array(df_tracker['section'].unique())[np.array(mask)])
    print('Elements found: ', l_sect)
    for sect in l_sect:
        new_lines = []
        sub_df = df_tracker.copy()
        
        if sect == 'Captions':
            sub_df = sub_df[(sub_df['section'] == 'Captions')]
        elif 'Table File' in sect:
            sub_df = sub_df[(sub_df['section'].str.contains(sect))]
        elif sect == 'Main text' :
             sub_df = sub_df[~((sub_df['section'] == 'Captions') | (sub_df['section'].str.contains('Table File')))]
        if len(sub_df) > 0:
            parsed_list = parsed_doc[sect]
        else :
            continue
        d_phenotype = table_to_phenodict(sub_df)
        for ix, sent in enumerate(parsed_list): 
            txt = parsed_list[ix]
            passing = False
            if ix in d_phenotype:
                d_sort = sorted(d_phenotype[ix], key = lambda j: j['end'], reverse=True)
                for i in d_sort:
                    label = get_lbl(sub_df, i)
                    if label == 'FP':
                        start_str = '<span style="color:red" title="%s" >'  % ('HPO: ' + i['hp_id'] + '\nLABEL: ' + label + '\nCONF: ' + str(i['score']) + '\nPAT: ' + i['pat_id'] + '\nNEG: ' + str(i['negated']) + '\nREL: ' + str(i['relevant']) + '\nSECTION: ' + i['section'] ) # i['pat_id']
                    elif label == 'TP' : 
                        start_str = '<span style="color:lime" title="%s" >'  % ('HPO: ' + i['hp_id'] + '\nLABEL: ' + label + '\nCONF: ' + str(i['score']) + '\nPAT: ' + i['pat_id'] + '\nNEG: ' + str(i['negated']) + '\nREL: ' + str(i['relevant']) + '\nSECTION: ' + i['section'] ) 
                    start_int = int(i['start'])
                    end_int = int(i['end'])
                    txt = txt[:start_int] + ' ' + start_str + txt[start_int:end_int]  + end_str + ' ' + txt[end_int:] 
            new_lines.append(txt)
        sect = re.sub(r" ", "_", sect)
        with open("results/%s/3_annotations/Colored_%s_Classification_%s.html" % (title, sect, pheno), "w", encoding="utf-8") as file:
            file.write(' '.join(new_lines))
    return 