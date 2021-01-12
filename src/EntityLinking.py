import re


import spacy
from negspacy.negation import Negex
from spacy.matcher import Matcher
import ast

global nlp
nlp = spacy.load("en_core_sci_sm")
negex = Negex(nlp, language = "en_clinical_sensitive") # , chunk_prefix = ["no"] en_clinical_sensitive   en_clinical
nlp.add_pipe(negex)

## PATIENT IDENTIFICATION

def search_entities(d_patient_ids, iteration, pat, d_section, ambigious=False):
    """
    Assign contextual properties to the found entities.
    
    Input: 
        d_patient_ids = dictionary with all patients that were found in text
        iteration = parameters for iterations. Tuple consisting of the line number
            and the concerned line.
        pat = regex pattern to search for in case studies
        d_section = dictionary storing the coordinates of every
                section in the case study.
        ambigious = boolean indicating the connotations of the flag. If the 
            phenotype is important then this boolean is set to False (default) 
            
    Output:
        d_patient_ids = updated dictionary with patients
    """
    ix, line = iteration 
    pattern = re.compile(pat)
    r = pattern.search(line)
    while r:
        d_match = {}
        if d_section != {}:
            section = infer_property(ix, d_section)
        else :
            section = 'Unassigned'
        d_match['pattern'] = pat
        d_match['matched'] = r.group()
        d_match['line'] = ix
        d_match['start'] = r.start()
        d_match['end'] = r.end() - 1
        d_match['pat_id'] = r.groups()[0]
        d_match['is_in_title'] = is_in_title(r.groups()[0], d_section)
        d_match['section'] = section
        d_match['ambigious'] = ambigious
        if ix in d_patient_ids.keys():
            d_patient_ids[ix].append(d_match)
        else : 
            d_patient_ids[ix] = [d_match]
        r = pattern.search(line,r.start() + 1)
    return d_patient_ids

def escape_special_characters(l_patterns):
    """
    Escape special characters to ensure that the regex works properly.
    Also make the words lowercase and add brackets to ensure that the
    entity is captured!
    
    Input:
        l_pattern = list of user provided patterns to extract entities 
    Output:
        new_patterns = list with updated patterns for regex
    """
    new_patterns = []
    
    # Special characters to escape
    
    for pat in l_patterns:
        pat = pat.translate(str.maketrans({"-":  r"\-",
                                          "]":  r"\]",
                                          "\\": r"\\",
                                          "^":  r"\^",
                                          "$":  r"\$",
                                          "*":  r"\*",
                                          ".":  r"\."}))
        pat = pat.lower()
        new_patterns.append('(' + pat + ')')
    print('Updated entities:', new_patterns)
    return new_patterns
    
#d_patient_ids['ix'] = [{'pattern' :  , 'matched' : , 'line' : , 'start'}]

def identify_patient(lines, stringent=True, l_patterns=[]):
    """
    Search for patients in the text & extract said patient
    
    ToDo: once a patient code is found. Ensure that those independent patient codes
    are also found. For example, you want 2 to be recognized, if 1 is recognized:
        1 'individual #193 has a headache' (found)
        2 'a headache is found in #193' (found!)
    
    Input: 
        lines = segmented content of case study
        stringent = boolean indicating whether or not to perform a stringent patient linking
        l_patterns = user provided list of entities (e.g. V:6). If a user doesn't 
            provide a list of entities -> then use default rules to initiate unsupervised
            entity linking!
    Output:
        d_patient_ids = dictionary with all patients that were found in text
    """
    if l_patterns == []: # if no patterns provided -> then perform unsupervised entity screening
        l_patterns = ['(?:individual|patient|proband|parent|member|subject|case|family)(?:,?)\s([A-z]+\d+)', # combination characters + numbers
                  '(?:individual|patient|proband|parent|member|subject|case|family)(?:,?)\s(\#[A-z]*\d+)',  # Hashtags
                 '(?:individual|patient|proband|parent|member|subject|case|family)(?:,?)\s([i,v,x]+)', # Roman numbers
                 '((?:individual|patient|proband|parent|member|subject|case|family)(?:,?)\s\d+)', # just a number
                 ]
    else :
        l_patterns = escape_special_characters(l_patterns)
    l_fake = ['[^a-z](cousin|parent|mom|mother|dad|father|grandmother|grandfather|grandparent|brother|sister|sibling|uncle|aunt|nephew|niece|son|daughter|grandchild)(?:,?\s)', # referring to family member
              '[^a-z](literature|report|studies)(?:,?\s)', # referring to previous findings
              '[^a-z](this study)(?:,?\s)',
              '[^a-z](families|cases|individuals|patients|parents|members|subjects|probands)(?:,?\s)' # plural = almost always generic
             ]
    #l_paragraphs = ['(\n)']
    d_patient_ids = {}
    d_section = get_sections(lines)
    pat_found = False
    
    for ix, line in enumerate(lines):
        line = line.lower()
        for pat in l_patterns: # 
            d_patient_ids = search_entities(d_patient_ids, (ix, line), pat, d_section, ambigious=False)
            if ix in d_patient_ids.keys():
                if any([i['ambigious'] for i in d_patient_ids[ix]])==False: # no ambigious # True
                    pat_found = True
            
        if '<br>' in line and stringent: 
            if ix not in d_patient_ids.keys(): # assume we can disrupt the reign of whathever patient was prev. found
                d_patient_ids = search_entities(d_patient_ids, (ix, line), '(\<br\>)', d_section, ambigious=True)
            elif (any([i['is_in_title'] for i in d_patient_ids[ix]])!=True): # Only disrupt the reign if pat is not found in title
                d_patient_ids = search_entities(d_patient_ids, (ix, line), '(\<br\>)', d_section, ambigious=True)
        #for fake_pat in l_fake: # 
        #    d_patient_ids = search_entities(d_patient_ids, (ix, line), fake_pat, d_section, ambigious=True)
    
    #print(d_patient_ids)
    #print('PAT FOUND:', pat_found)
    if pat_found == False:
        #print('No patients found')
        d_patient_ids = {0: [{'pattern' : '' , 'matched' : '', 'line' : 0, 
                                 'start': 0, 'end' : 1, 'pat_id' : 'all', 'is_in_title' : True, 
                                 'section' : 'None', 'ambigious' : True}]}
        #print(eql)
        #print(len(d_patient_ids))
    return d_patient_ids

## EXTRACT CONTEXTUAL PROPERTIES

def get_sections(lines):
    """
    Build a dictionary indicating the location of the sections
    within the paper
    
    Input:
        lines = segmented content of case study (list)
    Output:
        d_section = dictionary storing the coordinates of every
                section in the case study.
    """
    d_section = {}
    for ix, line in enumerate(lines):
        #print(line)
        matches = re.findall(r"<h[1-6]>([^<]*)</h[1-6]>", line)
        
        if matches != []:
            d_section[ix] = matches[0]
    return d_section

def is_in_title(pat, d_section):
    """
    Check if patient identifier is in title. To recognize whenever
    a paragraph is referring to a specific patient
    
    Input:
        pat = identifier of patient
        d_section = dictionary storing the coordinates of every
            section in the case study.
    """
    if (any([True if pat in i else False for i in d_section.values()])==True):
        return True
    else :
        return False
    
def infer_property(location, d):
    """
    Assign overarching elements to the provided location.
    
    Input:
        location = line number (after parsing)
        d = dictionary with overarching elements
        
    Output:
        d[last_key] = corresponding value for key
    """
    last_key = 0
    for key in sorted(d):
        #print("%s: %s" % (key, d[key]))
        if key > location:
            if last_key in d.keys(): 
                return d[last_key]
            else :
                return 'Unassigned'
        else :
            last_key = key
    return 'Unassigned'

def is_negated(line, phenotype, start_hpo):
    """
    Goal: Reduce type I error (decrease False Positives)
    
    Check if found phenotype is negated by leveraging Negex & spacy!
    
    We built a custom matcher every single time, to ensure 
    that we capture even the most rare phenotypes 
    rather than depending on the quality of pretrained entity linking modules.
    
    Potentially multiple matches are found in the same sentence. In this
    case we look at the location of both matches, and ensure that we
    assign a label by taking the coordinates into account
    
    We also check if the negation is not mentioned in a chunk
    
    Input: 
        line = single line from case study
        phenotype = intercepted phenotype (string)
        start_hpo = starting position of phenotype in line
    """
    
    global nlp
    
    in_chunk = False
    
    matcher = Matcher(nlp.vocab)
    pheno_pat = [ast.literal_eval('{"ORTH": "' + str(i.lower().replace('"', '') +'"}')) for i in phenotype.split(' ')]
    matcher.add("phenotype", None, pheno_pat)
    
    doc = nlp(line)

    d_match = {}
    matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        d_match[start] =span._.negex
    
    # CHECK IF NO NEGATION IN CHUNK
    neg_in_chunk = ['no']
    
    for chunk in doc.noun_chunks:
        #E.g: check if 'no' is in chunk! (only counts if 'no' is not a part of HPO-concept)
        if phenotype.lower() in chunk.text and any([i in chunk.text for i in neg_in_chunk]) \
        and any([i in phenotype.lower() for i in neg_in_chunk]) != True: # check if this is correct!
            in_chunk = True
            break
            
    if in_chunk:
        #print('FOUND IN CHUNK!')
        return True
    elif d_match != {}: 
        return d_match[min(d_match.keys(), key=lambda x:abs(x-start_hpo))] # span._.negex
    else :
        return False

def get_flags(line, *flagsets):
    """
    Acquire all flags found in the line.
    
    Input:
        line = single line from case study
        
        flagsets = set with patterns (flags) to capture important contextual 
            properties with regard to the found phenotype
    """
    #line = add_lemmas(set(line), screening)
    returnFlags = set()
    for flagset in flagsets:
        #flagset = add_lemmas(set(flagset), screening)
        for word in flagset:
            if word in line: returnFlags.add(word)
    return returnFlags

def predict_relevancy(section, pat_present):
    """
    Predict whether or not the found phenotype is actually relevant!
    
    ToDo: if ambigious marker than it should not be relevant?
    
    Input:
        section = section in case study where phenotype is found
    Output:
        relevancy = boolean indicating whether or not the phenotype is relevant
    """
    relevancy = False  # Physical examination
    if section != 'unassigned': 
        score = 1*('case' in section) + 1 * ('description' in section) + 1 * ('presentation' in section) + 1 * ('summary' in section) + \
        1 * ('examination' in section) + 1 * ('physical' in section)  + 1 * ('laboratory' in section) + \
        -1 * ('acknowledgments' in section) + -1 * ('introduction' in section) + -1 * ('discussion' in section)  + \
        -1 * ('background' in section)
        if score > -1 and pat_present: # score has to be positive
            relevancy = True
    return relevancy

def predict_patient(ix, d_patient_ids, flags_found):
    """
    Assign patient id by retrieving the most recently mentioned patient
    
    Check if it is not obstructed by an ambigious label
    
    ToDo: check if in title (than more power - at least till next paragraph, we are most certain 
    that it corresponds to mentioned patient)
    
    Input:
        ix = line number
        d_patient_ids = dictionary with all patients that were found in text
    """
    entities = infer_property(ix, d_patient_ids)
    pat_id = 'None' 
    if type(entities) == list:
        for ent in entities:
            if ent['ambigious'] == False & flags_found == False: # and there is no seperator between the two 
                pat_id = ent['pat_id']
                break
            elif flags_found == True: # maybe too strict
                break
    return pat_id
   
## LINK PHENOTYPES TO CONTEXTUAL PROPERTIES
    
def mass_flagging(lines, d_phenotype, d_patient_ids):
    """
    Evaluate the context of the found phenotypes. 
    
    Which patient is referred to? Do we find any red flags?
    
    Input:
        lines = segmented content of case study
        d_phenotype = Collection of captured phenotypes 
        d_patient_ids = dictionary with all patients that were found in text
    Output:
        d_phenotype = Updated phenotype dictionary. Now with extra contextual
            properties like surrounding flags, relevancy and corresponding
            patient & location in paper (section). 
    """
    negative_flags = ["no", "not", "none", "negative", "non", "never", "without", "denies"]
    family_flags = ["cousin", "parent", "mom", "mother", "dad", "father", "grandmother", "grandfather", "grandparent",  "brother", "sister", "sibling", "uncle", "aunt", "nephew", "niece", "son", "daughter", "grandchild", "families", "cases", "individuals", "patients", "parents", "members", "subjects", "probands"] # "family",
    healthy_flags = ["normal"]
    #disease_flags = ["associated", "gene", "recessive", "dominant", "variant", "cause", "literature", "individuals"]
    history_flags = ['previously', 'previous']
    generic_flags= ['report', 'study', 'literature', 'studies']
    uncertain_flags = ['may', 'indicate', 'suggest', 'suggest', 'indicative', 'associated', 'association', 'presumed to be', 'probably', 'sign', 'signs', 'uncertain', 'uncertainty']

    d_section = get_sections(lines)
    
    for ix, line in enumerate(lines): 
        line = line.lower() # lower
        passing = False
        if ix in d_phenotype:
            d_sort = sorted(d_phenotype[ix], key = lambda j: j['end'], reverse=True)
            
            if d_section != {}:
                section = infer_property(ix, d_section)
            else :
                section = 'Unassigned'
            for i in d_sort:
                #print(i)
                #d_sort['negated']
                flags = get_flags(line.split(" "), negative_flags, family_flags, healthy_flags, history_flags, uncertain_flags, generic_flags)
                pheno_name = line[int(i['start']):int(i['end'])]
                i['negated'] = is_negated(line, pheno_name, int(i['start'])) # hpo_id 
                i['flags'] = flags
                i['section'] = section
                if len(d_patient_ids) != 1:
                    i['pat_id'] = predict_patient(ix, d_patient_ids, len(flags)!=0) # all
                else :
                    #print('No patients found in text!')
                    i['pat_id'] = predict_patient(ix, d_patient_ids, len(flags)!=0)
                i['relevant'] = predict_relevancy(section.lower(), i['pat_id'])
    #print(d_phenotype)
    return d_phenotype

