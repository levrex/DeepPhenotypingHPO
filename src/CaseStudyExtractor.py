## argv[]

## scraping - connect to library
## PhenoTool= 'ncr', 'txt2hpo', 'clinphen'
## entity_linking = True or False
## typo correction
## 
# python src/CaseStudyExtractor.py -h
# python src/CaseStudyExtractor.py -html PhenoTool/results/downloaded_html/Orenstein_2017_Wiley.html
# python src/CaseStudyExtractor.py https://onlinelibrary.wiley.com/doi/full/10.1111/cge.12930 -login 1
# python src/CaseStudyExtractor.py https://onlinelibrary.wiley.com/doi/full/10.1111/cge.12930 -login 1 -el 1 
# python src/CaseStudyExtractor.py https://www.wjgnet.com/1007-9327/full/v26/i15/1841.htm -intercept 1 1 1

import sys
sys.path.append('../')
sys.path.append('../src/')

import argparse
from argparse import RawTextHelpFormatter

#import DeepPhenotyping_functions as func # initializing this script takes alot of time, hence we put it in the main function

# Count the arguments
#print(eqal)
def scraping(URL, LOGIN, HTML, TIME):
    """
    URL = link to case study
    LOGIN = indicates whether or not to login to library to access papers behind paywall 
    HTML = path to downloaded html file
    TIME = appoint time to ensure a successfull connection in case you have a weak wifi (default:10)
    
    optional:
        LIBRARY = library-specific link
    """
    LIBRARY = 'https://login.proxy.library.uu.nl/login?auth=uushibboleth&url='
    
    if LOGIN == 1:
        try:
            d = func.getLoginData(login_file="../login_details.txt") # run from root
            func.set_login(d['USERNAME'],d['PASSWORD']) # Connect to Utrecht University
        except:
            try:
                d = func.getLoginData(login_file="login_details.txt") # run from root
                func.set_login(d['USERNAME'],d['PASSWORD']) # Connect to Utrecht University
            except:    
                print("Couldn't find any login data (no file named login_details.txt): make sure to place login details in same folder as were you run the script!")
        soup = func.scrapingCaseStudyLOGIN(URL, LIBRARY=LIBRARY, TIME=TIME)
    elif HTML != 'None':
        soup = func.scrapingCaseStudyHTML(HTML)
    else :
        soup = func.scrapingCaseStudy(URL, TIME=TIME)
    return soup

def screening(title, PARAMS=[3, 3, 5], SCR='clinphen'):
    """
    Rudimentary screening for phenotypic rich regions. Currently, this function only supports Clinphen.
    
    Input:
        title = title of case study
        PARAMS = Parameters for scanning text for phenotypic-rich regions (inspired by alignment with multiple ORFs)
        SCR = phenotypic screening algorithm (Clinphen is generally the best for the first rudimentary screening)
    """
    BIN_SIZE, MIN_POWER, FRAMES = PARAMS
    try : 
        BIN_SIZE = int(BIN_SIZE)
        MIN_POWER = int(MIN_POWER)
        FRAMES = int(FRAMES)
    except:
        print('For the intercept flage (-intercept) you are required to provide integers (BIN_SIZE, MIN_POWER and FRAMES)')
    
    items, first_intercept, lines = func.clinphen('results/%s/0_raw/Main_text_%s.html' % (title, title),'data', extensive=False)
    #print(type(first_intercept))
    
    parsed_doc = [item for sublist in lines for item in sublist]

    txt, d_phenotype = func.first_screening(parsed_doc, first_intercept, BIN_SIZE, MIN_POWER, FRAMES)

    with open("results/%s/1_extractions/Annotated_%s_%s.html" % (title, SCR, title), "w", encoding="utf-8") as file:
        file.write(txt)
    
    print('\nFinished Extraction + Screening!')
    print('Results are written to results/%s' % (title))
    return 

def main(arguments):
    """
    Apply Case study Extractor on provided link (URL). 
    
    
    """
    d = vars(arguments)
    URL = d['url']
    LOGIN = d['login']
    HTML = d['html']
    SCREEN = d['screenshots']
    PHENO = d['pheno']
    INTER = d['intercept']
    STRICT = d['strict']
    TIME = d['time']
    ENTITY_LINKING = d['el']
    #print(eqal)
    print('Vars:', d)
    
    print('SCREEN:', SCREEN)
    #print(SCREEN)
    if SCREEN == 1:
        SCREEN = bool(SCREEN)
    print('SCREEN:', SCREEN)
    # Scraping
    soup = scraping(URL, LOGIN, HTML, TIME)
    
    title= soup.title.string
    title = title[:50].strip().replace(' ', '_').replace(':', '').replace(';', '')
    func.createFolderStructure(title)
    
    print('Processing paper:', title)
    
    # Parsing
    if HTML != 'None': # we don't have a URL in case of downloaded html
        new_soup = func.parseCaseStudy(soup, title, 'google.com', screenshots=SCREEN, remove_accent=True) # remove_accent = True
    else :
        new_soup = func.parseCaseStudy(soup, title, URL, screenshots=SCREEN, remove_accent=True) # remove_accent = True
    
    # Screening
    screening(title, PARAMS=INTER)
    
    # Phenotyping
    df_hpo, parsed_doc = func.phenotypeCaseStudy(new_soup, title, pheno=PHENO, stringent=bool(d['strict']))
    
    # Annotating
    df_pheno, d_pat = func.annotateCaseStudy(title, pheno=PHENO, entity_linking=ENTITY_LINKING)
    
    # Generate Quality report
    func.write_HTML_report(title, phenotyper=PHENO)

if __name__ == '__main__':
    # CaseStudyExtractor.py -l www.google.com
    parser = argparse.ArgumentParser(description='Case Study Extractor', formatter_class=RawTextHelpFormatter)
    
    ## Scraping variables
    parser.add_argument('url', type=str, nargs='?',
                        help='url link to the case study')
    parser.add_argument('-login',  type=int, nargs='?', const=0, default=0,
                        help='access paper through public library account (provide login_details.txt)')
    parser.add_argument('-html',  type=str, nargs='?', default='None',
                        help='specify location of downloaded html')
    parser.add_argument('-time',  type=int, nargs='?', const=0, default=10,
                        help='appoint time to ensure a successfull connection in case you have a weak wifi (default:10)')
    
    # screenshots - by default don't make any screenshots (takes alot of time)
    parser.add_argument('-screenshots', nargs='?', const=0, type=int, default=0, help='take screenshots') 
    
    ## First Screening variables
    parser.add_argument("-intercept", nargs=3, default=[3, 3, 5],
                        help='Parameters for scanning text for phenotypic-rich regions, e.g: -intercept 3 3 5\n\tBIN_SIZE: number of subsentences to qualify as 1 region (default 3)\n\tMIN_POWER: minimal number of phenotypes required to qualify as phenotypic rich region (default 3)\n\tFRAMES: Number of "open" reading frames to alleviate the bias of initial binning (default 5)\n')
    
    ## Phenotyping variables
    parser.add_argument('-pheno',  type=str, nargs='?', const=0, default='ncr',
                        help='the HPO-extraction tool to employ (available options: ncr, clinphen, txt2hpo)')
    parser.add_argument('-strict',  type=int, nargs='?', const=0, default=0,
                        help='perform a stringent entity linking. Setting this parameter to 1 wil disrupt the reign of the patient within the narrative more easily.')
    
    ## Annotation variables
    parser.add_argument('-el',  type=int, nargs='?', const=0, default=0,
                        help='create phenotypic patient profiles for deep phenotyping (Warning: experimental)')
    
    #parser.add_argument('--foo', help='foo help')
    
    #parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                    const=sum, default=max,
    #                    help='sum the integers (default: find the max)')
    if len(sys.argv) > 2:
        import DeepPhenotyping_functions as func # initializing this script takes alot of time, hence we put it in the main function
    args = parser.parse_args()
    
    main(args)
    #print(args.accumulate(args.integers))
