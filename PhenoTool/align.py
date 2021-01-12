#!/usr/bin/python3
"""
DESCRIPTION:
    Template code for the Dynamic Programming assignment in the Algorithms in Sequence Analysis course at the VU.
    
INSTRUCTIONS:
    Complete the code (compatible with Python 3!) upload to CodeGrade via corresponding Canvas assignment.
AUTHOR:
    Tjardo Maarseveen - tmn221
"""
import argparse
import pickle
def parse_args():
    "Parses inputs from commandline and returns them as a Namespace object."
    parser = argparse.ArgumentParser(prog = 'python3 align.py',
        formatter_class = argparse.RawTextHelpFormatter, description =
        '  Aligns the first two sequences in a specified FASTA\n'
        '  file with a chosen strategy and parameters.\n'
        '\n'
        'defaults:\n'
        '  strategy = global\n'
        '  substitution matrix = pam250\n'
        '  gap penalty = 2')
        
    parser.add_argument('fasta', help='path to a FASTA formatted input file')
    parser.add_argument('output', nargs='*', 
        help='path to an output file where the alignment is saved\n'
             '  (if a second output file is given,\n'
             '   save the score matrix in there)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
        help='print the score matrix and alignment on screen', default=False)
    parser.add_argument('-s', '--strategy', dest='strategy',
        choices=['global','semiglobal','local'], default="global")
    parser.add_argument('-m', '--matrix', dest='substitution_matrix',
        choices=['pam250','blosum62','identity'], default='pam250')
    parser.add_argument('-g', '--gap_penalty', dest='gap_penalty', type=int,
        help='must be a positive integer', default=2)
    args = parser.parse_args()
    args.align_out = args.output[0] if args.output else False
    args.matrix_out = args.output[1] if len(args.output) >= 2 else False
                      # Fancy inline if-else statements. Use cautiously!
                      
    if args.gap_penalty <= 0:
        parser.error('gap penalty must be a positive integer')
    return args

def load_substitution_matrix(name):
    "Loads and returns the specified substitution matrix from a pickle (.pkl) file."
    # Substitution matrices have been prepared as nested dictionaries:
    # the score of substituting A for Z can be found with subst['A']['Z']
    # NOTE: Only works if working directory contains the correct folder and file!
    
    with open('substitution_matrices/%s.pkl' % name, 'rb') as f:
        subst = pickle.load(f)
    return subst
    
    
def load_sequences(filepath):
    "Reads a FASTA file and returns the first two sequences it contains."
    
    seq1 = []
    seq2 = []
    with open(filepath,'r') as f:
        for line in f:
            if line.startswith('>'):
                if not seq1:
                    current_seq = seq1
                elif not seq2:
                    current_seq = seq2
                else:
                    break # Stop if a 3rd sequence is encountered
            else:
                current_seq.append(line.strip())
    
    if not seq2:
        raise Exception('Error: Not enough sequences in specified FASTA file.')
    
    seq1 = ''.join(seq1)
    seq2 = ''.join(seq2)
    return seq1, seq2
def align(seq1, seq2, strategy, substitution_matrix, gap_penalty):
    "Do pairwise alignment using the specified strategy and parameters."
    # This function consists of 3 parts:
    #
    #   1) Initialize a score matrix as a "list of lists" of the appropriate length.
    #      Fill in the correct values for the first row and column given the strategy.
    #        (local / semiglobal = 0  --  global = stacking gap penalties)
    #   2) Fill in the rest of the score matrix using Dynamic Programming, accounting
    #      for the selected alignment strategy, substitution matrix and gap penalty.
    #   3) Perform the correct traceback routine on your filled in score matrix.
    #
    # Both the resulting alignment (sequences with gaps and the corresponding score)
    # and the filled in score matrix are returned as outputs.
    #
    # NOTE: You are strongly encouraged to think about how you can reuse (parts of)
    #       your code between steps 2 and 3 for the different strategies!
    
    
    ### 1: Initialize
    M = len(seq1)+1
    N = len(seq2)+1
    score_matrix = []
    for i in range(M):
        row = []
        score_matrix.append(row)
        for j in range(N):
            row.append(0)
    
    if strategy == 'global':
        #####################
        # START CODING HERE #
        #####################
        score_matrix[0] = [ -gap_penalty*i for i in range(N)]
        for i in range(1,M):
            score_matrix[i][0]= -gap_penalty*i 
        pass    # Change the zeroes in the first row and column to the correct values.
        #####################
        #  END CODING HERE  #
        #####################
    
    
    ### 2: Fill in Score Matrix
 
    #####################
    # START CODING HERE #
    #####################
    # def dp_function(...):
    #     ...
    #     return ...
    #
    # for i in range(1,M):
    #     for j in range(1,N):
    #         score_matrix[i][j] = dp_function(...)
    if strategy == 'semiglobal':
        score_matrix[0] = [ 0 for i in range(N)]
        for i in range(1,M):
            score_matrix[i][0]= 0 
        pass    # Change the zeroes in the first row and column to the correct values.
    def dp_function(sm, i, j, strategy):
        """
        Input:
            The following items are required as input since
            sm = score_matrix
            i = index of first sequence
            j = index of second sequence
        """
        g = gap_penalty
        opt1 = sm[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]]
        opt2 = sm[i][j-1] - g
        opt3 = sm[i-1][j] - g
        if strategy == 'local':
            return max(opt1, opt2, opt3, 0)
        elif strategy == 'semiglobal' and (j == 0 or i == 0): 
            return 0
        else :
            return max(opt1, opt2, opt3)
    top_i = -1
    top_j = -1
    l_cand = []
    l_test = []
    for i in range(1-(strategy == 'semiglobal'),M):
        for j in range(1-(strategy == 'semiglobal'),N):
            # print(i, j)
            score_matrix[i][j] = dp_function(score_matrix, i, j, strategy) # position previous -> above -> or horizontal
            if strategy == 'local' :
                if score_matrix[i][j] > score_matrix[top_i][top_j] or (score_matrix[i][j] == score_matrix[top_i][top_j] and j > top_j):
                    top_i, top_j  = i, j
            elif strategy == 'semiglobal' : 
                if (j == N-1 or i == M-1) and (score_matrix[i][j] > score_matrix[top_i][top_j] or (score_matrix[i][j] == score_matrix[top_i][top_j] and j > top_j)):
                    l_cand.append(score_matrix[i][j])
                    top_i, top_j  = i, j
    if strategy == 'semiglobal':
        # eerste item laatste kolom, laatste item eerste kolom -> evt. nieuwe top_i, top_j -> dan print hij gehele alignment
        if score_matrix[M-1][0] > score_matrix[top_i][top_j]:
            l_cand.append(score_matrix[i][j])
            top_i, top_j  = M-1, 0
        elif score_matrix[0][N-1] > score_matrix[top_i][top_j]:
            l_cand.append(score_matrix[i][j])
            top_i, top_j  = 0, N-1
    print(l_cand)
        
    #####################
    #  END CODING HERE  #
    #####################   
    
    
    ### 3: Traceback
    
    #####################
    # START CODING HERE #
    #####################   
    def traceback(sm, i, j, strategy):
        """
        Traces back the optimal path from specified position (i, j) it 
        determines the previous step (based on high-road hierarchy) and
        retrieves the amino acids for X and Y at that position as well as 
        the coordinates of the previous position (prev_i, prev_j)
        
        Input:
            i = row nr (int) in score matrix (sm)
            j = column nr (int) in score matrix (sm)
            sm = score matrix (list of lists)
            
        Output:
            aa1 = current amino acid or gap from seq1
            aa2 = current amino acid or gap from seq2
            prev_i = previous row nr (int) in score matrix (sm)
            prev_j = previous column nr (int) in score matrix (sm)
        """
        g = gap_penalty
        prev_i, prev_j, aa1, aa2 = 0, 0, '-', '-'
        if (sm[i][j] == sm[i-1][j] - g) and i > 0: # above
            aa1, aa2 =  seq1[i-1], '-'
            prev_i, prev_j = [i-1,j]
        elif (sm[i][j] == sm[i-1][j-1] + substitution_matrix[seq1[i-1]][seq2[j-1]]) and i > 0 and j > 0:
            aa1, aa2 = seq1[i-1], seq2[j-1]
            prev_i, prev_j = [i-1,j-1]
        elif (sm[i][j] == sm[i][j-1] - g) and j > 0: # left
            prev_i, prev_j =  [i,j-1]
            aa1, aa2 = '-', seq2[j-1]
        else : 
            if strategy == 'local': # if zero
                aa1, aa2 = seq1[i-1], seq2[j-1]
                prev_i, prev_j = [-1,-1]
            elif strategy == 'semiglobal':
                if i != 0 and j == 0:
                    aa1, aa2 = seq1[i-1], '-'
                    prev_i, prev_j = [i-1,0]
                elif i == 0 and j != 0: 
                    aa1, aa2 = '-', seq2[j-1]
                    prev_i, prev_j = [0,j-1]
        return aa1, aa2, prev_i, prev_j
        
    def assess_path(strategy):
        """
        
        d_start = Starting point differs based on the preferred strategy:
            global = bottom right (N-1, M-1)
            local = cell with highest score (top_i, top_j)
        """
        aligned_seq1 = ''  # These are dummy values! Change the code so that
        aligned_seq2 = ''
        d_start = {'local': [top_i, top_j], 'global': [M-1, N-1], 'semiglobal': [M-1, N-1]} # top_i, top_j
        n_i, n_j = d_start[strategy]
        #print(score_matrix[1][2])
        if strategy == 'semiglobal':
            align_score = score_matrix[top_i][top_j]
            if n_i >= top_i: 
                aligned_seq2 += '-'*(n_i-top_i)
                aligned_seq1 = seq1[top_i:n_i+1][::-1]
            if n_j >= top_j:
                aligned_seq1 += '-'*(n_j-top_j)
                aligned_seq2 = seq2[top_j:n_j+1][::-1]
            n_i, n_j = top_i, top_j
        else : 
            align_score = score_matrix[n_i][n_j] # bottom-right
        #print(top_i, top_j)
        #print(aligned_seq1, '\n' , aligned_seq2)
        while n_i > 0 or n_j > 0 :
            #print(n_i, n_j)
            aa1, aa2, n_i, n_j = traceback(score_matrix, n_i, n_j, strategy)
            if n_i >= 0:
                aligned_seq1 += aa1
            if n_j >= 0:
                aligned_seq2 += aa2
            if n_i == 0 and n_j == 0 :
                break
        aligned_seq1, aligned_seq2 = aligned_seq1[::-1], aligned_seq2[::-1]
        if len(aligned_seq1) > len(aligned_seq2):
            aligned_seq2 += '-'*(len(aligned_seq1)-len(aligned_seq2))
        elif len(aligned_seq2) > len(aligned_seq1):
            aligned_seq1 += '-'*(len(aligned_seq2)-len(aligned_seq1))
        #print(aligned_seq1[::-1], aligned_seq2[::-1])
        return aligned_seq1, aligned_seq2, align_score
    #aligned_seq1 = 'foot'  # These are dummy values! Change the code so that
    #aligned_seq2 = 'bart'  # aligned_seq1 and _seq2 contain the input sequences
    #align_score = 0        # with gaps inserted at the appropriate positions.    
    aligned_seq1, aligned_seq2, align_score = assess_path(strategy)
    print(aligned_seq1, aligned_seq2, len(aligned_seq1), len(aligned_seq2))
    #####################
    #  END CODING HERE  #
    #####################   
    alignment = (aligned_seq1, aligned_seq2, align_score)
    return (alignment, score_matrix)
def print_score_matrix(s1,s2,mat):
    "Pretty print function for a score matrix."
    
    # Prepend filler characters to seq1 and seq2
    s1 = '-' + s1
    s2 = ' -' + s2
    
    # Print them around the score matrix, in columns of 5 characters
    print(''.join(['%5s' % aa for aa in s2])) # Convert s2 to a list of length 5 strings, then join it back into a string
    for i,row in enumerate(mat):               # Iterate through the rows of your score matrix (and keep count with 'i').
        vals = ['%5i' % val for val in row]    # Convert this row's scores to a list of strings.
        vals.insert(0,'%5s' % s1[i])           # Add this row's character from s2 to the front of the list
        print(''.join(vals))                   # Join the list elements into a single string, and print the line.
def print_alignment(a):
    "Pretty print function for an alignment (and alignment score)."
    
    # Unpack the alignment tuple
    seq1 = a[0]
    seq2 = a[1]
    score = a[2]
    
    # Check which positions are identical
    match = ''
    for i in range(len(seq1)): # Remember: Aligned sequences have the same length!
        match += '|' if seq1[i] == seq2[i] else ' ' # Fancy inline if-else statement. Use cautiously!
            
    # Concatenate lines into a list, and join them together with newline characters.
    print('\n'.join([seq1,match,seq2,'','Score = %i' % score]))
def save_alignment(a,f):
    "Saves two aligned sequences and their alignment score to a file."
    with open(f,'w') as out:
        out.write(a[0] + '\n') # Aligned sequence 1
        out.write(a[1] + '\n') # Aligned sequence 2
        out.write('Score: %i' % a[2]) # Alignment score
    
def save_score_matrix(m,f):
    "Saves a score matrix to a file in tab-separated format."
    with open(f,'w') as out:
        for row in m:
            vals = [str(val) for val in row]
            out.write('\t'.join(vals)+'\n')
    
def main(args = False):
    # Process arguments and load required data
    if not args: args = parse_args()
    
    sub_mat = load_substitution_matrix(args.substitution_matrix)
    seq1, seq2 = load_sequences(args.fasta)
    # Perform specified alignment
    strat = args.strategy
    gp = args.gap_penalty
    alignment, score_matrix = align(seq1, seq2, strat, sub_mat, gp)
    # If running in "verbose" mode, print additional output
    if args.verbose:
        print_score_matrix(seq1,seq2,score_matrix)
        print('') # Insert a blank line in between
        print_alignment(alignment)
    
    # Save results
    if args.align_out: save_alignment(alignment, args.align_out)
    if args.matrix_out: save_score_matrix(score_matrix, args.matrix_out)
if __name__ == '__main__':
    main()
