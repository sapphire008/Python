# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 14:03:50 2014

EM: proportion of tumor cells in a sample read

@author: Edward
"""

import numpy as np
from numpy import log
from numpy import sum as nsum
from scipy.misc import comb as nchoosek
from scipy.optimize import fminbound
import csv, sys
from optparse import OptionParser

def EM_parse_opt():
    """
    Parse optional arguments to the script
    """
    usage = "usage: %prog [options] shared_snp_count1.txt shared_snp_count2.txt ..."
    parser = OptionParser(usage)
    parser.add_option("-r","--chr2use", dest="chr2use", help="chromosome to use, e.g. chr2,chr7", default=None)
    parser.add_option("-m","--mutation", dest="mutation", help="type of mutation, either UPD or DEL, or specify the copy number in amplification; default DEL", default='DEL')
    parser.add_option("-p","--position", dest="position_range", help="range of positions on chromosome", default=None)
    parser.add_option("-N", "--atleastN", dest="N_at_least", help="filter out SNPs conunts less than N", default=0)
    parser.add_option("-k","--maxiter", dest="maxiter", help="maximum number of iterations to run EM. Default 1000", default=1000)
    parser.add_option("-o","--output", dest="output", help="write to an output file, instead of stdout", default=None)
    parser.add_option("-a","--append", dest="append", help="append intead of overwriting the output file", default="store_false")
    parser.add_option("-i","--id", dest="id", help="assign subject id to each input file, e.g. subj01,subj02,subj03", default=None)
    options, args = parser.parse_args()
    if len(args) < 1:
		parser.error("Input at least 1 SNP count file required")
    return(options, args)

def p_read(theta, N, X, B, mutation):
    # Parse the mutation string
    if isinstance(mutation,(str)):
        if mutation.isdigit():
            mutation = float(mutation)
        else:
            mutation = mutation.upper()
            # turn string input into numeric cases
            mutation = {
                'DEL': 1.0,
                'UPD': 2.0
            }.get(mutation,mutation)
    # raise exception if the result is not numeric
    if not isinstance(mutation,(int, long, float)):
        raise Exception("Unrecognized mutation type %s" %(mutation))
    # probability of reads based on type of mutation
    if mutation == 1.0: # DEL
        p_A = B / (1.0-theta+B*theta)
        p_B = (B-B*theta) / (1.0-B*theta)
    elif mutation == 2.0: # UPD
        p_A = (B+B*theta) / (1.0-theta+2.0*B*theta)
        p_B = (B-B*theta) / (1.0+theta-2.0*B*theta)
    elif mutation > 2.0: # AMPLIFICATION
        p_A = ((mutation-2.0)*B*theta+theta) / ((mutation-2.0)*B*theta+1.0)
        p_B = ((mutation-1.0)*theta+(2.0-mutation)*B*theta) / ((mutation-2.0)*theta+(2.0-mutation)*B*theta+1.0)
    else: # catch other invalid copy numbers / mutations input
        raise Exception("Invalid copy number mutation %d" %(mutation))
    f_A = nchoosek(N,X) * (p_A**X) * ((1.0-p_A)**(N-X))
    f_B = nchoosek(N,X) * (p_B**X) * ((1.0-p_B)**(N-X))
    return (p_A, p_B, f_A, f_B)

def EM_Clonal_Abundance(N, X, B=0.5, mutation='DEL', theta_tol=1E-9, maxiter=1000, consecutive_convergence= 10, full_output = False, disp=False):
    """
    Inputs:
        N: a vector of total number of reads for each SNP
        X: a vector of total number of allele A reads, same length as N
        B: a vector of read bias, estimated by the number of reads of allele A 
           divided by the total number of reads in normal sample. Default is .5
        mutation: type of mutation ['DEL' | 'UPD' | {numeric}], chromosomal 
            deletion, uniparental disomy, or copy number (total number of 
            copies of the chromosomes) in the case of amplification. 
            Default 'DEL'.
        theta_tol (optional): tolerance of theta change for convergence, 
            default is 1E-9
        maxiter: maximum number of iterations to run, default is 1000
        consecutive_convergence: number of times that the change of theta has to be
            less than that of theta_tol, consecutively, to be deemed convergence
        full_output: flag to return additional outputs
        disp: display each iteration of EM
    
    Output:
        theta: estimated proprotion of tumor cells (between 0 and 1)
        
    If full_output is set to True, the following can also be returned
        it_count: number of iterations used
        
    """
    
    # Initialize / guess parameters
    theta = 0.5
    d_theta = 1.0
    it_count = 0
    d_theta_count = 0
    
    z_A = X/N # probability of z_A
    z_B = 1.0-z_A #probability of z_B
    
    # define objective functions for theta
    # Maximize theta so that the log likelihood is maximized
    # l(theta) = sum_i[sum_zi: Qi_zi*log(p(x_i, z_i; theta)/Qi_zi)]
    # l(theta) = sum_i[z_Ai*log(r_A) + z_Bi*log(r_B) + log(z_Ai*f_A + z_Bi*f_B)]
    def likelihood_func(theta, N, X, B, z_A, z_B, mutation):
        # Likelihood function for theta
        _, _, f_A, f_B = p_read(theta, N, X, B, mutation)
        l = -1.0*nsum(log(z_A*f_A + z_B*f_B))
        return l
    def likelihood_func_deriv(theta, N, X, B, z_A, z_B, mutation):
        # derivative of the likelihood of the function for theta
        p_A, p_B, f_A, f_B = p_read(theta, N, X, B, mutation)
        p_A_deriv = B * (1.0-B) * ((1-theta+B*theta) **(-2.0))
        p_B_deriv = ((1.0-B*theta) * (-B) - (B-B*theta) * (-B))/ ((1.0-B*theta)**2)
        f_A_deriv = nchoosek(N,X) * X *(p_A_deriv) * ((1.0-p_A)**(N-X)) + nchoosek(N,X) * (p_A**X) * (N-X) * (-p_A_deriv)
        f_B_deriv = nchoosek(N,X) * X *(p_B_deriv) * ((1.0-p_B)**(N-X)) + nchoosek(N,X) * (p_B**X) * (N-X) * (-p_B_deriv)
        l_deriv = -1.0*nsum(1.0/(z_A*f_A + z_B*f_B) * (z_A * f_A_deriv + z_B * f_B_deriv))
        return l_deriv
    
    while d_theta_count <= consecutive_convergence and it_count <= maxiter:
        it_count += 1
        # M-Step
        
        # r_A, r_B part
        # r_A = sum_i2K(z_Ai/N)
        r_A = nsum(z_A / np.size(z_A))
        r_B = nsum(z_B / np.size(z_B))

        # Maximize the log likelihood to estimate for theta, (minimize the 
        # negative of the log-likelihood)
        xopt, fval, ierr, numfunc =  fminbound(likelihood_func, 0.0, 1.0, args=(N, X, B, z_A, z_B, mutation), full_output=True)
        
        if disp:
            print("theta:%f, fval:%f, ierr:%d, numfunc:%d" %(xopt, fval, ierr, numfunc))
        # returns a new theta
        d_theta = np.abs(xopt - theta)
        theta = xopt
        
        if d_theta<theta_tol:
            d_theta_count += 1
        else:# if not consecutive convergence, set convergence count to zero
            d_theta_count = 0 
        
        # E-Step
        # Set Q_i(Z_i) = p(z_i =1 | x_i ; theta)
        
        # Recalculate probabilities        
        _, _, f_A, f_B = p_read(theta, N, X, B, mutation)
        f_X = r_A * f_A + r_B * f_B
        
        z_A = r_A * f_A / f_X
        z_B = 1.0 - z_A
        # end of while loop
    if it_count > maxiter and d_theta_count < 1:
        print("Theta not converged!")
    if full_output:
        return (theta, it_count)
    else:
        return theta

# Read in files
def EM_read_shared_snp_file(shared_snp_file, chr2use=None, position_range=None, 
                            pseudo_count=True, print_result=False, N_at_least=0, delimiter='\t'):
    """
    Read in an SNP call file, and calcualte parameters necessary for EM
    Inputs:
        shared_snp_file: delimited file (default tab delimited) with each row:
            chromosome, SNP position, reference normal, alternative normal, 
            reference normal count, alternative normal count, reference tumor,
            alternative tumor, reference tumor count, alternative tumor count
        chr2use: filter chromosomes to retain in the analysis
        position_range: range of SNP position [min, max]
        pseudo_count: avoid zero reads when calcualting bias term by adding all
            the counts by 1 (True | False)
        print_result: print (filtered) result of calculations for each SNP
        N_at_least: only read in SNPs with total count at least this number.
            Default 0, read in everything
        delimiter: default '\t' tab delimited
    Outputs: return the following as a tuple
        ch: chromosome name
        pos: position of SNP
        N: total number of reads in tumor cells
        X: total number of read of reference in tumor cells
        B: bias of reaading according to normal cell SNPs
    """
    ch, pos, N, X, B = [], [], np.array([]), np.array([]), np.array([])
    with open(shared_snp_file, 'rb') as FID:
        csvfile = csv.reader(FID, delimiter=delimiter)
        for i, row in enumerate(csvfile):
            c, p, ref_norm, alt_norm, ref_norm_count, alt_norm_count, ref_tum, alt_tum, ref_tum_count, alt_tum_count = row
            p, ref_norm_count, alt_norm_count, ref_tum_count, alt_tum_count = int(p), float(ref_norm_count), float(alt_norm_count), float(ref_tum_count), float(alt_tum_count)
            # apply filter
            if (chr2use is not None) and (c not in chr2use):
                continue
            if (position_range is not None) and (position_range[0] > p or position_range[1] < p):
                continue
            if (ref_tum_count + alt_tum_count < N_at_least):
                continue
            if (ref_norm_count<1E-6 or alt_norm_count<1E-6) and pseudo_count:
                ref_norm_count, alt_norm_count = ref_norm_count+1.0, alt_norm_count+1.0
            ch.append(c)
            pos.append(int(p))
            N = np.append(N, ref_tum_count + alt_tum_count)
            X = np.append(X, ref_tum_count + pseudo_count)
            B = np.append(B, ref_norm_count / (ref_norm_count + alt_norm_count))
            if print_result:
                print("%s, %d, N:%d, X:%d, B:%f\n" %(ch[-1], pos[-1], N[-1], X[-1], B[-1]))
        # give warnings if reading is very biased
        if any(B < 1E-6) or any(B > 1-1E-6):
            print('Some readings are very biased: %s' %(B))
    FID.close()
    return (ch, pos, N, X, B)
    
def EM_main2():
    """
    Estimate tumor percent given data
    """
    options, file_lists = EM_parse_opt()
    if options.chr2use is not None:
        options.chr2use = list([r for r in options.chr2use.split(',')])
    if options.position_range is not None:
        options.position_range=np.array([int(p) for p in options.position_range.split(',')])
    if options.id is None:
        options.id = np.arange(1,len(file_lists)+1, 1)
    else:
        options.id = list([sid for sid in options.id.split(',')])
        # check if there are the same number of ids as input files
        if len(file_lists) != len(options.id):
            # use numerical indices
            options.id = np.arange(1,len(file_lists)+1, 1)
    if options.output is not None:
        if options.append:
            FID  = open(options.output,'a')
        else:
            FID = open(options.output, 'wb')
        csvfile = csv.writer(FID,delimiter='\t')
        csvfile.writerow(["subject","nSNPs", "chromosome","mutation","theta"])
    else:
        sys.stdout.flush() # make sure it prints to shell
        print("\t".join(["subject","nSNPs", "chromosome","mutation","theta"]))
    for s, f in enumerate(file_lists):
        ch, pos, N, X, B = EM_read_shared_snp_file(f, options.chr2use, options.position_range, pseudo_count=True, print_result=False, N_at_least=int(options.N_at_least))    
        theta, it_count = EM_Clonal_Abundance(N, X, B, mutation = options.mutation, theta_tol=1E-9, maxiter=1000, full_output = True, disp=False)
        if options.output is None:
            print("\t".join([options.id[s], np.size(ch), ','.join(np.unique(ch)), options.mutation, theta]))
        else:
            csvfile.writerow([options.id[s], np.size(ch), ','.join(np.unique(ch)), options.mutation, theta])
    if options.output is not None:
        FID.close()

# Suppress output if imported as module
if __name__=="__main__":
    EM_main2()
    
    
    
    
    
