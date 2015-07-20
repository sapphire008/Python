# -*- coding: utf-8 -*-
"""
Call SNPs for EM
"""

import csv, os #, pickle
from optparse import OptionParser
#from operator import itemgetter

# These environment variables point to the location of the tools,
# created temporarily by the script calls upon it.
# Can also be set permanently with an absolute path
samtools = os.getenv("PATH_TO_SAMTOOLS")#"/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/samtools-1.1/samtools"
bcftools = os.getenv("PATH_TO_BCFTOOLS")#"/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/bcftools-1.1/bcftools"
SnpSift = os.getenv("PATH_TO_SNPSIFT")#"/mnt/NFS/homeG2/LaFramboiseLab/dxc430/Software/snpEff/SnpSift.jar"

#os.system("echo %s" %(samtools))
#os.system("echo %s" %(bcftools))
#os.system("echo %s" %(SnpSift))

"""
Parse optional arguments to the script
"""
usage = "usage: %prog [options] ref_genome norm_bam tum_bam result_dir"
parser = OptionParser(usage)
parser.add_option("-r","--chr2user", dest="chr2use", help="chromosome to use, e.g. chr7", default=None)
parser.add_option("-s","--step", dest="continue_at", help="stepper, used in case program crash; continue at step: 1. Call SNPs; 2. Make bedfiles; 3.Pileup; 4. Counting Reads; 5. Consolidation; Default 0", default=0)
parser.add_option("-n","--normal_only", dest="call_normal_only", help="only call normal vcf, all subsequent call follows normal heterozygosity", action="store_true")
options, args = parser.parse_args()
if len(args) != 4:
    parser.error("Wrong number of arguments")
    parser.print_help()
ref_genome = args[0]
bam_files = args[1:3]
result_dir = args[3]
continue_at= options.continue_at
if options.chr2use is not None:
    chr2use = "-r " + options.chr2use
else:
    chr2use = ""
#os.system("echo here")
# os.system("echo %s" %options.chr2use)
#print(options.call_normal_only)
#import sys
#sys.exit()
# create index files if does not exist
os.system("mkdir -p %s" %(result_dir))
for i, bam in enumerate(bam_files):
    if not os.path.isfile(bam+".bai"):
      tmp_bam = os.path.join(result_dir,os.path.basename(bam))
      os.system("ln -s %s %s" %(bam, tmp_bam))
      cmd = "%s index %s" %(samtools, tmp_bam)
      bam_files[i] = tmp_bam
      os.system(cmd)

# Step 1: Call SNPs, make snp.vcf files
vcf_files = []
print("Step 1: Call SNPs ...")
for i, bam in enumerate(bam_files):
    vcf_files.append(os.path.join(result_dir, os.path.basename(bam).split(".")[0]+".vcf"))
    if continue_at > 1:
        print("Skip")
        continue
    if i>1 and options.call_normal_only:
        break
    if not os.path.isdir(result_dir):
        os.system('mkdir %s' % result_dir)
    cmd = '%s mpileup -v -I -B -C 50 -q 40 -Q 30 -f %s %s %s | %s call -v -c - | java -jar %s filter "isHet( GEN[0] )" > %s' %(samtools, ref_genome, chr2use, bam, bcftools, SnpSift, vcf_files[i])
    print(cmd)
    os.system(cmd)

# Step 2: Make bedfiles and chromomosome dictionaries
bed_files = []
chrom_dict =[]
print("Step 2: Make bedfiles and chromosome dictionaries ...")
for i, vcf in enumerate(vcf_files):
    bed_files.append(os.path.join(result_dir, os.path.basename(vcf).split(".")[0]+".bed"))
    if continue_at <= 2:
        fh = open(bed_files[i], 'wb') # <-- bypass
        fh_write = csv.writer(fh, delimiter='\t') # <-- bypass
    else:
        print("Skip")
    chrom = {}
    with open(vcf, "rb") as csvfile:
        fid = csv.reader(csvfile, delimiter='\t')
        for i, row in enumerate(fid):
            if row[0][0] == "#":
                continue
            my_row = [row[0], int(row[1])-1, int(row[1])]
            if continue_at <= 2:
                fh_write.writerow(my_row) # <-- bypass
            chrom[(row[0], row[1])] = row[3], row[4]
    csvfile.close()
    if continue_at <= 2:
        fh.close() # <-- bypass
    chrom_dict.append(chrom)
#with open(os.path.join(result_dir,"chrom_dict.pkl"), 'wb') as pickle_save:
#    pickle.dump(chrom_dict, pickle_save)

# Step 3: pileup
pileup_files = []
print("Step 3: Pileup ...")
for i, bam in enumerate(bam_files):
    # broadcast bed files to generate pileups for each bam file
    pileup_files.append(os.path.join(result_dir,os.path.basename(bam).split(".")[0] + ".pileup"))
    if continue_at > 3:
        print("Skip")
        continue
    if len(bed_files) == 1:
        bi = 0
    else:
        bi = i
    cmd = "%s mpileup -B -C 50 -q 40 -Q 30 -f %s %s -l %s %s > %s" %(samtools, ref_genome, chr2use, bed_files[bi], bam, pileup_files[i])
    print(cmd)
    os.system(cmd) # <-- bypass

# counts the alleles from pileup
def count_alleles(test_str, ref):
    count_dict = {"A":0, "T":0, "C":0, "G":0, "IN":0, "DEL":0}
    pm_prev = False #keeps track of whether it is in indel
    indel_count = 0

    for let in test_str:
        if pm_prev == True:
            indel_count = int(let)
            pm_prev = False
        elif indel_count > 0:
            indel_count -= 1
            continue
        else:
            if let in '.,':
                count_dict[ref] += 1
            elif let == '$':
                continue
            elif let in "-+":
                pm_prev = True
                if let in '-':#AFTER ANALYSIS, indels count for ref as well!
                    count_dict["DEL"] += 1 #doesn't matter if in or del
                else:
                    count_dict["IN"] += 1
            elif let == "^":
                indel_count = 1
            elif let == "*":
                    count_dict["DEL"] += 1 #see samtools doct (not format pg)
            elif let in "ACGTacgt" and indel_count == 0:
                upper_let = let.upper()
                count_dict[upper_let] += 1
            else:
                #pass
                print test_str
                print "fell through... check into this"

    return count_dict

# Step 4: Counting Reads
#pileup_files=["C:\Users\Edward\Desktop\NGS13_out6_hg19.pileup",
#"C:\Users\Edward\Desktop\NGS14_out6_hg19.pileup"]
print("Step 4: Counting Reads ...")
tsv_files = []
for i, f in enumerate(pileup_files):
    tsv_files.append(f.replace(".pileup", ".tsv"))
    if continue_at > 4:
        print("Skip")
        continue
    infile_pile = open(f, "rb")
    intabfile_pile = csv.reader(infile_pile, delimiter = '\t')
    outfile = open(tsv_files[i], "wb") #<-- bypass
    outtabfile = csv.writer(outfile, delimiter = '\t') #<--bypass

    print "opening " + f

    rows = []

    #make dict for allele values from pileup
    #since it is from the pileup, never gonna be missing a value in the try except part
    allele_dict = {}
    if len(vcf_files) == 1:
        bi = 0
    else:
        bi = i

    for j, pile_line in enumerate(intabfile_pile):
        chrom, position, reference, = pile_line[0], pile_line[1], pile_line[2]
        ct_dict = count_alleles(pile_line[4], pile_line[2].upper())
        temp = [chrom, position, chrom_dict[bi][chrom, position][0], chrom_dict[bi][chrom,position][1]] # chrom, position, reference, alternative
        try:
            temp += [ct_dict[k] for k in ['A','T','C','G','IN','DEL']]
        except:
            # print "missing a value"
            temp += ["."]*6
        allele_dict[(chrom, position)] = temp

    for key, value in allele_dict.iteritems():
        rows.append(value)

    outtabfile.writerows(rows) #<--bypass

    #infile.close()
    infile_pile.close()
    outfile.close() #<--bypass

#tsv_files = ["C:\Users\Edward\Desktop\snp_norm11.pileup.norm_tum_11_12_snp.tsv",
#"C:\Users\Edward\Desktop\snp_tum12.pileup.norm_tum_11_12_snp.tsv"]

# Step 5: Consolidate
variant_dict = {}
snp_dict = {}
print("Step 5: Consolidating ...")
for i, extract_file in enumerate(tsv_files):
    print "Consolidating " + extract_file
    infile = open(extract_file, "rb")
    intabfile = csv.reader(infile, delimiter = '\t')
    # write a description line at the top for understanding

    #extract the sample information from the two types of files: assuming the
    #first file specified is normal, and second file specified is tumor
    if i == 0:
        mytype = "normal"
    else:
        mytype = "tumor"

    for line in intabfile:
        chrom, pos, ref_base, alt_base, a_ad, t_ad, c_ad, g_ad, in_ad, del_ad  = line
        key = chrom, int(pos)
        val_var = [ref_base, alt_base, a_ad, t_ad, c_ad, g_ad, in_ad, del_ad]
        base_dict = {'A':a_ad, 'T':t_ad,'C':c_ad,'G':g_ad, 'IN':in_ad, 'DEL':del_ad}
        snp_row = [ref_base, alt_base, base_dict[ref_base], str(sum([int(base_dict[bs]) for bs in alt_base.split(',')]))]
        if key in variant_dict:
            if mytype == "normal":
                variant_dict[key] = val_var + variant_dict[key][8:]
                snp_dict[key] = snp_row + snp_dict[key][4:]
            else: #tumor
                variant_dict[key] = variant_dict[key][:8] + val_var
                snp_dict[key]= snp_dict[key][:4] + snp_row
        else:
            if mytype == "normal":
                variant_dict[key] = val_var + ["."]*8
                snp_dict[key] = snp_row + ["."]*4
            else:
                variant_dict[key] = ["."]*8 + val_var
                snp_dict[key] = ["."]*4 + snp_row

    infile.close()
#with open(os.path.join(result_dir,"variant_dict.pkl"), 'wb') as variant_pickle:
#    pickle.dump(variant_dict, variant_pickle)
#with open(os.path.join(result_dir,"snp_dict.pkl"), 'wb') as snp_pickle:
#    pickle.dump(snp_dict, snp_pickle)

#filters for variants only present in both... Should it be only one?
def snp_dict2csv(mydict, target_dir, header = [], row_filter = '.'):
    # sort the dictionary first
    with open(target_dir, "wb") as csvfile:
        outfile = csv.writer(csvfile, delimiter = '\t')
        if not header:
            outfile.writerows([header])
        for key, val in sorted(mydict.iteritems()): # filter out any entries that have missing SNPs
            if any(a == row_filter for a in val):
                continue
            outfile.writerow(list(key)+val)
        csvfile.close()


header = ["chr", "pos", "ref_norm", "alt_norm", "A_norm","T_norm","C_norm","G_norm","INS_norm","DEL_norm",
                            "ref_norm", "alt_tum", "A_tum", "T_tum", "C_tum", "G_tum", "INS_tum", "DEL_tum"]
snp_dict2csv(variant_dict, os.path.join(result_dir, "shared_snp.txt"), header)

header = ["chr","pos","ref_norm", "alt_norm", "ref_norm_count", "alt_norm_count",
          "ref_tum","alt_tum", "ref_tum_count", "alt_tum_count"]
snp_dict2csv(snp_dict, os.path.join(result_dir, "shared_snp_count.txt"), header)


#mydict = {('ch3','23234'):1,('ch3','232d434'):3,('ch4','839293'):2,('ch1','339293'):2}
#for k, v in sorted(mydict.iteritems()):
